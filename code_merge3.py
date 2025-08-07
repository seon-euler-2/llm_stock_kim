import streamlit as st
import praw
import ssl
import re
import requests
import pandas as pd
from collections import Counter, defaultdict
import yfinance as yf
import io
import warnings
import contextlib
from urllib3.exceptions import InsecureRequestWarning

from curl_cffi.requests import Session
import yfinance as yf
import pandas as pd
import numpy as np
import ffn
from langchain_core.tools import tool
# from duckduckgo_search import DDGS
from openai import OpenAI
import os
import html
import urllib.parse
import requests
from bs4 import BeautifulSoup
from googlesearch import search  # google 라이브러리 사용
import requests
from bs4 import BeautifulSoup
import os
from pytrends.request import TrendReq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from no_ssl_verification import no_ssl_verification
load_dotenv()
import time
from collections import Counter
import praw
from openai import OpenAI
from dotenv import load_dotenv
import os
from no_ssl_verification import no_ssl_verification
# .env 파일에서 환경 변수 로드
load_dotenv()

import pandas as pd
import os
import numpy as np
import openai
import json 
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from curl_cffi.requests import Session
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


# SSL 인증서 우회
ssl._create_default_https_context = ssl._create_unverified_context
old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
    requests.Session.merge_environment_settings = merge_environment_settings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings
        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

session = Session(
        impersonate="chrome110",  # optional, mimics Chrome browser
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        verify=False  # optional; disable SSL verify if you're behind a proxy
    )



# 미국 전체 상장 종목 자동 로딩
@st.cache_data
def get_all_us_tickers():
    with no_ssl_verification():
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        nasdaq_df = pd.read_csv(io.StringIO(requests.get(nasdaq_url).text), sep="|")
        other_df = pd.read_csv(io.StringIO(requests.get(other_url).text), sep="|")

        nasdaq_df = nasdaq_df[~nasdaq_df.iloc[:, 0].str.contains("File Creation Time", na=False)]
        other_df = other_df[~other_df.iloc[:, 0].str.contains("File Creation Time", na=False)]

        nasdaq_col = next((col for col in nasdaq_df.columns if "Symbol" in col), None)
        other_col = next((col for col in other_df.columns if "ACT Symbol" in col), None)

        if not nasdaq_col or not other_col:
            raise ValueError("❌ 티커 컬럼을 찾을 수 없습니다.")

        tickers = pd.concat([nasdaq_df[nasdaq_col], other_df[other_col]]).dropna().unique().tolist()
        return set(tickers)

# 종목 추출, 시가총액 포맷, 주가 정보 수집
def extract_tickers_from_text(text, valid_tickers):
    words = re.findall(r'\b[A-Z]{1,5}\b', text.upper())
    return [word for word in words if word in valid_tickers]


def filter_relevant_tickers_with_gpt(title: str, body: str, tickers: list) -> list:
    """
    추출된 ticker 중 실제 투자 관련 맥락에서 언급된 것만 GPT로 필터링
    """
    if not tickers:
        return []

    system_msg = {
        "role": "system",
        "content": """다음은 Reddit의 투자 관련 게시글입니다.
제목과 본문을 읽고, 제시된 종목명 리스트 중 실제 투자 맥락에서 언급된 종목만 필터링하세요.

다음과 같은 형식으로 응답하세요:
["TSLA", "NVDA"]
"""
    }

    user_msg = {
        "role": "user",
        "content": f"제목: {title}\n본문: {body}\n종목 리스트: {tickers}"
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    except Exception as e:
        print(f"GPT 필터링 실패: {e}")
        return []

def format_market_cap(value):
    if not value:
        return "N/A"
    for suffix, factor in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if value >= factor:
            return f"${value / factor:.2f}{suffix}"
    return f"${value:.2f}"

def get_detailed_info(ticker):
    try:
        stock = yf.Ticker(ticker, session= session)
        info = stock.info
        data = stock.history(period="2d")
        if data.shape[0] < 2:
            return None
        price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2]
        change_pct = ((price - prev_price) / prev_price) * 100
        return {
            "ticker": ticker,
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None)
        }
    except Exception:
        return None

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
# ────────────────────────────────────────────────
st.title("🧠 Reddit 종목 언급 + GPT 기반 필터링")

with st.sidebar:
    st.header("설정")
    subreddit_input = st.text_input("Subreddit 이름", "wallstreetbets")
    post_limit = st.slider("게시물 수", 10, 100, 30)
    top_n = st.slider("상위 종목 수", 5, 20, 10)
    if st.button("🚀 실행"):
        st.session_state["run"] = True

if "run" not in st.session_state:
    st.session_state["run"] = False

if st.session_state["run"]:
    try:
        tickers_set = get_all_us_tickers()

        reddit = praw.Reddit(
            client_id="RVKUBtrh7ExzRSbddfBDtg",
            client_secret="cT4m_YrJnZhZpZ2vPkVTfMT8hqj07A",
            user_agent="retail_stock_v1.0 (by /u/TraditionalIce9098)",
            check_for_async=False
        )
        reddit._core._requestor._http.verify = False

        st.info(f"📡 r/{subreddit_input}에서 게시물 수집 중...")
        posts = list(reddit.subreddit(subreddit_input).hot(limit=post_limit))
        st.success(f"{len(posts)}개 게시물 수집 완료")

        ticker_counter = Counter()
        ticker_posts = defaultdict(list)

        for post in posts:
            combined_text = (post.title or "") + " " + (post.selftext or "")
            raw_tickers = extract_tickers_from_text(combined_text, tickers_set)
            filtered_tickers = filter_relevant_tickers_with_gpt(post.title, post.selftext, raw_tickers)

            for ticker in filtered_tickers:
                ticker_counter[ticker] += 1
                ticker_posts[ticker].append({
                    "title": post.title or "[제목 없음]",
                    "url": f"https://www.reddit.com{post.permalink}",
                    "body": post.selftext or ""
                })

        top_tickers = ticker_counter.most_common(top_n)

        st.subheader(f"📊 Reddit 언급 종목 Top {top_n}")
        if top_tickers:
            for rank, (ticker, count) in enumerate(top_tickers, 1):
                info = get_detailed_info(ticker)
                if info:
                    change_symbol = "📈" if info["change_pct"] > 0 else "📉" if info["change_pct"] < 0 else "➡️"
                    market_cap = f"${info['market_cap']:,}" if info["market_cap"] else "N/A"
                    pe = f"{info['pe_ratio']:.2f}" if info["pe_ratio"] else "N/A"

                    st.markdown(
                        f"""
**{rank}. [{ticker}](https://finance.yahoo.com/quote/{ticker})** — 언급 {count}회  
> 🏢 **{info['name']}**  
> 💵 현재가: **${info['price']}** ({change_symbol} {info['change_pct']}%)  
> 🏷️ 섹터: {info['sector']} / {info['industry']}  
> 💰 시가총액: {market_cap}  
> 📊 PER: {pe}
""", unsafe_allow_html=True)

                    st.markdown("📌 **관련 게시글:**")
                    for i, post in enumerate(ticker_posts[ticker][:5], 1):
                        body = post.get("body", "")
                        body_preview = (body[:200] + "…") if len(body) > 200 else (body or "📄 본문 없음")
                        st.markdown(
                            f"""
<b>{i}. <a href="{post['url']}" target="_blank">{post['title']}</a></b><br>
<pre>{body_preview}</pre>
""", unsafe_allow_html=True)
        else:
            st.warning("❗ 종목명이 추출되지 않았습니다.")
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")