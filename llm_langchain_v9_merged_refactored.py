import requests
old_merge_environment_settings = requests.Session.merge_environment_settings


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
from bs4 import BeautifulSoup
from googlesearch import search  # google 라이브러리 사용
from pytrends.request import TrendReq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from no_ssl_verification import no_ssl_verification
load_dotenv()
import time
from collections import Counter
import praw
# from no_ssl_verification import no_ssl_verification
# .env 파일에서 환경 변수 로드
load_dotenv()
import io
import warnings
import contextlib
import openai
import json 
from requests.auth import HTTPBasicAuth

import streamlit as st

from code_merge3 import get_all_us_tickers, extract_tickers_from_text, filter_relevant_tickers_with_gpt, get_detailed_info
from collections import Counter, defaultdict

import ssl
import re
from urllib3.exceptions import InsecureRequestWarning

# from duckduckgo_search import DDGS
load_dotenv()
# .env 파일에서 환경 변수 로드
load_dotenv()




def draw_backtest_chart(df_all: pd.DataFrame):
    """
    백테스트 결과(df_all)를 Streamlit 차트로 시각화합니다.
    """
    if df_all is None or df_all.empty:
        st.warning("📭 시각화할 데이터가 없습니다.")
        return

    st.subheader("📈 누적 수익률 비교")
    st.line_chart(df_all)
    st.success("✅ 백테스트 누적 수익률이 시각화되었습니다.")
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

def run_reddit_summary(limit=20, subreddit="wallstreetbets") -> str:
    reddit = praw.Reddit(
        client_id="RVKUBtrh7ExzRSbddfBDtg",
        client_secret="cT4m_YrJnZhZpZ2vPkVTfMT8hqj07A",
        user_agent="retail_stock_v1.0 (by /u/TraditionalIce9098)",
        check_for_async=False
    )
    reddit._core._requestor._http.verify = False

    tickers_set = get_all_us_tickers()
    posts = list(reddit.subreddit(subreddit).hot(limit=limit))

    ticker_counter = Counter()
    ticker_posts = defaultdict(list)

    for post in posts:
        combined_text = (post.title or "") + " " + (post.selftext or "")
        raw_tickers = extract_tickers_from_text(combined_text, tickers_set)
        filtered = filter_relevant_tickers_with_gpt(post.title, post.selftext, raw_tickers)

        for ticker in filtered:
            ticker_counter[ticker] += 1
            ticker_posts[ticker].append({
                "title": post.title,
                "url": f"https://www.reddit.com{post.permalink}",
                "body": post.selftext
            })

    if not ticker_counter:
        return "❗ 유효한 종목 언급이 포함된 게시물이 없습니다."

    top = ticker_counter.most_common(10)
    result = "# 📊 Reddit 인기 종목 분석\n\n"
    for rank, (ticker, count) in enumerate(top, 1):
        info = get_detailed_info(ticker)
        if not info:
            continue
        symbol = "📈" if info["change_pct"] > 0 else "📉" if info["change_pct"] < 0 else "➡️"
        market_cap = info["market_cap"]
        mc = f"${market_cap:,}" if market_cap else "N/A"

        result += f"""
### {rank}. [{ticker}](https://finance.yahoo.com/quote/{ticker}) — {count}회 언급  
- 💵 현재가: ${info['price']} ({symbol} {info['change_pct']}%)  
- 🏷️ 섹터: {info['sector']} / {info['industry']}  
- 💰 시가총액: {mc}  
- 📊 PER: {info['pe_ratio'] or 'N/A'}  
- 🔗 관련 게시글:\n"""

        for post in ticker_posts[ticker][:3]:
            preview = post["body"][:100].replace("\n", " ") + "…" if post["body"] else ""
            result += f"  - [{post['title']}]({post['url']}) — {preview}\n"

        result += "\n"
    return result.strip()


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Reddit 인증 정보
reddit_client_id = "RVKUBtrh7ExzRSbddfBDtg"
reddit_client_secret = "cT4m_YrJnZhZpZ2vPkVTfMT8hqj07A"
reddit_user_agent = "retail_stock_v1.0 (by /u/TraditionalIce9098)"

# Reddit 클라이언트 생성
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent,
    check_for_async=False
)

# GPT를 사용해 관련 종목 추출
def extract_related_tickers(title, body, model="gpt-4o"):
    prompt = f"""
다음 Reddit 게시글 제목, 본문, 이미지 등에서 언급된 주요 주식 종목(symbol)을 최대 3개까지 추출하세요.
주식은 티커로 언급될 수도 있지만, 종목명(예: Microsoft;MSFT)으로 언급될 수도 있습니다.
미국 주식 기준으로 종목 코드(TSLA, GME 등)를 반환하고, 없으면 빈 리스트를 반환하세요.

제목: {title}
본문: {body[:1000]}

결과는 JSON 배열 형식으로만 주세요. 예: ["TSLA", "AAPL"]
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        tickers = eval(content) if content.startswith("[") else []
        return tickers
    except Exception as e:
        print(f"❌ GPT 오류: {e}")
        return []

llm = ChatOpenAI(model = 'gpt-4o')



openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

session = Session(
        impersonate="chrome110",  # optional, mimics Chrome browser
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        verify=False  # optional; disable SSL verify if you're behind a proxy
    )
def get_yf_close_prices(tickers: list[str], period: str) -> pd.DataFrame:
    ticker_objs = yf.Tickers(" ".join(tickers),  session= session)
    history_dict = {
        symbol: ticker_objs.tickers[symbol].history(period=period)
        for symbol in tickers
    }

    # 각 티커의 'Close' 시리즈를 모아 하나의 DataFrame으로
    close_df = pd.DataFrame({
        symbol: df["Close"]
        for symbol, df in history_dict.items()
        if "Close" in df.columns
    })
    close_df.index = close_df.index.strftime("%Y-%m-%d")
    close_df.index = pd.DatetimeIndex(close_df.index)
    return close_df

def get_returns_df(df, N=1, log=False):
    if log:
        return np.log(df / df.shift(N)).iloc[N-1:].fillna(0)
    else:
        return df.pct_change(N, fill_method=None).iloc[N-1:].fillna(0)

def get_cum_returns_df(return_df, log=False):
    if log:
        return np.exp(return_df.cumsum())
    else:
        return (1 + return_df).cumprod()    # same with (return_df.cumsum() + 1)

def get_CAGR_series(cum_rtn_df, num_day_in_year=250):
    cagr_series = cum_rtn_df.iloc[-1]**(num_day_in_year/(len(cum_rtn_df))) - 1
    return cagr_series

def get_sharpe_ratio(log_rtn_df, yearly_rfr = 0.025):
    excess_rtns = log_rtn_df.mean()*252 - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(252))

def get_drawdown_infos(cum_returns_df): 
    # 1. Drawdown
    cummax_df = cum_returns_df.cummax()
    dd_df = cum_returns_df / cummax_df - 1
 
    # 2. Maximum drawdown
    mdd_series = dd_df.min()

    # 3. longest_dd_period
    dd_duration_info_list = list()
    max_point_df = dd_df[dd_df == 0]
    for col in max_point_df:
        _df = max_point_df[col]
        _df.loc[dd_df[col].last_valid_index()] = 0
        _df = _df.dropna()

        periods = _df.index[1:] - _df.index[:-1]

        days = periods.days
        max_idx = days.argmax()

        longest_dd_period = days.max()
        dd_mean = int(np.mean(days))
        dd_std = int(np.std(days))

        dd_duration_info_list.append(
            [
                dd_mean,
                dd_std,
                longest_dd_period,
                "{} ~ {}".format(_df.index[:-1][max_idx].date(), _df.index[1:][max_idx].date())
            ]
        )

    dd_duration_info_df = pd.DataFrame(
        dd_duration_info_list,
        index=dd_df.columns,
        columns=['drawdown mean', 'drawdown std', 'longest days', 'longest period']
    )
    return dd_df, mdd_series, dd_duration_info_df

def get_rebal_dates(price_df, period="month"):
    _price_df = price_df.reset_index()
    if period == "month":
         groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.month]
    elif period == "quarter":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.quarter]
    elif period == "halfyear":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.month // 7]
    elif period == "year":
        groupby = [_price_df['Date'].dt.year, _price_df['Date'].dt.year]
    rebal_dates = pd.to_datetime(_price_df.groupby(groupby)['Date'].last().values)
    return rebal_dates

from functools import reduce

def calculate_portvals(price_df, weight_df):
    cum_rtn_up_until_now = 1 
    individual_port_val_df_list = []

    prev_end_day = weight_df.index[0]
    for end_day in weight_df.index[1:]:
        sub_price_df = price_df.loc[prev_end_day:end_day]
        sub_asset_flow_df = sub_price_df / sub_price_df.iloc[0]

        weight_series = weight_df.loc[prev_end_day]
        indi_port_cum_rtn_series = (sub_asset_flow_df * weight_series) * cum_rtn_up_until_now
    
        individual_port_val_df_list.append(indi_port_cum_rtn_series)

        total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
        cum_rtn_up_until_now = total_port_cum_rtn_series.iloc[-1]

        prev_end_day = end_day 

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    return individual_port_val_df

def get_backtest(ticker_list, period:str = '5y'):
    # Reddit에서 ticker get
    tickers = ticker_list
    df_close = get_yf_close_prices(tickers, period)
    index = ['SPY', 'QQQ']
    df_index = get_yf_close_prices(index, "5y")

    rebal_dates = get_rebal_dates(df_close, 'month')
    rebal_index = rebal_dates
    result_portval_dict = {} 
    n_assets = df_close.shape[1]
    target_ratios = np.array([1/n_assets] * n_assets)

    target_weight_df = pd.DataFrame(
        [[1/len(df_close.columns)]*len(df_close.columns)]* len(rebal_index),
        index=rebal_index,
        columns=df_close.columns
    )

    cum_rtn_at_last_month_end = 1
    individual_port_val_df_list = []

    individual_port_val_df = calculate_portvals(df_close, target_weight_df)
    individual_port_val_df.head()
    result_portval_dict['port'] = individual_port_val_df.sum(axis=1)

    import ffn
    pd.concat([pd.DataFrame(result_portval_dict), df_index], axis = 1).dropna().rebase().plot();
    df_all = pd.concat([pd.DataFrame(result_portval_dict), df_index], axis = 1).dropna().rebase()

        # 구간별
    stats = df_all.rebase().calc_stats()
    stats.display()
    # stats.stats.to_clipboard()

    # qs.reports.full(df_all.loc[:,'port'], df_all.loc[:, 'SPY']).to_markdown() 
    return df_all.to_markdown()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
# from graph_chart import draw_backtest_chart
from datetime import datetime
import pytz
from langchain_core.messages import HumanMessage
import streamlit.components.v1 as components
load_dotenv()



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







llm = ChatOpenAI(model = 'gpt-4o')

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

@tool
def get_yf_stock_history(ticker: str, period: str) -> str:
    """
    종목의 주가 이력을 조회해 텍스트로 반환합니다.
    시각화는 별도 도구에서 처리합니다.
    """
    stock = yf.Ticker(ticker, session= session)
    df = stock.history(period=period)

    if df.empty:
        return f"{ticker}의 {period} 기간 주가 데이터가 없습니다."

    # ✅ 시각화를 위해 데이터 저장
    st.session_state["latest_history_chart"] = df[["Close"]].copy()
    st.session_state["latest_history_chart"].index = st.session_state["latest_history_chart"].index.strftime("%Y-%m-%d")

    return df.tail().to_markdown()


@tool
def get_yf_cumulative_returns_tool(ticker_list: list, period: str = "3mo") -> str:
    """
    여러 종목의 누적 수익률 (Cumulative Returns)을 계산하여 비교 표로 반환합니다.
    """
    price_df = pd.DataFrame()

    for ticker in ticker_list:
        try:
            df = yf.Ticker(ticker, session=session).history(period=period)
            if df.empty:
                continue
            price_df[ticker] = df["Close"]
        except Exception as e:
            continue

    if price_df.empty:
        return "❗ 유효한 종목 가격 데이터를 가져올 수 없습니다."

    # 누적 수익률 계산
    rtn_df = price_df.pct_change().fillna(0)
    cum_rtn_df = (1 + rtn_df).cumprod()

    # 마지막 기준 수익률 요약
    final_returns = (cum_rtn_df.iloc[-1] - 1).sort_values(ascending=False) * 100
    summary_df = final_returns.to_frame(name="누적 수익률 (%)")
    summary_df.index.name = "종목"

    # ✅ 세션 상태 저장 (차트용)
    st.session_state["latest_cum_rtn_df"] = cum_rtn_df.copy()

    return f"📊 {period} 기간 누적 수익률 비교\n\n" + summary_df.round(2).to_markdown()





@tool
def get_yf_stock_info(ticker: str) -> str:
    """해당 종목의 Yahoo Finance 정보를 반환합니다."""
    stock = yf.Ticker(ticker, session= session)
    info = stock.info
    return str(info)

@tool
def get_yf_stock_recommendations(ticker: str) -> str:
    """해당 종목의 리서치 추천 정보를 반환합니다."""
    stock = yf.Ticker(ticker, session= session)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return f"{ticker}에 대한 추천 리서치 데이터가 없습니다."
    return recommendations.to_markdown()

@tool
def get_backtest_tool(ticker_list: list[str], period: str = "5y") -> str:
    """
    주어진 종목 리스트와 기간으로 백테스트를 실행하고,
    누적 수익률 차트를 시각화하며, 요약 리포트를 반환합니다.
    """
    try:
        df_close = get_yf_close_prices(ticker_list, period)
        index = ['SPY', 'QQQ']
        df_index = get_yf_close_prices(index, "5y")

        rebal_dates = get_rebal_dates(df_close, 'month')
        rebal_index = rebal_dates

        n_assets = df_close.shape[1]
        target_weight_df = pd.DataFrame(
            [[1 / n_assets] * n_assets] * len(rebal_index),
            index=rebal_index,
            columns=df_close.columns
        )

        individual_port_val_df = calculate_portvals(df_close, target_weight_df)
        df_port = pd.DataFrame({'port': individual_port_val_df.sum(axis=1)})
        df_all = pd.concat([df_port, df_index], axis=1).dropna().rebase()

        # ✅ 시각화를 위한 저장
        st.session_state["latest_history_chart"] = df_all

        return df_all.tail().to_markdown()

    except Exception as e:
        return f"❗ 백테스트 도중 오류 발생: {str(e)}"


@tool
def plot_history_chart() -> str:
    """
    가장 최근에 조회한 주가 데이터를 기반으로 차트를 시각화합니다.
    
    - 단일 종목 (get_yf_stock_history 호출 시): 종가 차트
    - 복수 종목 (get_yf_cumulative_returns_tool 호출 시): 누적 수익률 차트
    """
    chart_data = st.session_state.get("latest_history_chart")
    cum_rtn_data = st.session_state.get("latest_cum_rtn_df")

    if chart_data is not None and not chart_data.empty:
        st.subheader("📈 단일 종목 주가 히스토리 차트")
        st.line_chart(chart_data, use_container_width=True)
        return "✅ 단일 종목 주가 차트 시각화 완료"

    elif cum_rtn_data is not None and not cum_rtn_data.empty:
        st.subheader("📈 복수 종목 누적 수익률 비교 차트")
        st.line_chart(cum_rtn_data, use_container_width=True)
        return "✅ 누적 수익률 차트 시각화 완료"

    else:
        return "❗ 시각화할 데이터가 없습니다. 먼저 get_yf_stock_history 또는 get_yf_cumulative_returns_tool을 호출해주세요."



@tool
def get_backtest_summary_tool(ticker_list: list[str], period: str = "5y") -> str:
    """
    포트폴리오 성과지표 요약 (Total Return, CAGR, Sharpe, MDD, Volatility, 1M/3M/6M/YTD/1Y 수익률)
    """
    try:
        df_close = get_yf_close_prices(ticker_list, period)
        df_index = get_yf_close_prices(["SPY", "QQQ"], period)

        rebal_dates = get_rebal_dates(df_close, 'month')
        n_assets = len(df_close.columns)

        weight_df = pd.DataFrame(
            [[1 / n_assets] * n_assets] * len(rebal_dates),
            index=rebal_dates,
            columns=df_close.columns
        )

        portval_df = calculate_portvals(df_close, weight_df)
        port_series = portval_df.sum(axis=1)

        df_all = pd.concat([port_series.rename("port"), df_index], axis=1).dropna()
        rtn = get_returns_df(df_all, log=True)
        cum_rtn = get_cum_returns_df(rtn, log=True)

        today = df_all.index[-1]
        years = (today - df_all.index[0]).days / 365

        def get_cagr(series):
            return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

        def get_sharpe(series, rf=0.025):
            excess = series.mean() * 252 - rf
            return excess / (series.std() * np.sqrt(252))

        def get_mdd(series):
            cummax = series.cummax()
            dd = series / cummax - 1
            return dd.min()

        def get_weekly_vol(series):
            weekly_rtn = series.resample("W").last().pct_change().dropna()
            return weekly_rtn.std() * np.sqrt(52)

        def get_return_since(series, from_date):
            from_date = pd.to_datetime(from_date)
            if from_date not in series.index:
                from_date = series.index[series.index.get_indexer([from_date], method='nearest')[0]]
            return (series.loc[series.index[-1]] / series.loc[from_date]) - 1

        summary_data = []
        for col in df_all.columns:
            s_cum = cum_rtn[col]
            s_log = rtn[col]
            s_price = df_all[col]

            total_rtn = s_cum.iloc[-1] - 1
            cagr = get_cagr(s_cum)
            sharpe = get_sharpe(s_log)
            mdd = get_mdd(s_cum)
            vol = get_weekly_vol(s_price)

            one_month = today - pd.Timedelta(days=30)
            three_month = today - pd.Timedelta(days=90)
            six_month = today - pd.Timedelta(days=180)
            one_year = today - pd.Timedelta(days=365)
            ytd_start = pd.Timestamp(year=today.year, month=1, day=1)

            r_1m = get_return_since(s_cum, one_month)
            r_3m = get_return_since(s_cum, three_month)
            r_6m = get_return_since(s_cum, six_month)
            r_1y = get_return_since(s_cum, one_year)
            r_ytd = get_return_since(s_cum, ytd_start)

            summary_data.append([
                f"{total_rtn * 100:.2f}%",
                f"{cagr * 100:.2f}%",
                f"{sharpe:.2f}",
                f"{mdd * 100:.2f}%",
                f"{vol * 100:.2f}%",
                f"{r_1m * 100:.2f}%",
                f"{r_3m * 100:.2f}%",
                f"{r_6m * 100:.2f}%",
                f"{r_ytd * 100:.2f}%",
                f"{r_1y * 100:.2f}%",
            ])

        summary_df = pd.DataFrame(
            summary_data,
            columns=["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Volatility (Weekly)",
                     "1M", "3M", "6M", "YTD", "1Y"],
            index=df_all.columns
        )

        return "📊 백테스트 성과 요약:\n\n" + summary_df.to_markdown()

    except Exception as e:
        return f"❗ 성과 요약 계산 중 오류: {str(e)}"


# SSL 인증서 우회
@tool
def get_news_list(company_name: str, display=10) -> list:
    '''
    Naver 뉴스 API로 기업 관련 뉴스 검색 후 분석 결과를 반환합니다.
    '''
    client_id = os.getenv("NAVER_CLIENT_ID")  # 환경 변수에서 클라이언트 ID 가져오기
    client_secret = os.getenv("NAVER_CLIENT_SECRET")  # 환경 변수에서 클라이언트 Secret 가져오기
    
    # 쿼리 URL 인코딩
    query = urllib.parse.quote(company_name)
    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&sort=date"
    
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    
    print("API 호출 성공!")
    items = response.json().get("items", [])
    
    if not items:
        print(f"❗ '{company_name}' 관련 뉴스가 없습니다.")
        return []

    # 뉴스 기사 내용 정제 함수
    def clean(text):
        return html.unescape(BeautifulSoup(text, "html.parser").get_text())

    news_items = []
    
    for item in items:
        title = clean(item.get("title", "제목 없음"))
        desc = clean(item.get("description", ""))
        link = item.get("originallink") or item.get("link") or f"https://search.naver.com/search.naver?query={query}"
        pub_date = item.get("pubDate", "날짜 없음")  # 기사 발행 날짜
        
        # 발행 날짜 형식 변환
        try:
            pub_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            pub_date = "날짜 변환 오류"  # 날짜 형식 오류 시 기본 메시지
            
        # 결과에 제목, 링크, 설명, 날짜 추가
        news_items.append({
            "제목": title,
            "내용": desc,
            "링크": link,
            "발행 날짜": pub_date  # 날짜 형식 추가
        })
    
    return news_items




@tool
def get_naver_news_sentiment(company_name: str, display: int = 5) -> str:
    """
    NAVER 뉴스에서 기업 관련 최신 뉴스 검색 → GPT로 관련성 필터링 + 한글 요약 + 감성 분석.
    결과는 기사 제목, 링크, 요약, 감성으로 구성됩니다.
    """
    try:
        # 1. NAVER 뉴스 API 호출
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        query = urllib.parse.quote(company_name)
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&sort=date"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"❗ NAVER 뉴스 API 호출 오류: {response.status_code}"

        items = response.json().get("items", [])
        if not items:
            return f"❗ '{company_name}' 관련 뉴스가 없습니다."

        # 2. 기사 정제
        def clean(text):
            return html.unescape(BeautifulSoup(text, "html.parser").get_text())

        articles = []
        for i, item in enumerate(items):
            title = clean(item.get("title", "제목 없음"))
            desc = clean(item.get("description", ""))
            link = item.get("originallink") or item.get("link") or f"https://search.naver.com/search.naver?query={query}"
            articles.append(f"{i+1}. 제목: {title}\n내용: {desc}\n링크: {link}")

        news_text = "\n\n".join(articles)

        # 3. GPT 프롬프트 구성
        prompt = f"""
'{company_name}'이라는 키워드로 검색된 뉴스 목록입니다. 하지만 이 중 일부는 관련 없는 기사일 수 있습니다
(예: 애플민트, 애플우드 등). 아래 작업을 수행하세요:

1. 실제 기업 '{company_name}'과 관련 있는 기사만 골라주세요.
2. 각 기사를 한국어로 자연스럽게 요약해주세요 (보도문 스타일로).
3. 각 기사별로 '긍정', '부정', '중립' 중 감성을 판단해주세요.
4. 각 기사에 제목/링크/요약/감성을 포함해주세요.

[뉴스 목록]
{news_text}

💬 출력 예시:
### 1. AI 투자 확대  
링크: https://example.com/apple-ai  
요약: ...  
감성: 긍정
"""
        gpt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return gpt_response.choices[0].message.content.strip()

    except Exception as e:
        return f"❗ 뉴스 요약 또는 감성 분석 중 오류 발생: {str(e)}"
    
@tool
def get_google_trend_tool(keyword: str, geo: str = "world") -> str:
    """
    Google 검색 트렌드 데이터를 기반으로 검색량 변화를 분석합니다.
    - keyword: 검색할 키워드 또는 기업명
    - geo: 국가 코드 (예: 'KR', 'US', 'JP'; 기본값 'world')
    """
    with no_ssl_verification():
        try:
            # 1. pytrends 세팅
            pytrends = TrendReq(hl="en-US", tz=360)
            geo_code = "" if geo.lower() == "world" else geo.upper()

            # 2. 트렌드 요청
            pytrends.build_payload([keyword], cat=0, timeframe="today 3-m", geo=geo_code, gprop="")

            data = pytrends.interest_over_time()
            if data.empty:
                return f"❗ '{keyword}'에 대한 검색 트렌드 데이터를 찾을 수 없습니다."

            trend_series = data[keyword]

            # 3. Streamlit 차트 출력
            st.subheader(f"🔍 최근 3개월 Google 검색 트렌드: '{keyword}' ({geo_code or 'Global'})")
            st.line_chart(trend_series)

            # 4. 요약 분석
            recent = trend_series.iloc[-1]
            peak = trend_series.max()
            avg = trend_series.mean()
            min_ = trend_series.min()

            result = (
                f"✅ '{keyword}'에 대한 최근 3개월간 검색 트렌드 분석:\n"
                f"- 🔼 최고치: {peak:.0f}\n"
                f"- 🔽 최저치: {min_:.0f}\n"
                f"- 📊 평균: {avg:.1f}\n"
                f"- 📅 최근 검색량 (마지막일 기준): {recent:.0f}"
            )
            return result

        except Exception as e:
            return f"❗ Google 트렌드 분석 중 오류 발생: {str(e)}"

@tool
def compare_google_trend_tool(keywords: list[str], geo: str = "world") -> str:
    """
    여러 키워드에 대한 Google 검색 트렌드를 비교 분석합니다.
    - keywords: 검색어 리스트 (예: ['Apple', 'Microsoft'])
    - geo: 국가 코드 (기본: 'world' → 전세계, 예: 'KR', 'US')
    """
    with no_ssl_verification():
        try:
            if not keywords or len(keywords) < 2:
                return "❗ 2개 이상의 키워드를 입력해주세요."

            pytrends = TrendReq(hl="en-US", tz=360)
            geo_code = "" if geo.lower() == "world" else geo.upper()

            pytrends.build_payload(keywords, cat=0, timeframe="today 3-m", geo=geo_code, gprop="")

            data = pytrends.interest_over_time()
            if data.empty:
                return f"❗ '{', '.join(keywords)}'에 대한 검색 트렌드 데이터를 찾을 수 없습니다."

            data = data.drop(columns=["isPartial"], errors="ignore")

            # ✅ 차트 출력
            st.subheader(f"🔍 최근 3개월 검색 트렌드 비교: {', '.join(keywords)}")
            st.line_chart(data)

            # ✅ 요약 메시지 생성
            summaries = []
            for kw in keywords:
                recent = data[kw].iloc[-1]
                peak = data[kw].max()
                avg = data[kw].mean()
                summaries.append(
                    f"- **{kw}** → 🔼 최고: {peak:.0f}, 📊 평균: {avg:.1f}, 📅 최근: {recent:.0f}"
                )

            return "✅ Google 검색 트렌드 비교 결과:\n" + "\n".join(summaries)

        except Exception as e:
            return f"❗ Google 트렌드 비교 중 오류 발생: {str(e)}"




# 도구 바인딩
# ----- 도구 바인딩 -----
tools = [
    get_current_time,
    get_yf_stock_info,
    get_yf_stock_history,
    get_yf_cumulative_returns_tool,
    get_yf_stock_recommendations,
    plot_history_chart,  # ✅ 추가
    get_backtest_tool,  # ✅ 추가
    get_backtest_summary_tool,
    get_google_trend_tool, 
    compare_google_trend_tool,  # ✅ 추가
    get_news_list,
    get_naver_news_sentiment,
]

# name을 키로 하는 dict 생성
tool_dict = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)


# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)  # ① LLM + tool 사용

    gathered = None  # ② 전체 메시지를 누적할 변수

    for chunk in response:
        yield chunk

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    # ③ tool_calls가 있다면, 해당 tool 호출 후 ToolMessage로 응답
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)

        for tool_call in gathered.tool_calls:
            try:
                selected_tool = tool_dict[tool_call["name"]]
                tool_output = selected_tool.invoke(tool_call)

                tool_msg = ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tool_output)
                )
                st.session_state.messages.append(tool_msg)

            except Exception as e:
                st.session_state.messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"❌ 도구 호출 실패: {str(e)}"
                ))

        # ④ tool 실행 후, 재귀적으로 LLM 호출
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk


# Streamlit 앱
st.title("💬 GPT-4o Stock Chain Bot")

with st.expander("📌 사용할 수 있는 기능 요약 보기"):
    st.markdown("""
### 🎯 지원 기능 및 예시 명령어

| 기능 이름 | 설명 | 🔍 예시 명령어 |
|-----------|------|----------------|
| `get_current_time` | 지정한 타임존의 현재 시간을 보여줍니다. | `"서울의 현재 시간을 알려줘"` |
| `get_yf_stock_info` | 입력한 종목 티커의 Yahoo Finance 정보를 반환합니다. | `"AAPL 종목 정보 알려줘"` |
| `get_yf_stock_history` | 종목의 주가 이력을 가져옵니다. | `"TSLA의 최근 주가 흐름 보여줘"` |
| `get_yf_cumulative_returns_tool` | 여러 종목의 누적 수익률을 가져옵니다. | `"TSLA, PLTR의 1년 누적수익률 흐름 보여줘"` |
| `get_yf_stock_recommendations` | 애널리스트들의 종목 추천 데이터를 보여줍니다. | `"NVDA에 대한 리서치 추천 보여줘"` |
| `plot_history_chart` | 주가 이력을 시각화합니다. (이전 조회 필요) | `"차트로 보여줘"` |
| `get_backtest_tool` | 포트폴리오 누적 수익률 백테스트 실행 | `"AAPL과 MSFT로 5년 백테스트 해줘"` |
| `get_backtest_summary_tool` | 포트 수익률, 변동성, 샤프지수 등 요약 | `"테슬라와 엔비디아로 포트 성과 요약해줘"` |
| `get_naver_news_sentiment` | NAVER 뉴스에서 관련 뉴스 검색 + 감성 분석 | `"삼성전자 관련 뉴스 감성 분석해줘"` |
| `get_company_news_with_sentiment` | Naver 뉴스에서 기사 + 감성 점수로 요약 | `"LG에너지솔루션 뉴스 감성 요약"` |
| `get_google_trend_tool` | Google 검색 트렌드 분석 (최근 3개월) | `"애플 검색 트렌드 보여줘"` |
| `compare_google_trend_tool` | 여러 키워드 검색량 비교 | `"삼성전자와 애플 검색량 비교"` |
| `reddit_stock_summary_tool` | Reddit에서 인기 게시글 수집 | `"wallstreetbets에서 최근 인기 글 가져와줘"` |
    """)


# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 투자자를 돕기 위해 최선을 다하는 인공지능 봇이다."),  
        AIMessage("How can I help you?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(msg.content)


# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    response = get_ai_response(st.session_state["messages"])
    
    result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장    
    
    
with st.expander("🧠 Reddit 종목 요약 직접 실행"):
    if st.button("Reddit 종목 분석 실행"):
        with st.spinner("Reddit 분석 중..."):
            summary = run_reddit_summary(limit=30, subreddit="wallstreetbets")
            st.markdown(summary, unsafe_allow_html=True)