from curl_cffi.requests import Session
import yfinance as yf
import pandas as pd
import numpy as np
import ffn
from langchain_core.tools import tool
from duckduckgo_search import DDGS
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
load_dotenv()


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

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from graph_chart import draw_backtest_chart
from langchain_core.tools import tool
from datetime import datetime
import pytz
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()


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
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        return f"{ticker}의 {period} 기간 주가 데이터가 없습니다."

    # ✅ 시각화를 위해 데이터 저장
    st.session_state["latest_history_chart"] = df[["Close"]].copy()
    st.session_state["latest_history_chart"].index = st.session_state["latest_history_chart"].index.strftime("%Y-%m-%d")

    return df.tail().to_markdown()

@tool
def get_yf_stock_info(ticker: str) -> str:
    """해당 종목의 Yahoo Finance 정보를 반환합니다."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return str(info)

@tool
def get_yf_stock_recommendations(ticker: str) -> str:
    """해당 종목의 리서치 추천 정보를 반환합니다."""
    stock = yf.Ticker(ticker)
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
    (get_yf_stock_history 이후에 호출되어야 합니다)
    """
    df = st.session_state.get("latest_history_chart")

    if df is None or df.empty:
        return "❗ 시각화할 주가 데이터가 없습니다. 먼저 get_yf_stock_history를 호출해주세요."

    st.subheader("📈 주가 히스토리 차트")
    st.line_chart(df, use_container_width=True)
    return "✅ 주가 히스토리 차트 시각화 완료"


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
    get_yf_stock_recommendations,
    plot_history_chart,  # ✅ 추가
    get_backtest_tool,  # ✅ 추가
    get_backtest_summary_tool,
    get_naver_news_sentiment,  # ✅ NAVER 뉴스 감성 분석 도구,  # ✅ 추가
    get_google_trend_tool, 
    compare_google_trend_tool,  # ✅ 추가
]
# name을 키로 하는 dict 생성
tool_dict = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)


# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(messages) # ① llm.stream()을 llm_with_tools.stream()로 변경
    
    gathered = None # ②
    for chunk in response:
        yield chunk
        
        if gathered is None: #  ③
            gathered = chunk
        else:
            gathered += chunk
 
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call) 
            print(tool_msg, type(tool_msg))
            st.session_state.messages.append(tool_msg)
           
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk


# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다. "),  
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