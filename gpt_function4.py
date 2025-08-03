from datetime import datetime
import pytz
import yfinance as yf
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI

from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# llm = ChatOpenAI(model="gpt-4o")

# gpt_functions3.py 상단 또는 Streamlit main script 상단
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from curl_cffi.requests import Session
import yfinance as yf

session = Session(
        impersonate="chrome110",  # optional, mimics Chrome browser
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        verify=False  # optional; disable SSL verify if you're behind a proxy
    )


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)

def get_current_time(timezone: str = 'Asia/Seoul'):
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"{now} {timezone}"

def get_yf_stock_info(ticker: str):
    stock = yf.Ticker(ticker, session=session)
    return str(stock.info)

def get_yf_stock_history(ticker: str, period: str):
    stock = yf.Ticker(ticker, session=session)
    return stock.history(period=period).to_markdown()

def get_yf_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker, session=session)
    return stock.recommendations.to_markdown()

def get_latest_news_with_korean_translation(query: str):
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "❗ TAVILY_API_KEY is missing."

    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 3,
            "include_raw_content": True
        },
    )
    results = response.json().get("results", [])
    if not results:
        return "❗ 관련 뉴스가 없습니다."

    news_text = "\n\n".join(
        f"제목: {item['title']}\n내용: {item['content']}\n출처: {item['url']}"
        for item in results
    )

    llm = ChatOpenAI(model="gpt-4o")
    prompt = [
        {
            "role": "system",
            "content": (
                "당신은 신문기사를 작성하는 전문 기자 AI입니다.\n"
                "주어진 뉴스 정보를 바탕으로, 한국어로 구조적이고 신뢰할 수 있는 신문기사를 작성하세요."
            ),
        },
        {
            "role": "user",
            "content": (
                f"다음 정보를 참고하여 `{query}`에 대한 기사 형식의 보고서를 작성해주세요.\n\n"
                f'정보:\n"""\n{news_text}\n"""\n\n'
                "- Markdown 문법을 사용하고, MLA 스타일을 따르세요.\n"
                "- 각 뉴스 출처는 반드시 명시해주세요."
            ),
        },
    ]
    messages = convert_openai_messages(prompt)
    article = llm.invoke(messages)

    return article.content

from curl_cffi.requests import Session
import yfinance as yf
import pandas as pd
import numpy as np
import ffn

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

    import quantstats as qs
    qs.extend_pandas()
    report_html = qs.reports.html(df_all['port'], df_all['SPY'], output=True)
    # qs.reports.full(df_all.loc[:,'port'], df_all.loc[:, 'SPY']).to_markdown() 
    return df_all.to_markdown()


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "해당 타임존의 날짜와 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "현재 시간을 확인할 타임존 (예: Asia/Seoul)"
                    }
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_info",
            "description": "해당 종목의 Yahoo Finance 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "조회할 종목 티커 (예: AAPL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_history",
            "description": "해당 종목의 과거 주가 데이터를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "조회할 종목 티커"
                    },
                    "period": {
                        "type": "string",
                        "description": "조회 기간 (예: 1d, 5d, 1mo, 1y, 5y)"
                    }
                },
                "required": ["ticker", "period"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_recommendations",
            "description": "해당 종목의 애널리스트 추천 데이터를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "조회할 종목 티커"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_news_with_korean_translation",
            "description": "해당 키워드에 대한 최신 뉴스와 한국어 번역을 제공합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 또는 종목명"
                    }
                },
                "required": ["query"]
            }
        }
    },
{
    "type": "function",
    "function": {
        "name": "get_backtest",
        "description": "티커 리스트를 넣으면 백테스팅을 수행함",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker_list": {  # ✅ 수정됨
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "조회할 종목 티커 리스트 입력"
                },
                "period": {
                    "type": "string",
                    "description": "조회 기간 (예: 1d, 5d, 1mo, 1y, 5y)"
                }
            },
            "required": ["ticker_list", "period"]
        }
    }
}

]
