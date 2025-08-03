import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.tools import tool
from datetime import datetime
import pytz
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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
    해당 종목의 Yahoo Finance 주가 정보를 반환합니다.
    
    period 예: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y 등
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    if history.empty:
        return f"No price history available for {ticker} with period '{period}'."
    return history.to_markdown()

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


# 도구 바인딩
# ----- 도구 바인딩 -----
tools = [
    get_current_time,
    get_yf_stock_info,
    get_yf_stock_history,
    get_yf_stock_recommendations,
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
