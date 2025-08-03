import ssl
import streamlit as st
import pandas as pd
from graph_chart import draw_backtest_chart
ssl._create_default_https_context = ssl._create_unverified_context

from gpt_function4 import (
    get_current_time,
    get_yf_stock_info,
    get_yf_stock_history,
    get_yf_stock_recommendations,
    get_latest_news_with_korean_translation,
    get_backtest,
    tools,
)
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import streamlit.components.v1 as components

# content가 리스트로 들어오는 오류 방지용 메시지 정제 함수
def sanitize_messages(messages):
    for m in messages:
        if isinstance(m.get("content"), list):
            m["content"] = " ".join(str(c) for c in m["content"])
        elif not isinstance(m.get("content"), str):
            m["content"] = str(m.get("content"))
    return messages

# 환경 변수 로드 및 OpenAI 클라이언트
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# OpenAI 클라이언트를 안전하게 초기화하는 함수
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❗ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.stop()
    return OpenAI(api_key=api_key)

# Streamlit UI 시작
st.title("💬 투자 챗봇 프로")

# 메시지 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "너는 사용자의 주식 문의에 답하고, 최신 뉴스도 반영해주는 주식 칼럼 챗봇이야.",
        }
    ]

# 이전 메시지 출력
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant"):
        if msg.get("content"):
            st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 받기
if user_input := st.chat_input("궁금한 종목이나 회사를 입력해주세요"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    client = get_openai_client()
    messages_to_send = sanitize_messages([
        m for m in st.session_state.messages if m.get("content") is not None
    ])

    ai_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_to_send,
        tools=tools,
    )
    ai_message = ai_response.choices[0].message

    # tool_call 처리
    tool_calls = ai_message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)

            try:
                if tool_name == "get_current_time":
                    func_result = get_current_time(arguments["timezone"])
                elif tool_name == "get_yf_stock_info":
                    func_result = get_yf_stock_info(arguments["ticker"])
                elif tool_name == "get_yf_stock_history":
                    func_result = get_yf_stock_history(arguments["ticker"], arguments["period"])
                elif tool_name == "get_yf_stock_recommendations":
                    func_result = get_yf_stock_recommendations(arguments["ticker"])
                elif tool_name == "get_latest_news_with_korean_translation":
                    func_result = get_latest_news_with_korean_translation(arguments["query"])
                elif tool_name == "get_backtest":
                    df_all = get_backtest(arguments["ticker_list"], arguments.get("period", "5y"))
                    if isinstance(df_all, pd.DataFrame):
                        draw_backtest_chart(df_all)
                        func_result = "✅ 백테스트 결과 시각화 완료"
                    else:
                        func_result = "❗ 백테스트 결과가 없습니다."
            except Exception as e:
                func_result = f"❗ 함수 실행 중 오류 발생: {str(e)}"

            st.session_state.messages.append({
                "role": "function",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": func_result,
            })

        st.session_state.messages.append({
            "role": "system",
            "content": "이제 기능 실행 결과를 참고해서 사용자에게 응답해줘.",
        })

        ai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=sanitize_messages([
                m for m in st.session_state.messages if m.get("content") is not None
            ]),
            tools=tools,
        )
        ai_message = ai_response.choices[0].message

    # content가 존재할 경우에만 출력
    if ai_message.content:
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_message.content,
        })
        st.chat_message("assistant").markdown(ai_message.content)

# 환경변수 확인 (디버그용)
print("✅ API KEY:", os.getenv("OPENAI_API_KEY"))
