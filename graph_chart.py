import streamlit as st
import pandas as pd

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