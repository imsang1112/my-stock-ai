import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="간편 AI 주가 예측", layout="wide")
st.title("📈 초간편 주가 분석 봇")

with st.sidebar:
    target_code = st.text_input("종목 코드 (예: 005930)", value="005930")
    period = st.slider("예측 기간 (일)", 7, 30, 14)
    analyze_btn = st.button("분석 시작")

if analyze_btn:
    try:
        # 데이터 가져오기
        df = fdr.DataReader(target_code, datetime.now() - timedelta(days=365), datetime.now())
        
        if not df.empty:
            # AI 모델링 (지수 평활법 - 매우 가볍고 빠름)
            model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
            forecast = model.forecast(period)
            
            # 날짜 생성
            last_date = df.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, period + 1)]
            
            # 시각화
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[-60:], df['Close'].tail(60), label='Past') # 최근 60일 데이터
            ax.plot(forecast_dates, forecast, label='Predicted', linestyle='--', color='red')
            ax.legend()
            st.pyplot(fig)
            
            st.success(f"예측 완료! {period}일 뒤 예상 종가: {forecast.iloc[-1]:,.0f}원")
    except Exception as e:
        st.error(f"오류 발생: {e}")

