import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI 주가 예측 서비스", layout="wide")
st.title("📈 AI 주가 분석 대시보드")
st.markdown("가벼운 통계 모델을 사용하여 주가 추세를 예측합니다.")

# 사이드바 설정
with st.sidebar:
    st.header("🔍 설정")
    target_code = st.text_input("종목 코드 (6자리)", value="005930")
    period = st.slider("예측 기간 (일)", 7, 30, 14)
    analyze_btn = st.button("분석 시작")

if analyze_btn:
    try:
        with st.spinner('데이터 분석 중...'):
            # 1. 데이터 수집 (최근 1년)
            df = fdr.DataReader(target_code, datetime.now() - timedelta(days=365), datetime.now())
            
            if df.empty:
                st.error("종목 코드를 확인해 주세요.")
            else:
                # 2. AI 모델링 (Holt-Winters 지수평활법)
                # Prophet보다 훨씬 가볍고 빠르며 추세 분석에 강합니다.
                model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
                forecast = model.forecast(period)
                
                # 날짜 생성
                last_date = df.index[-1]
                forecast_dates = [last_date + timedelta(days=i) for i in range(1, period + 1)]
                
                # 3. 차트 시각화 (Plotly 사용)
                fig = go.Figure()
                # 과거 데이터
                fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), name='과거 주가'))
                # 예측 데이터
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name='AI 예측', line=dict(dash='dash', color='red')))
                
                fig.update_layout(title=f"[{target_code}] 주가 예측 결과", xaxis_title="날짜", yaxis_title="가격(원)")
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. 결과 요약
                col1, col2 = st.columns(2)
                col1.metric("현재가", f"{df['Close'].iloc[-1]:,.0f}원")
                col2.metric(f"{period}일 후 예상가", f"{forecast.iloc[-1]:,.0f}원", f"{forecast.iloc[-1] - df['Close'].iloc[-1]:,.0f}원")

    except Exception as e:
        st.error(f"알 수 없는 오류가 발생했습니다: {e}")
else:
    st.info("왼쪽 사이드바에서 종목 코드를 입력하고 [분석 시작]을 눌러주세요.")


