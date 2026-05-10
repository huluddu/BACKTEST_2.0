"""
main.py
=======
Backtest 2.0 - Streamlit 메인 앱

탭 구성:
  Tab 1. 📡 오늘의 시그널  - 설정된 전략의 현재 매수/매도 신호
  Tab 2. 📋 전략 프리셋    - 저장된 전략 일괄 분석 & 관리
  Tab 3. 🔬 백테스트      - 단일 전략 상세 백테스트
  Tab 4. ⚡ 전략 최적화   - Optuna 베이지안 파라미터 탐색
  Tab 5. 📊 구간 스트레스 - 5/10/15/20년 구간별 성과 비교
  Tab 6. 📓 매매일지      - 실제 매매 기록 관리
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from modules.engine import (
    StrategyParams, BacktestResult,
    prepare_data, run_backtest, get_today_signal,
)
from modules.optimizer import (
    SearchSpace, OptimizeConstraints,
    run_optimization, apply_optimal_params,
)
from modules.portfolio import (
    preset_to_params, run_portfolio_scan, run_period_stress_test,
)
from modules.utils import (
    save_strategy, load_strategies, delete_strategy,
    calc_monthly_returns, calc_annual_returns,
    format_result_metric, sharpe_ratio, calmar_ratio,
    init_session_state, get_state, set_state,
    save_journal_row, load_journal,
)

# ══════════════════════════════════════════════════════════
# 앱 설정
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Backtest 2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
init_session_state({
    "params":    StrategyParams(),
    "result":    None,
    "data":      None,
    "presets":   {},
    "opt_result": None,
    "gemini_key": "",
    "sheet_name": "backtest_strategies",
    "sheet_tab":  "전략목록",
})

# ══════════════════════════════════════════════════════════
# 사이드바: 공통 설정
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Backtest 2.0")
    st.divider()

    # ── 기간 설정 ──────────────────────────────────────
    st.subheader("📅 분석 기간")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            value=datetime.date.today() - datetime.timedelta(days=365 * 5),
            key="start_date",
        )
    with col2:
        end_date = st.date_input("종료일", value=datetime.date.today(), key="end_date")

    st.divider()

    # ── 티커 설정 ──────────────────────────────────────
    st.subheader("🔖 티커")
    signal_ticker = st.text_input("시그널 티커",  value="SOXL", key="sig_ticker").upper()
    trade_ticker  = st.text_input("매매 티커",    value="SOXL", key="trd_ticker").upper()
    market_ticker = st.text_input("시장 필터 티커", value="SPY",  key="mkt_ticker").upper()

    st.divider()

    # ── 전략 파라미터 ──────────────────────────────────
    st.subheader("⚙️ 전략 파라미터")

    with st.expander("📈 매수 조건", expanded=True):
        ma_buy       = st.selectbox("매수 이평선", [5,10,20,50,60,120,200], index=3, key="ma_buy")
        buy_operator = st.selectbox("조건 연산자", [">", "<"], key="buy_op")
        col1, col2   = st.columns(2)
        with col1:
            offset_cl_buy = st.selectbox("종가 오프셋", [1,5,10,20], key="off_cl_buy")
        with col2:
            offset_ma_buy = st.selectbox("MA 오프셋", [1,5,10,20], key="off_ma_buy")

    with st.expander("📉 매도 조건"):
        sell_operator = st.selectbox("조건 연산자", ["<", ">", "OFF"], key="sell_op")
        ma_sell       = st.selectbox("매도 이평선", [5,10,20,50,60,120,200], index=1, key="ma_sell")
        col1, col2    = st.columns(2)
        with col1:
            offset_cl_sell = st.selectbox("종가 오프셋", [1,5,10,20], key="off_cl_sell")
        with col2:
            offset_ma_sell = st.selectbox("MA 오프셋", [1,5,10,20], key="off_ma_sell")

    with st.expander("🔀 추세 필터"):
        use_trend_buy  = st.toggle("매수 시 추세 필터", value=True, key="use_trend_buy")
        use_trend_sell = st.toggle("매도 시 역추세 필터", value=False, key="use_trend_sell")
        col1, col2 = st.columns(2)
        with col1:
            ma_ts = st.selectbox("단기 추세선", [5,10,20,50], index=2, key="ma_ts")
            off_ts = st.selectbox("단기 오프셋", [1,5,10,20], key="off_ts")
        with col2:
            ma_tl = st.selectbox("장기 추세선", [20,50,60,120,200], index=1, key="ma_tl")
            off_tl = st.selectbox("장기 오프셋", [1,5,10,20], key="off_tl")

    with st.expander("🎯 볼린저 밴드"):
        use_bb = st.toggle("볼린저 밴드 모드", value=False, key="use_bb")
        if use_bb:
            bb_period = st.slider("기간", 10, 60, 20, key="bb_period")
            bb_std    = st.slider("표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std")
            bb_entry  = st.selectbox("진입 기준", [
                "상단선 돌파 (추세)", "하단선 이탈 (역추세)", "중심선 돌파"
            ], key="bb_entry")
            bb_exit   = st.selectbox("청산 기준", [
                "중심선(MA) 이탈", "상단선 복귀", "하단선 이탈"
            ], key="bb_exit")
        else:
            bb_period, bb_std = 20, 2.0
            bb_entry = "상단선 돌파 (추세)"
            bb_exit  = "중심선(MA) 이탈"

    with st.expander("📊 MACD 필터"):
        use_macd = st.toggle("MACD 필터 사용", value=False, key="use_macd")
        if use_macd:
            col1, col2, col3 = st.columns(3)
            with col1:
                macd_fast   = st.number_input("Fast", value=12, min_value=2, key="macd_fast")
            with col2:
                macd_slow   = st.number_input("Slow", value=26, min_value=2, key="macd_slow")
            with col3:
                macd_signal = st.number_input("Signal", value=9, min_value=2, key="macd_signal")
            macd_mode = st.selectbox("신호 방식", ["히스토그램 양전환", "골든크로스"], key="macd_mode")
        else:
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            macd_mode = "히스토그램 양전환"

    with st.expander("📉 RSI 필터"):
        use_rsi = st.toggle("RSI 필터 사용", value=False, key="use_rsi")
        if use_rsi:
            rsi_period = st.slider("RSI 기간", 5, 30, 14, key="rsi_period")
            rsi_min, rsi_max = st.slider("허용 RSI 범위", 0, 100, (30, 70), key="rsi_range")
        else:
            rsi_period, rsi_min, rsi_max = 14, 30, 70

    with st.expander("🌍 시장 필터"):
        use_mkt = st.toggle("시장 필터 사용", value=False, key="use_mkt")
        mkt_ma_p = st.slider("시장 MA 기간", 50, 300, 200, key="mkt_ma_p") if use_mkt else 200

    with st.expander("🛡 손절 / 익절"):
        use_atr_stop = st.toggle("ATR 손절", value=False, key="use_atr_stop")
        if use_atr_stop:
            atr_mult    = st.slider("ATR 배수", 1.0, 5.0, 2.0, 0.1, key="atr_mult")
            stop_pct    = 0.0
        else:
            atr_mult    = 2.0
            stop_pct    = st.slider("고정 손절(%)", 0, 50, 0, key="stop_pct")
        tp_pct = st.slider("익절(%)", 0, 100, 0, key="tp_pct")
        min_hold = st.slider("최소 보유일", 0, 30, 0, key="min_hold")

    with st.expander("💰 매매 비용"):
        strategy_behavior = st.radio(
            "동시 신호 처리",
            ["priority_sell", "mutual"],
            format_func=lambda x: "매도 우선" if x == "priority_sell" else "매도 후 즉시 재매수",
            key="strategy_behavior",
        )
        initial_cash = st.number_input("초기 자금 (원)", value=5_000_000, step=1_000_000, key="init_cash")
        fee_bps  = st.slider("수수료 (bps)", 0, 100, 25, key="fee_bps")
        slip_bps = st.slider("슬리피지 (bps)", 0, 50, 1, key="slip_bps")

    st.divider()

    # ── 구글 시트 설정 ────────────────────────────────
    with st.expander("🔗 구글 시트"):
        sheet_name = st.text_input("시트 이름", value=get_state("sheet_name"), key="sheet_name_input")
        sheet_tab  = st.text_input("탭 이름",   value=get_state("sheet_tab"),  key="sheet_tab_input")
        set_state("sheet_name", sheet_name)
        set_state("sheet_tab",  sheet_tab)

        if st.button("📥 전략 불러오기"):
            with st.spinner("불러오는 중..."):
                loaded = load_strategies(sheet_name, sheet_tab)
                if loaded:
                    set_state("presets", loaded)
                    st.success(f"{len(loaded)}개 전략 로드 완료")
                else:
                    st.warning("저장된 전략이 없거나 연결 실패")

    st.divider()

    # ── 현재 파라미터 조합 저장 ────────────────────────
    st.subheader("💾 전략 저장")
    save_name = st.text_input("전략 이름", placeholder="예: SOXL_MA50_추세", key="save_name")
    if st.button("💾 현재 설정 저장", use_container_width=True):
        if not save_name:
            st.warning("전략 이름을 입력해주세요")
        else:
            params_dict = _collect_params_dict()  # 아래 정의
            presets = get_state("presets")
            presets[save_name] = params_dict
            set_state("presets", presets)
            save_strategy(sheet_name, sheet_tab, save_name, params_dict)


# ── 파라미터 dict 수집 함수 (사이드바 값 → dict) ──────
def _collect_params_dict() -> dict:
    return {
        "signal_ticker_input": signal_ticker,
        "trade_ticker_input":  trade_ticker,
        "market_ticker_input": market_ticker,
        "ma_buy":              ma_buy,
        "buy_operator":        buy_operator,
        "offset_cl_buy":       offset_cl_buy,
        "offset_ma_buy":       offset_ma_buy,
        "ma_sell":             ma_sell,
        "sell_operator":       sell_operator,
        "offset_cl_sell":      offset_cl_sell,
        "offset_ma_sell":      offset_ma_sell,
        "use_trend_in_buy":    use_trend_buy,
        "use_trend_in_sell":   use_trend_sell,
        "ma_compare_short":    ma_ts,
        "ma_compare_long":     ma_tl,
        "offset_compare_short": off_ts,
        "offset_compare_long": off_tl,
        "use_bollinger":       use_bb,
        "bb_period":           bb_period,
        "bb_std":              bb_std,
        "bb_entry_type":       bb_entry,
        "bb_exit_type":        bb_exit,
        "use_macd":            use_macd,
        "macd_fast":           macd_fast,
        "macd_slow":           macd_slow,
        "macd_signal_period":  macd_signal,
        "macd_mode":           macd_mode,
        "use_rsi_filter":      use_rsi,
        "rsi_period":          rsi_period,
        "rsi_min":             rsi_min,
        "rsi_max":             rsi_max,
        "use_market_filter":   use_mkt,
        "market_ma_period":    mkt_ma_p,
        "use_atr_stop":        use_atr_stop,
        "atr_multiplier":      atr_mult,
        "stop_loss_pct":       stop_pct,
        "take_profit_pct":     tp_pct,
        "min_hold_days":       min_hold,
        "strategy_behavior":   strategy_behavior,
        "initial_cash":        initial_cash,
        "fee_bps":             fee_bps,
        "slip_bps":            slip_bps,
    }


def _collect_params() -> StrategyParams:
    """사이드바 설정값 → StrategyParams 변환."""
    return StrategyParams(
        signal_ticker      = signal_ticker,
        trade_ticker       = trade_ticker,
        market_ticker      = market_ticker,
        ma_buy             = ma_buy,
        buy_operator       = buy_operator,
        offset_cl_buy      = offset_cl_buy,
        offset_ma_buy      = offset_ma_buy,
        ma_sell            = ma_sell,
        sell_operator      = sell_operator,
        offset_cl_sell     = offset_cl_sell,
        offset_ma_sell     = offset_ma_sell,
        use_trend_buy      = use_trend_buy,
        use_trend_sell     = use_trend_sell,
        ma_trend_short     = ma_ts,
        ma_trend_long      = ma_tl,
        offset_trend_short = off_ts,
        offset_trend_long  = off_tl,
        use_bollinger      = use_bb,
        bb_period          = bb_period,
        bb_std             = bb_std,
        bb_entry_type      = bb_entry,
        bb_exit_type       = bb_exit,
        use_macd           = use_macd,
        macd_fast          = macd_fast,
        macd_slow          = macd_slow,
        macd_signal_period = macd_signal,
        macd_mode          = macd_mode,
        use_rsi_filter     = use_rsi,
        rsi_period         = rsi_period,
        rsi_min            = rsi_min,
        rsi_max            = rsi_max,
        use_market_filter  = use_mkt,
        market_ma_period   = mkt_ma_p,
        use_atr_stop       = use_atr_stop,
        atr_multiplier     = atr_mult,
        stop_loss_pct      = stop_pct,
        take_profit_pct    = tp_pct,
        min_hold_days      = min_hold,
        strategy_behavior  = strategy_behavior,
        initial_cash       = float(initial_cash),
        fee_bps            = float(fee_bps),
        slip_bps           = float(slip_bps),
    )


# ══════════════════════════════════════════════════════════
# 차트 공통 함수
# ══════════════════════════════════════════════════════════

def _draw_price_chart(chart_data: dict, trade_log: list, p: StrategyParams) -> go.Figure:
    """캔들 + 이평선 + 매매 시그널 차트."""
    base = chart_data["base"]
    dates = base["Date"]

    rows = 1
    specs = [[{"type": "xy"}]]

    # MACD 패널
    if p.use_macd and chart_data.get("macd_hist") is not None:
        rows += 1
        specs.append([{"type": "xy"}])

    # RSI 패널
    if p.use_rsi_filter and chart_data.get("rsi") is not None:
        rows += 1
        specs.append([{"type": "xy"}])

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7] + [0.15] * (rows - 1),
        specs=specs,
    )

    # ── 가격 (캔들) ─────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates,
        open=base["Open_trd"], high=base["High_trd"],
        low=base["Low_trd"],   close=base["Close_trd"],
        name="가격", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # ── 이평선 ──────────────────────────────────────
    if chart_data.get("ma_buy_arr") is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=chart_data["ma_buy_arr"],
            name=f"MA{p.ma_buy}(매수)", line=dict(color="#2196F3", width=1.5)
        ), row=1, col=1)

    if chart_data.get("ma_sell_arr") is not None and p.sell_operator != "OFF":
        fig.add_trace(go.Scatter(
            x=dates, y=chart_data["ma_sell_arr"],
            name=f"MA{p.ma_sell}(매도)", line=dict(color="#FF9800", width=1.5)
        ), row=1, col=1)

    # ── 볼린저 밴드 ──────────────────────────────────
    if p.use_bollinger and chart_data.get("bb_upper") is not None:
        for band, name, color in [
            ("bb_upper", "BB상단", "rgba(130,130,255,0.7)"),
            ("bb_mid",   "BB중심", "rgba(130,130,255,0.9)"),
            ("bb_lower", "BB하단", "rgba(130,130,255,0.7)"),
        ]:
            fig.add_trace(go.Scatter(
                x=dates, y=chart_data[band],
                name=name, line=dict(color=color, width=1, dash="dot")
            ), row=1, col=1)

    # ── 매매 시그널 마커 ─────────────────────────────
    buys  = [l for l in trade_log if l["신호"] == "BUY"]
    sells = [l for l in trade_log if l["신호"] == "SELL"]

    if buys:
        fig.add_trace(go.Scatter(
            x=[l["날짜"] for l in buys],
            y=[l["체결가"] for l in buys],
            mode="markers",
            marker=dict(symbol="triangle-up", color="#26a69a", size=12),
            name="매수",
            text=[l.get("상세", "") for l in buys],
        ), row=1, col=1)

    if sells:
        fig.add_trace(go.Scatter(
            x=[l["날짜"] for l in sells],
            y=[l["체결가"] for l in sells],
            mode="markers",
            marker=dict(symbol="triangle-down", color="#ef5350", size=12),
            name="매도",
            text=[l.get("이유", "") for l in sells],
        ), row=1, col=1)

    # ── MACD ────────────────────────────────────────
    cur_row = 2
    if p.use_macd and chart_data.get("macd_hist") is not None:
        hist = chart_data["macd_hist"]
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
        fig.add_trace(go.Bar(
            x=dates, y=hist, name="MACD Hist",
            marker_color=colors, showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=chart_data["macd_line"],
            name="MACD", line=dict(color="#2196F3", width=1),
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=chart_data["macd_signal"],
            name="Signal", line=dict(color="#FF9800", width=1),
        ), row=cur_row, col=1)
        cur_row += 1

    # ── RSI ─────────────────────────────────────────
    if p.use_rsi_filter and chart_data.get("rsi") is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=chart_data["rsi"],
            name="RSI", line=dict(color="#9C27B0", width=1.5),
        ), row=cur_row, col=1)
        for lvl, color in [(70, "red"), (30, "green"), (50, "gray")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                          line_width=0.8, row=cur_row, col=1)

    fig.update_layout(
        height=500 + (rows - 1) * 150,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _draw_equity_chart(result: BacktestResult, chart_data: dict) -> go.Figure:
    """자산 곡선 vs B&H 비교 차트."""
    dates = chart_data["base"]["Date"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=chart_data["asset_curve"],
        name="전략", line=dict(color="#2196F3", width=2), fill="tozeroy",
        fillcolor="rgba(33,150,243,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=chart_data["bh_curve"],
        name="B&H", line=dict(color="#FF9800", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="자산(원)",
    )
    return fig


def _draw_monthly_heatmap(result: BacktestResult, chart_data: dict) -> go.Figure:
    """월별 수익률 히트맵."""
    dates = chart_data["base"]["Date"]
    monthly_df = calc_monthly_returns(chart_data["asset_curve"], dates)

    if monthly_df.empty:
        return go.Figure()

    z     = monthly_df.values.tolist()
    years = [str(y) for y in monthly_df.index]
    months= list(monthly_df.columns)

    fig = go.Figure(go.Heatmap(
        z=z, x=months, y=years,
        colorscale=[
            [0.0,  "#d32f2f"], [0.35, "#ef9a9a"],
            [0.5,  "#f5f5f5"],
            [0.65, "#a5d6a7"], [1.0,  "#2e7d32"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        height=max(200, len(years) * 30 + 80),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(side="top"),
    )
    return fig


# ══════════════════════════════════════════════════════════
# 메인 탭 구성
# ══════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 오늘의 시그널",
    "📋 전략 프리셋",
    "🔬 백테스트",
    "⚡ 전략 최적화",
    "📊 구간 스트레스",
    "📓 매매일지",
])


# ══════════════════════════════════════════════════════════
# Tab 1: 오늘의 시그널
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("📡 오늘의 시그널")
    st.caption("현재 사이드바 설정 기준으로 최신 신호를 확인합니다.")

    p = _collect_params()

    col_run, col_space = st.columns([1, 3])
    with col_run:
        run_signal = st.button("🔍 신호 확인", use_container_width=True, type="primary")

    if run_signal:
        with st.spinner("데이터 로드 중..."):
            data = prepare_data(
                p.signal_ticker, p.trade_ticker, p.market_ticker,
                start_date, end_date, p
            )

        if data is None:
            st.error("데이터 로드 실패. 티커와 기간을 확인해주세요.")
        else:
            sig = get_today_signal(data, p)

            # 시그널 상태 박스
            status = sig["status"]
            if "매수" in status:
                st.success(f"## {status}")
            elif "매도" in status:
                st.warning(f"## {status}")
            elif "중복" in status:
                st.error(f"## {status}")
            else:
                st.info(f"## {status}")

            st.caption(f"기준일: {sig['ref_date']}  |  추세: {'📈 상승추세' if sig['trend_up'] else '📉 하락추세'}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 매수 조건")
                c = "green" if sig["buy_ok"] else "red"
                icon = "✅" if sig["buy_ok"] else "❌"
                st.markdown(f":{c}[{icon} {sig['buy_msg']}]")

            with col2:
                st.markdown("#### 📉 매도 조건")
                c = "orange" if sig["sell_ok"] else "gray"
                icon = "✅" if sig["sell_ok"] else "❌"
                st.markdown(f":{c}[{icon} {sig['sell_msg']}]")

    # 프리셋 전체 시그널 일람
    presets = get_state("presets")
    if presets:
        st.divider()
        st.subheader("📋 저장된 전략 시그널 일람")
        rows = []
        for name, pd_dict in presets.items():
            try:
                pp = preset_to_params(pd_dict)
                d = prepare_data(pp.signal_ticker, pp.trade_ticker, pp.market_ticker,
                                 start_date, end_date, pp)
                if d:
                    sig = get_today_signal(d, pp)
                    rows.append({"전략명": name, "티커": pp.trade_ticker, "신호": sig["status"]})
                else:
                    rows.append({"전략명": name, "티커": pp.trade_ticker, "신호": "❌ 데이터 없음"})
            except Exception:
                rows.append({"전략명": name, "티커": "?", "신호": "❌ 오류"})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# Tab 2: 전략 프리셋 관리 + 일괄 분석
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("📋 전략 프리셋")

    presets = get_state("presets")

    sub1, sub2 = st.tabs(["🗂 전략 목록 & 시그널", "📊 구간별 성과 비교"])

    with sub1:
        if not presets:
            st.info("💡 저장된 전략이 없습니다. 사이드바에서 전략을 저장해주세요.")
        else:
            col_scan, col_del = st.columns([2, 1])
            with col_scan:
                if st.button("🔄 전체 분석 실행", type="primary", use_container_width=True):
                    prog = st.progress(0)
                    scan_df = run_portfolio_scan(presets, start_date, end_date,
                                                progress_placeholder=prog)
                    set_state("scan_result", scan_df)

            scan_result = get_state("scan_result")
            if scan_result is not None and not scan_result.empty:
                st.dataframe(
                    scan_result.style.applymap(
                        lambda v: "color: #26a69a" if "매수" in str(v) else
                                  "color: #ef5350" if "매도" in str(v) else "",
                        subset=["오늘신호"]
                    ),
                    use_container_width=True, hide_index=True,
                )

            # 전략 삭제
            with col_del:
                del_name = st.selectbox("삭제할 전략", [""] + list(presets.keys()), key="del_select")
                if st.button("🗑️ 삭제", use_container_width=True):
                    if del_name:
                        presets.pop(del_name, None)
                        set_state("presets", presets)
                        delete_strategy(
                            get_state("sheet_name"), get_state("sheet_tab"), del_name
                        )
                        st.rerun()

    with sub2:
        if not presets:
            st.info("저장된 전략이 없습니다.")
        else:
            if st.button("📊 구간별 성과 분석", type="primary", use_container_width=True):
                prog = st.progress(0)
                stress_df = run_period_stress_test(presets, progress_placeholder=prog)
                set_state("stress_result", stress_df)

            stress = get_state("stress_result")
            if stress is not None and not stress.empty:
                st.dataframe(stress, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 3: 백테스트
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("🔬 백테스트")

    p = _collect_params()

    run_bt = st.button("▶️ 백테스트 실행", type="primary", use_container_width=True)

    if run_bt:
        with st.spinner("데이터 준비 중..."):
            data = prepare_data(
                p.signal_ticker, p.trade_ticker, p.market_ticker,
                start_date, end_date, p
            )

        if data is None:
            st.error("데이터 로드 실패. 티커와 기간을 확인해주세요.")
        else:
            with st.spinner("백테스트 실행 중..."):
                result = run_backtest(data, p)
            set_state("result", result)
            set_state("data",   data)
            set_state("params", p)

    result: BacktestResult = get_state("result")
    data   = get_state("data")
    p_used: StrategyParams = get_state("params") or _collect_params()

    if result and result.is_valid:
        # ── 핵심 지표 ─────────────────────────────────
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        metrics = [
            (col1, "📈 수익률",      format_result_metric(result.total_return_pct),   result.total_return_pct - result.bh_return_pct),
            (col2, "📊 B&H 수익률", format_result_metric(result.bh_return_pct),       None),
            (col3, "📉 MDD",         format_result_metric(result.mdd_pct),             result.mdd_pct - result.bh_mdd_pct),
            (col4, "🎯 승률",         format_result_metric(result.win_rate_pct),        None),
            (col5, "⚡ Profit Factor", f"{result.profit_factor:.2f}",                  None),
            (col6, "🔄 매매횟수",     f"{result.total_trades}회",                       None),
        ]
        for col, label, val, delta in metrics:
            col.metric(
                label, val,
                delta=f"{delta:+.1f}%" if delta is not None else None,
                delta_color="normal" if delta and delta > 0 else "inverse" if delta and delta < 0 else "off",
            )

        # 추가 지표
        col1, col2, col3 = st.columns(3)
        sr = sharpe_ratio(result.asset_curve)
        cr = calmar_ratio(result.total_return_pct, result.mdd_pct)
        col1.metric("📐 샤프 비율", f"{sr:.2f}")
        col2.metric("🏆 Calmar Ratio", f"{cr:.2f}")
        col3.metric("💰 최종 자산", f"₩{result.asset_curve[-1]:,.0f}")

        st.divider()

        # ── 차트 ─────────────────────────────────────
        st.subheader("📈 가격 & 시그널")
        fig_price = _draw_price_chart(result.chart_data, result.trade_log, p_used)
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("💹 자산 곡선")
        fig_equity = _draw_equity_chart(result, result.chart_data)
        st.plotly_chart(fig_equity, use_container_width=True)

        st.subheader("🗓 월별 수익률 히트맵")
        fig_heatmap = _draw_monthly_heatmap(result, result.chart_data)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # ── 매매 내역 ─────────────────────────────────
        st.subheader("📋 매매 내역")
        if result.trade_log:
            log_df = pd.DataFrame(result.trade_log)
            log_df["날짜"] = pd.to_datetime(log_df["날짜"]).dt.strftime("%Y-%m-%d")
            st.dataframe(
                log_df.style.applymap(
                    lambda v: "color: #26a69a; font-weight:bold" if v == "BUY" else
                              "color: #ef5350; font-weight:bold" if v == "SELL" else "",
                    subset=["신호"]
                ),
                use_container_width=True, hide_index=True,
            )

    elif result and result.error:
        st.error(f"백테스트 실패: {result.error}")


# ══════════════════════════════════════════════════════════
# Tab 4: 전략 최적화
# ══════════════════════════════════════════════════════════
with tab4:
    st.header("⚡ 전략 최적화 (Optuna 베이지안)")
    st.caption("Train/Test 분리 검증으로 과적합을 방지하며 최적 파라미터를 탐색합니다.")

    # ── 최적화 모드 선택 ──────────────────────────────────
    opt_mode = st.radio(
        "최적화 모드",
        ["🎯 현재 설정 기반 최적화", "🤖 AI 풀옵션 자동 탐색"],
        horizontal=True,
        key="opt_mode",
    )
    st.caption(
        "**현재 설정 기반:** 사이드바 설정을 기준으로 수치 파라미터만 최적화  \n"
        "**AI 풀옵션:** 매수/매도 조건, 필터, 손절 등 모든 옵션을 AI가 자동 탐색"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        n_trials     = st.slider("탐색 횟수 (많을수록 정확)", 30, 500, 100, step=10, key="opt_n_trials")
        opt_target   = st.selectbox("최적화 목표", [
            "수익률 (%)", "다중 목적 (수익률↑ + MDD↓)", "Profit Factor", "승률 (%)", "MDD 최소화"
        ], key="opt_target")
    with col2:
        split_ratio  = st.slider("Train 비율 (앞부분)", 0.3, 0.8, 0.6, 0.05, key="opt_split")
        min_trades   = st.slider("최소 매매 횟수 (필터)", 1, 30, 5, key="opt_min_trades")
    with col3:
        max_mdd      = st.slider("최대 허용 MDD (%) (0=제한없음)", 0, 100, 0, key="opt_max_mdd")
        min_test_ret = st.slider("Test 구간 최소 수익률 (%)", -100, 100, -50, key="opt_min_test")

    # AI 풀옵션 모드 추가 설정
    if opt_mode == "🤖 AI 풀옵션 자동 탐색":
        st.divider()
        st.markdown("##### 🤖 AI 탐색 범위 설정")
        col1, col2, col3 = st.columns(3)
        with col1:
            ai_use_bb    = st.toggle("볼린저 밴드 포함", value=True,  key="ai_use_bb")
            ai_use_macd  = st.toggle("MACD 필터 포함",   value=True,  key="ai_use_macd")
        with col2:
            ai_use_rsi   = st.toggle("RSI 필터 포함",    value=True,  key="ai_use_rsi")
            ai_use_trend = st.toggle("추세 필터 포함",    value=True,  key="ai_use_trend")
        with col3:
            ai_use_atr   = st.toggle("ATR 손절 포함",    value=True,  key="ai_use_atr")
            ai_use_mkt   = st.toggle("시장 필터 포함",    value=False, key="ai_use_mkt")

        ai_ma_choices = st.multiselect(
            "탐색할 이평선 기간",
            [5, 10, 20, 50, 60, 120, 200],
            default=[5, 10, 20, 50, 60, 120],
            key="ai_ma_choices",
        )

    # 익절 탐색 끄기 (기존 disable_tp_checkbox 기능 그대로)
    st.divider()
    disable_tp = st.checkbox(
        "🚫 익절(Take Profit) 탐색 끄기 (0%로 고정, 탐색 속도 향상)",
        value=False, key="opt_disable_tp"
    )

    p_base = _collect_params()

    if st.button("🚀 최적화 시작", type="primary", use_container_width=True):
        prog_bar  = st.progress(0)
        status_ph = st.empty()

        def _progress(cur, total):
            prog_bar.progress(int(cur / total * 100))
            status_ph.caption(f"⏳ Trial {cur}/{total} 완료...")

        # ── 탐색 공간 구성 ────────────────────────────────
        from modules.optimizer import make_full_search_space, make_simple_search_space
        if opt_mode == "🤖 AI 풀옵션 자동 탐색":
            ma_list = ai_ma_choices if ai_ma_choices else [5, 10, 20, 50, 60, 120]
            ss = make_full_search_space(
                ma_choices=ma_list,
                use_trend=ai_use_trend,
                use_atr=ai_use_atr,
            )
            p_base.use_bollinger      = ai_use_bb
            p_base.use_macd          = ai_use_macd
            p_base.use_rsi_filter    = ai_use_rsi
            p_base.use_market_filter = ai_use_mkt
        else:
            ss = make_simple_search_space(p_base)

        with st.spinner("최적화 실행 중... (탐색 횟수가 많을수록 시간이 걸립니다)"):
            opt_df, opt_study = run_optimization(
                signal_ticker = p_base.signal_ticker,
                trade_ticker  = p_base.trade_ticker,
                start_date    = start_date,
                end_date      = end_date,
                split_ratio   = split_ratio,
                base_params   = p_base,
                search_space  = ss,
                constraints   = OptimizeConstraints(
                    min_trades   = min_trades,
                    max_mdd      = float(max_mdd) if max_mdd > 0 else 0,
                    min_test_ret = float(min_test_ret),
                ),
                n_trials    = n_trials,
                target      = opt_target,
                disable_tp  = disable_tp,
                progress_cb = _progress,
            )

        prog_bar.empty()
        status_ph.empty()
        set_state("opt_result", opt_df)
        set_state("opt_study",  opt_study)
        set_state("opt_mode_used", opt_mode)
        set_state("opt_target_used", opt_target)

    # ── 결과 표시 ─────────────────────────────────────────
    opt_df       = get_state("opt_result")
    opt_target_u = get_state("opt_target_used", "수익률 (%)")
    is_multi     = (opt_target_u == "다중 목적 (수익률↑ + MDD↓)")

    if opt_df is not None and not opt_df.empty:
        mode_used = get_state("opt_mode_used", "")
        label = "Pareto Front 후보" if is_multi else "유효 결과"
        st.success(f"✅ {len(opt_df)}개 {label} 발견 ({mode_used})")

        if is_multi:
            st.info("수익률↑ + MDD↓ 동시 최적화 결과입니다. AI가 찾아낸 **공격형 ~ 안정형** 포트폴리오 중 마음에 드는 것을 선택하세요.")

        # 결과 테이블 — 성과 지표 / 파라미터 탭으로 분리
        res_cols = ["Full_수익률(%)", "Full_MDD(%)", "Full_승률(%)", "Full_PF",
                    "Full_매매횟수", "Train_수익률(%)", "Test_수익률(%)", "Test_MDD(%)"]
        param_cols = [c for c in opt_df.columns if c not in res_cols]

        rtab1, rtab2 = st.tabs(["📊 성과 지표", "⚙️ 파라미터"])
        with rtab1:
            st.dataframe(
                opt_df[res_cols].head(20).style.background_gradient(
                    subset=["Full_수익률(%)"], cmap="RdYlGn"
                ),
                use_container_width=True, hide_index=True,
            )
        with rtab2:
            st.dataframe(opt_df[param_cols].head(20), use_container_width=True, hide_index=True)

        st.divider()

        # ── 최적 파라미터 적용 (버그 수정: 위젯 키에 직접 값 주입) ──
        st.subheader("🏆 최적 파라미터 사이드바 적용")
        st.caption("적용 후 페이지가 새로고침되며 사이드바 값이 변경됩니다. Tab 3에서 바로 백테스트 실행하세요.")

        top_idx = st.selectbox(
            "적용할 순위 선택",
            list(range(min(10, len(opt_df)))),
            format_func=lambda i: (
                f"#{i+1}  수익률 {opt_df.iloc[i]['Full_수익률(%)']:.1f}%  |  "
                f"MDD {opt_df.iloc[i]['Full_MDD(%)']:.1f}%  |  "
                f"승률 {opt_df.iloc[i]['Full_승률(%)']:.1f}%  |  "
                f"매매 {int(opt_df.iloc[i]['Full_매매횟수'])}회"
            ),
            key="opt_apply_idx",
        )

        # 선택된 파라미터 미리보기
        selected_row = opt_df.iloc[top_idx]
        with st.expander("📋 선택된 파라미터 미리보기"):
            preview_cols = st.columns(3)
            preview_items = [(k, v) for k, v in selected_row.items() if k in param_cols]
            for idx_p, (k, v) in enumerate(preview_items):
                preview_cols[idx_p % 3].metric(k, v)

        if st.button("✅ 사이드바에 적용하기", type="primary", use_container_width=True, key="opt_apply_btn"):
            row = opt_df.iloc[top_idx]

            # [버그 수정] session_state 위젯 키에 직접 값 주입
            # → Streamlit은 위젯 키로 값을 읽으므로, 키에 직접 써야 사이드바에 반영됨
            key_map = {
                "ma_buy":          ("ma_buy",        int),
                "ma_sell":         ("ma_sell",       int),
                "offset_cl_buy":   ("off_cl_buy",    int),
                "offset_ma_buy":   ("off_ma_buy",    int),
                "offset_cl_sell":  ("off_cl_sell",   int),
                "offset_ma_sell":  ("off_ma_sell",   int),
                "buy_operator":    ("buy_op",        str),
                "sell_operator":   ("sell_op",       str),
                "use_trend_buy":   ("use_trend_buy", lambda x: str(x).lower() == "true"),
                "use_trend_sell":  ("use_trend_sell",lambda x: str(x).lower() == "true"),
                "ma_trend_short":  ("ma_ts",         int),
                "ma_trend_long":   ("ma_tl",         int),
                "stop_loss_pct":   ("stop_pct",      int),
                "take_profit_pct": ("tp_pct",        int),
                "use_atr_stop":    ("use_atr_stop",  lambda x: str(x).lower() == "true"),
                "atr_multiplier":  ("atr_mult",      float),
            }

            applied = []
            for col_name, (widget_key, cast_fn) in key_map.items():
                if col_name in row.index:
                    try:
                        val = cast_fn(row[col_name])
                        st.session_state[widget_key] = val
                        applied.append(f"{widget_key}={val}")
                    except Exception:
                        pass

            # selectbox 위젯은 값이 아닌 인덱스로 저장되므로 별도 처리
            ma_choices = [5, 10, 20, 50, 60, 120, 200]
            off_choices = [1, 5, 10, 20]
            op_buy_choices = [">", "<"]
            op_sell_choices = ["<", ">", "OFF"]
            ts_choices = [5, 10, 20, 50]
            tl_choices = [20, 50, 60, 120, 200]

            def _safe_idx(lst, val, cast=int):
                try:
                    return lst.index(cast(val))
                except (ValueError, TypeError):
                    return 0

            st.session_state["ma_buy"]      = int(row.get("ma_buy", 50))
            st.session_state["ma_sell"]     = int(row.get("ma_sell", 10))
            st.session_state["off_cl_buy"]  = int(row.get("offset_cl_buy", 1))
            st.session_state["off_ma_buy"]  = int(row.get("offset_ma_buy", 1))
            st.session_state["off_cl_sell"] = int(row.get("offset_cl_sell", 1))
            st.session_state["off_ma_sell"] = int(row.get("offset_ma_sell", 1))
            st.session_state["buy_op"]      = str(row.get("buy_operator", ">"))
            st.session_state["sell_op"]     = str(row.get("sell_operator", "<"))
            st.session_state["use_trend_buy"]  = str(row.get("use_trend_buy", "True")).lower() == "true"
            st.session_state["use_trend_sell"] = str(row.get("use_trend_sell", "False")).lower() == "true"
            st.session_state["ma_ts"]       = int(row.get("ma_trend_short", 20))
            st.session_state["ma_tl"]       = int(row.get("ma_trend_long", 50))
            st.session_state["stop_pct"]    = int(float(row.get("stop_loss_pct", 0)))
            st.session_state["tp_pct"]      = int(float(row.get("take_profit_pct", 0)))
            st.session_state["use_atr_stop"]= str(row.get("use_atr_stop", "False")).lower() == "true"
            st.session_state["atr_mult"]    = float(row.get("atr_multiplier", 2.0))

            st.success("✅ 사이드바 적용 완료! 좌측 사이드바 값이 변경되었습니다.")
            st.info("👉 Tab 3 (백테스트) 탭으로 이동해서 ▶️ 백테스트 실행을 눌러주세요.")
            st.rerun()

    elif opt_df is not None:
        st.warning("유효한 최적화 결과가 없습니다. 아래를 확인해보세요.")
        st.markdown("""
        - 탐색 횟수를 늘려보세요 (100 → 200)
        - 최소 매매 횟수를 줄여보세요 (5 → 2)
        - Test 최소 수익률 조건을 완화해보세요 (-50 → -100)
        - 최대 허용 MDD를 0(제한없음)으로 설정해보세요
        """)


# ══════════════════════════════════════════════════════════
# Tab 5: 구간 스트레스 테스트 (단일 전략)
# ══════════════════════════════════════════════════════════
with tab5:
    st.header("📊 구간 스트레스 테스트")
    st.caption("현재 사이드바 설정 기준으로 5/10/15/20년 구간별 성과를 확인합니다.")

    p_stress = _collect_params()

    if st.button("🔬 구간 분석 실행", type="primary", use_container_width=True):
        pseudo_presets = {"현재 전략": _collect_params_dict()}
        prog = st.progress(0)
        stress_df = run_period_stress_test(pseudo_presets, progress_placeholder=prog)
        set_state("single_stress", stress_df)

    single_stress = get_state("single_stress")
    if single_stress is not None and not single_stress.empty:
        st.dataframe(single_stress, use_container_width=True)

        # 수익률 막대차트
        ret_cols = [c for c in single_stress.columns if "수익률" in str(c)]
        if ret_cols:
            vals = []
            for c in ret_cols:
                v = single_stress.iloc[0][c]
                try:
                    vals.append(float(str(v).replace("%", "").split("(")[0]))
                except Exception:
                    vals.append(None)

            fig_yr = go.Figure(go.Bar(
                x=[str(c[1]) if isinstance(c, tuple) else str(c) for c in ret_cols],
                y=vals,
                marker_color=["#26a69a" if (v or 0) > 0 else "#ef5350" for v in vals],
            ))
            fig_yr.update_layout(
                title="구간별 수익률", yaxis_title="%",
                height=300, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_yr, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 6: 매매일지
# ══════════════════════════════════════════════════════════
with tab6:
    st.header("📓 매매일지")

    sub_j1, sub_j2 = st.tabs(["✍️ 기록 추가", "📖 전체 조회"])

    with sub_j1:
        col1, col2 = st.columns(2)
        with col1:
            j_date   = st.date_input("날짜", value=datetime.date.today(), key="j_date")
            j_ticker = st.text_input("종목", key="j_ticker")
            j_signal = st.selectbox("신호", ["BUY", "SELL"], key="j_signal")
        with col2:
            j_price  = st.number_input("체결가", min_value=0.0, step=0.01, key="j_price")
            j_qty    = st.number_input("수량",   min_value=0,   step=1,    key="j_qty")
            j_memo   = st.text_area("메모", height=68, key="j_memo")

        buy_amt  = j_price * j_qty
        cur_px   = st.number_input("현재가", min_value=0.0, step=0.01, key="j_cur_px")
        pnl_pct  = ((cur_px - j_price) / j_price * 100) if j_price > 0 else 0.0
        pnl_sign = "+" if pnl_pct >= 0 else ""
        st.caption(f"📊 매수금액: ₩{buy_amt:,.0f}  |  평가손익: {pnl_sign}{pnl_pct:.2f}%")

        if st.button("💾 매매일지 저장", type="primary", use_container_width=True):
            row = {
                "날짜":       str(j_date),
                "종목":       j_ticker,
                "신호":       j_signal,
                "체결가":     j_price,
                "수량":       j_qty,
                "매수금액":   buy_amt,
                "현재가":     cur_px,
                "평가손익(%)": f"{pnl_sign}{pnl_pct:.2f}",
                "메모":       j_memo,
            }
            save_journal_row(
                get_state("sheet_name"),
                get_state("sheet_tab") + "_일지",
                row
            )

    with sub_j2:
        if st.button("📥 일지 불러오기", use_container_width=True):
            j_df = load_journal(
                get_state("sheet_name"),
                get_state("sheet_tab") + "_일지",
            )
            set_state("journal_df", j_df)

        j_df = get_state("journal_df")
        if j_df is not None and not j_df.empty:
            st.dataframe(j_df, use_container_width=True, hide_index=True)

            # 수익률 요약
            try:
                pnl_col = j_df["평가손익(%)"].str.replace("%", "").str.replace("+", "").astype(float)
                wins = (pnl_col > 0).sum()
                total = len(pnl_col)
                avg_pnl = pnl_col.mean()
                st.info(f"📊 총 {total}건  |  승 {wins}건 / 패 {total-wins}건  |  평균 손익: {avg_pnl:+.2f}%")
            except Exception:
                pass
        elif j_df is not None:
            st.info("저장된 매매일지가 없습니다.")
