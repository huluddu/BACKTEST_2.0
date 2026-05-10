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
    make_full_search_space, make_simple_search_space,
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
    "_ma_buy":        50,
    "_ma_sell":       10,
    "_off_cl_buy":    1,
    "_off_ma_buy":    1,
    "_off_cl_sell":   1,
    "_off_ma_sell":   1,
    "_buy_op":        ">",
    "_sell_op":       "<",
    "_use_trend_buy": True,
    "_use_trend_sell":False,
    "_ma_ts":         20,
    "_ma_tl":         50,
    "_stop_pct":      0,
    "_tp_pct":        0,
    "_use_atr_stop":  False,
    "_atr_mult":      2.0,
    "_apply_pending": False,   # 최적화 결과 적용 대기 플래그
})

# ══════════════════════════════════════════════════════════
# 최적화 결과 적용 처리 (위젯 렌더링 전에 실행해야 함)
# ══════════════════════════════════════════════════════════
if st.session_state.get("_apply_pending"):
    st.session_state["_apply_pending"] = False
    # 위젯 키에 직접 값 주입 (렌더링 전이므로 허용됨)
    for wk, sk in [
        ("ma_buy",        "_ma_buy"),
        ("ma_sell",       "_ma_sell"),
        ("off_cl_buy",    "_off_cl_buy"),
        ("off_ma_buy",    "_off_ma_buy"),
        ("off_cl_sell",   "_off_cl_sell"),
        ("off_ma_sell",   "_off_ma_sell"),
        ("buy_op",        "_buy_op"),
        ("sell_op",       "_sell_op"),
        ("use_trend_buy", "_use_trend_buy"),
        ("use_trend_sell","_use_trend_sell"),
        ("ma_ts",         "_ma_ts"),
        ("ma_tl",         "_ma_tl"),
        ("stop_pct",      "_stop_pct"),
        ("tp_pct",        "_tp_pct"),
        ("use_atr_stop",  "_use_atr_stop"),
        ("atr_mult",      "_atr_mult"),
    ]:
        st.session_state[wk] = st.session_state[sk]

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
        ma_buy = st.number_input(
            "매수 이평선 (직접 입력)", min_value=1, max_value=500,
            value=int(st.session_state["_ma_buy"]), step=1, key="ma_buy")
        buy_operator = st.selectbox(
            "매수 연산자", [">", "<"],
            index=[">","<"].index(st.session_state["_buy_op"])
                  if st.session_state["_buy_op"] in [">","<"] else 0,
            key="buy_op")
        col1, col2 = st.columns(2)
        with col1:
            offset_cl_buy = st.number_input(
                "종가 오프셋(매수)", min_value=1, max_value=60,
                value=int(st.session_state["_off_cl_buy"]), step=1, key="off_cl_buy")
        with col2:
            offset_ma_buy = st.number_input(
                "MA 오프셋(매수)", min_value=1, max_value=60,
                value=int(st.session_state["_off_ma_buy"]), step=1, key="off_ma_buy")
        st.session_state["_ma_buy"]     = int(ma_buy)
        st.session_state["_buy_op"]     = buy_operator
        st.session_state["_off_cl_buy"] = int(offset_cl_buy)
        st.session_state["_off_ma_buy"] = int(offset_ma_buy)

    with st.expander("📉 매도 조건"):
        sell_operator = st.selectbox(
            "매도 연산자", ["<", ">", "OFF"],
            index=["<",">","OFF"].index(st.session_state["_sell_op"])
                  if st.session_state["_sell_op"] in ["<",">","OFF"] else 0,
            key="sell_op")
        ma_sell = st.number_input(
            "매도 이평선 (직접 입력)", min_value=1, max_value=500,
            value=int(st.session_state["_ma_sell"]), step=1, key="ma_sell")
        col1, col2 = st.columns(2)
        with col1:
            offset_cl_sell = st.number_input(
                "종가 오프셋(매도)", min_value=1, max_value=60,
                value=int(st.session_state["_off_cl_sell"]), step=1, key="off_cl_sell")
        with col2:
            offset_ma_sell = st.number_input(
                "MA 오프셋(매도)", min_value=1, max_value=60,
                value=int(st.session_state["_off_ma_sell"]), step=1, key="off_ma_sell")
        st.session_state["_ma_sell"]     = int(ma_sell)
        st.session_state["_sell_op"]     = sell_operator
        st.session_state["_off_cl_sell"] = int(offset_cl_sell)
        st.session_state["_off_ma_sell"] = int(offset_ma_sell)

    with st.expander("🔀 추세 필터"):
        use_trend_buy = st.toggle(
            "매수 시 추세 필터",
            value=bool(st.session_state["_use_trend_buy"]), key="use_trend_buy")
        use_trend_sell = st.toggle(
            "매도 시 역추세 필터",
            value=bool(st.session_state["_use_trend_sell"]), key="use_trend_sell")
        col1, col2 = st.columns(2)
        with col1:
            ma_ts  = st.number_input(
                "단기 추세선", min_value=1, max_value=500,
                value=int(st.session_state["_ma_ts"]), step=1, key="ma_ts")
            off_ts = st.number_input("단기 오프셋", min_value=1, max_value=60, value=1, step=1, key="off_ts")
        with col2:
            ma_tl  = st.number_input(
                "장기 추세선", min_value=1, max_value=500,
                value=int(st.session_state["_ma_tl"]), step=1, key="ma_tl")
            off_tl = st.number_input("장기 오프셋", min_value=1, max_value=60, value=1, step=1, key="off_tl")
        st.session_state["_use_trend_buy"]  = use_trend_buy
        st.session_state["_use_trend_sell"] = use_trend_sell
        st.session_state["_ma_ts"]          = int(ma_ts)
        st.session_state["_ma_tl"]          = int(ma_tl)

    with st.expander("🎯 볼린저 밴드"):
        use_bb = st.toggle("볼린저 밴드 모드", value=False, key="use_bb")
        if use_bb:
            bb_period = st.slider("BB 기간", 10, 60, 20, key="bb_period")
            bb_std    = st.slider("BB 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std")
            bb_entry  = st.selectbox("BB 진입 기준", [
                "상단선 돌파 (추세)", "하단선 이탈 (역추세)", "중심선 돌파"], key="bb_entry")
            bb_exit   = st.selectbox("BB 청산 기준", [
                "중심선(MA) 이탈", "상단선 복귀", "하단선 이탈"], key="bb_exit")
        else:
            bb_period, bb_std = 20, 2.0
            bb_entry = "상단선 돌파 (추세)"
            bb_exit  = "중심선(MA) 이탈"

    with st.expander("📊 MACD 필터"):
        use_macd = st.toggle("MACD 필터 사용", value=False, key="use_macd")
        if use_macd:
            col1, col2, col3 = st.columns(3)
            with col1:
                macd_fast   = st.number_input("MACD Fast",   value=12, min_value=2, key="macd_fast")
            with col2:
                macd_slow   = st.number_input("MACD Slow",   value=26, min_value=2, key="macd_slow")
            with col3:
                macd_signal = st.number_input("MACD Signal", value=9,  min_value=2, key="macd_signal")
            macd_mode = st.selectbox("MACD 신호 방식", ["히스토그램 양전환", "골든크로스"], key="macd_mode")
        else:
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            macd_mode = "히스토그램 양전환"

    with st.expander("📉 RSI 필터"):
        use_rsi = st.toggle("RSI 필터 사용", value=False, key="use_rsi")
        if use_rsi:
            rsi_period = st.slider("RSI 기간", 5, 30, 14, key="rsi_period")
            rsi_min, rsi_max = st.slider("RSI 허용 범위", 0, 100, (30, 70), key="rsi_range")
        else:
            rsi_period, rsi_min, rsi_max = 14, 30, 70

    with st.expander("🌍 시장 필터"):
        use_mkt  = st.toggle("시장 필터 사용", value=False, key="use_mkt")
        mkt_ma_p = st.slider("시장 MA 기간", 50, 300, 200, key="mkt_ma_p") if use_mkt else 200

    with st.expander("🛡 손절 / 익절"):
        use_atr_stop = st.toggle(
            "ATR 손절 사용",
            value=bool(st.session_state["_use_atr_stop"]), key="use_atr_stop")
        if use_atr_stop:
            atr_mult = st.slider("ATR 배수", 1.0, 5.0,
                                 float(st.session_state["_atr_mult"]), 0.1, key="atr_mult")
            stop_pct = 0.0
        else:
            atr_mult = 2.0
            stop_pct = st.slider("고정 손절(%)", 0, 50,
                                 int(st.session_state["_stop_pct"]), key="stop_pct")
        tp_pct   = st.slider("익절(%)", 0, 100,
                             int(st.session_state["_tp_pct"]), key="tp_pct")
        min_hold = st.slider("최소 보유일", 0, 30, 0, key="min_hold")
        st.session_state["_use_atr_stop"] = use_atr_stop
        st.session_state["_atr_mult"]     = atr_mult
        st.session_state["_stop_pct"]     = int(stop_pct)
        st.session_state["_tp_pct"]       = int(tp_pct)

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

    # 구글 시트 미연결 시 안내
    has_gsheet = bool(sheet_name and sheet_tab)
    if not has_gsheet:
        st.caption("⚠️ 구글 시트 미연결: 앱 재시작 시 사라지는 임시 저장입니다.")
    else:
        st.caption(f"✅ 구글 시트 저장: `{sheet_name}` > `{sheet_tab}`")

    save_name = st.text_input("전략 이름", placeholder="예: SOXL_MA50_추세", key="save_name")
    if st.button("💾 현재 설정 저장", use_container_width=True):
        if not save_name:
            st.warning("전략 이름을 입력해주세요")
        else:
            # session_state에서 직접 읽어서 dict 구성 (함수 호출 불필요)
            params_dict = {
                "signal_ticker_input": st.session_state.get("sig_ticker", "SOXL"),
                "trade_ticker_input":  st.session_state.get("trd_ticker", "SOXL"),
                "market_ticker_input": st.session_state.get("mkt_ticker", "SPY"),
                "ma_buy":              st.session_state.get("ma_buy", 50),
                "buy_operator":        st.session_state.get("buy_op", ">"),
                "offset_cl_buy":       st.session_state.get("off_cl_buy", 1),
                "offset_ma_buy":       st.session_state.get("off_ma_buy", 1),
                "ma_sell":             st.session_state.get("ma_sell", 10),
                "sell_operator":       st.session_state.get("sell_op", "<"),
                "offset_cl_sell":      st.session_state.get("off_cl_sell", 1),
                "offset_ma_sell":      st.session_state.get("off_ma_sell", 1),
                "use_trend_in_buy":    st.session_state.get("use_trend_buy", True),
                "use_trend_in_sell":   st.session_state.get("use_trend_sell", False),
                "ma_compare_short":    st.session_state.get("ma_ts", 20),
                "ma_compare_long":     st.session_state.get("ma_tl", 50),
                "offset_compare_short": st.session_state.get("off_ts", 1),
                "offset_compare_long": st.session_state.get("off_tl", 1),
                "use_bollinger":       st.session_state.get("use_bb", False),
                "bb_period":           st.session_state.get("bb_period", 20),
                "bb_std":              st.session_state.get("bb_std", 2.0),
                "bb_entry_type":       st.session_state.get("bb_entry", "상단선 돌파 (추세)"),
                "bb_exit_type":        st.session_state.get("bb_exit", "중심선(MA) 이탈"),
                "use_macd":            st.session_state.get("use_macd", False),
                "macd_fast":           st.session_state.get("macd_fast", 12),
                "macd_slow":           st.session_state.get("macd_slow", 26),
                "macd_signal_period":  st.session_state.get("macd_signal", 9),
                "macd_mode":           st.session_state.get("macd_mode", "히스토그램 양전환"),
                "use_rsi_filter":      st.session_state.get("use_rsi", False),
                "rsi_period":          st.session_state.get("rsi_period", 14),
                "rsi_min":             30,
                "rsi_max":             70,
                "use_market_filter":   st.session_state.get("use_mkt", False),
                "market_ma_period":    st.session_state.get("mkt_ma_p", 200),
                "use_atr_stop":        st.session_state.get("use_atr_stop", False),
                "atr_multiplier":      st.session_state.get("atr_mult", 2.0),
                "stop_loss_pct":       st.session_state.get("stop_pct", 0),
                "take_profit_pct":     st.session_state.get("tp_pct", 0),
                "min_hold_days":       st.session_state.get("min_hold", 0),
                "strategy_behavior":   st.session_state.get("strategy_behavior", "priority_sell"),
                "initial_cash":        st.session_state.get("init_cash", 5000000),
                "fee_bps":             st.session_state.get("fee_bps", 25),
                "slip_bps":            st.session_state.get("slip_bps", 1),
            }
            presets = get_state("presets")
            presets[save_name] = params_dict
            set_state("presets", presets)
            ok = save_strategy(sheet_name, sheet_tab, save_name, params_dict)
            if not ok and not has_gsheet:
                st.info("💡 구글 시트에 영구 저장하려면 시트 이름과 탭을 설정하세요.")


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

            st.caption(
                f"📅 기준일(종료일): **{end_date}**  |  "
                f"신호 계산 종가: **{sig['sig_date']}** (오프셋{p.offset_cl_buy})  |  "
                f"추세: {'📈 상승추세' if sig['trend_up'] else '📉 하락추세'}"
            )

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
                log_df.style.map(
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
    st.header("⚡ 전략 최적화 (Optuna 베이지안 AI 풀옵션)")
    st.caption("Train/Test 분리 검증으로 과적합을 방지하며 최적 파라미터를 자동 탐색합니다.")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        n_trials   = st.slider("탐색 횟수 (많을수록 정확)", 30, 500, 100, step=10, key="opt_n_trials")
        opt_target = st.selectbox("최적화 목표", [
            "수익률 (%)", "다중 목적 (수익률↑ + MDD↓)", "Profit Factor", "승률 (%)", "MDD 최소화"
        ], key="opt_target")
    with col2:
        split_ratio = st.slider("Train 비율 (앞부분)", 0.3, 0.8, 0.6, 0.05, key="opt_split")
        min_trades  = st.slider("최소 매매 횟수 (필터)", 1, 30, 5, key="opt_min_trades")
    with col3:
        max_mdd      = st.slider("최대 허용 MDD (%) (0=제한없음)", 0, 100, 0, key="opt_max_mdd")
        min_test_ret = st.slider("Test 구간 최소 수익률 (%)", -100, 100, -50, key="opt_min_test")

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

        # 탐색 공간: 항상 AI 풀옵션 (MA = [1]+range(5,121,5))
        ss = make_full_search_space(
            ma_choices=None,           # None → _MA_FULL 자동 사용
            use_trend=ai_use_trend,
            use_atr=ai_use_atr,
        )
        p_base.use_bollinger      = ai_use_bb
        p_base.use_macd           = ai_use_macd
        p_base.use_rsi_filter     = ai_use_rsi
        p_base.use_market_filter  = ai_use_mkt

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
        set_state("opt_result",      opt_df)
        set_state("opt_study",       opt_study)
        set_state("opt_target_used", opt_target)

    # ── 결과 표시 ─────────────────────────────────────────
    opt_df       = get_state("opt_result")
    opt_target_u = get_state("opt_target_used", "수익률 (%)")
    is_multi     = (opt_target_u == "다중 목적 (수익률↑ + MDD↓)")

    if opt_df is not None and not opt_df.empty:
        label = "Pareto Front 후보" if is_multi else "유효 결과"
        st.success(f"✅ {len(opt_df)}개 {label} 발견")
        if is_multi:
            st.info("수익률↑ + MDD↓ 동시 최적화 결과입니다. 공격형 ~ 안정형 중 마음에 드는 것을 선택하세요.")

        res_cols   = ["Full_수익률(%)", "Full_MDD(%)", "Full_승률(%)", "Full_PF",
                      "Full_매매횟수", "Train_수익률(%)", "Test_수익률(%)", "Test_MDD(%)"]
        param_cols = [c for c in opt_df.columns if c not in res_cols]

        rtab1, rtab2 = st.tabs(["📊 성과 지표", "⚙️ 파라미터"])
        with rtab1:
            st.dataframe(opt_df[res_cols].head(20), use_container_width=True, hide_index=True)
        with rtab2:
            st.dataframe(opt_df[param_cols].head(20), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🏆 최적 파라미터 사이드바 적용")
        st.caption("적용 버튼을 누르면 사이드바 값이 변경됩니다. 이후 Tab 3에서 백테스트를 실행하세요.")

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

        with st.expander("📋 선택된 파라미터 미리보기"):
            selected_row = opt_df.iloc[top_idx]
            preview_items = [(k, v) for k, v in selected_row.items() if k in param_cols]
            pc = st.columns(3)
            for i, (k, v) in enumerate(preview_items):
                pc[i % 3].metric(k, v)

        if st.button("✅ 사이드바에 적용하기", type="primary", use_container_width=True, key="opt_apply_btn"):
            row = opt_df.iloc[top_idx]

            def _si(v, d):
                try: return int(float(v))
                except: return d
            def _sf(v, d):
                try: return float(v)
                except: return d
            def _sb(v, d=False):
                return str(v).lower() in ["true", "1", "t"]

            # _ 키에 새 값 저장
            st.session_state["_ma_buy"]        = _si(row.get("ma_buy"), 50)
            st.session_state["_ma_sell"]       = _si(row.get("ma_sell"), 10)
            st.session_state["_off_cl_buy"]    = _si(row.get("offset_cl_buy"), 1)
            st.session_state["_off_ma_buy"]    = _si(row.get("offset_ma_buy"), 1)
            st.session_state["_off_cl_sell"]   = _si(row.get("offset_cl_sell"), 1)
            st.session_state["_off_ma_sell"]   = _si(row.get("offset_ma_sell"), 1)
            st.session_state["_buy_op"]        = str(row.get("buy_operator", ">"))
            st.session_state["_sell_op"]       = str(row.get("sell_operator", "<"))
            st.session_state["_use_trend_buy"] = _sb(row.get("use_trend_buy"))
            st.session_state["_use_trend_sell"]= _sb(row.get("use_trend_sell"))
            st.session_state["_ma_ts"]         = _si(row.get("ma_trend_short"), 20)
            st.session_state["_ma_tl"]         = _si(row.get("ma_trend_long"), 50)
            st.session_state["_stop_pct"]      = _si(row.get("stop_loss_pct"), 0)
            st.session_state["_tp_pct"]        = _si(row.get("take_profit_pct"), 0)
            st.session_state["_use_atr_stop"]  = _sb(row.get("use_atr_stop"))
            st.session_state["_atr_mult"]      = _sf(row.get("atr_multiplier"), 2.0)

            # 플래그 설정 → 다음 실행 사이클 최상단에서 위젯 키에 주입됨
            st.session_state["_apply_pending"] = True
            st.rerun()

    elif opt_df is not None:
        st.warning("유효한 최적화 결과가 없습니다.")
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
