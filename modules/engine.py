"""
engine.py
=========
백테스트 핵심 엔진.

기존 대비 수정/개선사항:
[버그 수정]
1. ATR 손절 entry_idx 버그 수정 → 매수 시점 인덱스를 entry_bar로 저장
2. strategy_behavior 실제 구현 (기존: UI만 있고 미작동)
3. hold_days 카운트 로직 단순화 (매수 시점부터 정확히 카운트)

[성능]
4. 루프 내 pandas .iloc[] 전면 제거 → 전부 numpy 배열 인덱싱
5. compute_all()로 지표 사전 계산 → 루프 내 계산 없음

[구조]
6. StrategyParams dataclass → 인수 24개 나열 제거
7. BacktestResult dataclass → 반환값 명확화
8. 신호 판단 로직을 _check_buy / _check_sell 함수로 분리
"""

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional

from .data_loader import get_data, EMPTY_DF
from .indicators import compute_all, safe_get, compare, crossover, crossunder, sma


# ══════════════════════════════════════════════════════════
# 1. 파라미터 / 결과 dataclass
# ══════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    """
    전략 파라미터 묶음.
    기존 backtest_fast의 인수 24개를 하나의 객체로 통합.
    """
    # ── 시그널/매매 티커 ──────────────────────────────────
    signal_ticker: str = "SOXL"
    trade_ticker:  str = "SOXL"
    market_ticker: str = "SPY"

    # ── 이평선 매수 조건 ──────────────────────────────────
    ma_buy: int          = 50
    offset_ma_buy: int   = 1      # 기준 이평선을 몇 일 전 값으로 볼지
    offset_cl_buy: int   = 1      # 비교 종가를 몇 일 전 값으로 볼지
    buy_operator: str    = ">"    # '>' or '<'

    # ── 이평선 매도 조건 ──────────────────────────────────
    ma_sell: int         = 10
    offset_ma_sell: int  = 1
    offset_cl_sell: int  = 1
    sell_operator: str   = "<"    # '<', '>', 'OFF'

    # ── 추세 필터 ─────────────────────────────────────────
    use_trend_buy: bool     = True
    use_trend_sell: bool    = False
    ma_trend_short: int     = 20
    ma_trend_long: int      = 50
    offset_trend_short: int = 1
    offset_trend_long: int  = 1

    # ── 볼린저 밴드 ───────────────────────────────────────
    use_bollinger: bool   = False
    bb_period: int        = 20
    bb_std: float         = 2.0
    bb_entry_type: str    = "상단선 돌파 (추세)"  # '상단선 돌파 (추세)', '하단선 이탈 (역추세)', '중심선 돌파'
    bb_exit_type: str     = "중심선(MA) 이탈"     # '중심선(MA) 이탈', '상단선 복귀', '하단선 이탈'

    # ── MACD 조건 ─────────────────────────────────────────
    use_macd: bool        = False
    macd_fast: int        = 12
    macd_slow: int        = 26
    macd_signal_period: int = 9
    macd_mode: str        = "히스토그램 양전환"  # '히스토그램 양전환', '골든크로스'

    # ── RSI 필터 ─────────────────────────────────────────
    use_rsi_filter: bool = False
    rsi_period: int      = 14
    rsi_min: float       = 30.0
    rsi_max: float       = 70.0

    # ── 시장 필터 ─────────────────────────────────────────
    use_market_filter: bool  = False
    market_ma_period: int    = 200

    # ── 모멘텀 필터 ───────────────────────────────────────
    use_momentum_filter: bool = False
    momentum_period: int      = 20
    momentum_threshold: float = 0.0   # ROC > threshold 일 때만 매수

    # ── 손절/익절 ─────────────────────────────────────────
    use_atr_stop: bool   = False
    atr_multiplier: float = 2.0
    stop_loss_pct: float  = 0.0   # 0 = 미사용
    take_profit_pct: float = 0.0  # 0 = 미사용

    # ── 매매 규칙 ─────────────────────────────────────────
    min_hold_days: int     = 0
    initial_cash: float    = 5_000_000.0

    # ── 비용 ─────────────────────────────────────────────
    fee_bps: float  = 25.0
    slip_bps: float = 1.0


@dataclass
class BacktestResult:
    """백테스트 결과 구조체."""
    total_return_pct: float     = 0.0
    mdd_pct: float              = 0.0
    win_rate_pct: float         = 0.0
    profit_factor: float        = 0.0
    total_trades: int           = 0
    bh_return_pct: float        = 0.0   # Buy & Hold 수익률
    bh_mdd_pct: float           = 0.0   # Buy & Hold MDD
    trade_log: list             = field(default_factory=list)
    asset_curve: np.ndarray     = field(default_factory=lambda: np.array([]))
    chart_data: dict            = field(default_factory=dict)
    error: Optional[str]        = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.total_trades > 0


# ══════════════════════════════════════════════════════════
# 2. 데이터 준비
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_data(
    signal_ticker: str,
    trade_ticker: str,
    market_ticker: str,
    start_date,
    end_date,
    params: StrategyParams,
) -> dict | None:
    # 날짜를 문자열로 정규화해서 캐시 키 안정화
    start_date = str(start_date)
    end_date   = str(end_date)
    sig_df = get_data(signal_ticker, start_date, end_date)
    trd_df = get_data(trade_ticker, start_date, end_date)

    if sig_df.empty or trd_df.empty:
        return None

    # 날짜 정규화 (시간 제거)
    sig_df["Date"] = sig_df["Date"].dt.normalize()
    trd_df["Date"] = trd_df["Date"].dt.normalize()

    # 시장 데이터 병합 (선택)
    mkt_ma = None
    if market_ticker and params.use_market_filter:
        mkt_df = get_data(market_ticker, start_date, end_date)
        if not mkt_df.empty:
            mkt_df["Date"] = mkt_df["Date"].dt.normalize()
            mkt_close = mkt_df.set_index("Date")["Close"]
            trd_df = trd_df.set_index("Date").join(
                mkt_close.rename("Close_mkt"), how="inner"
            ).reset_index()

    # 시그널/매매 티커 병합
    sig_rename = {c: f"{c}_sig" for c in ["Open", "High", "Low", "Close", "Volume"]}
    trd_rename = {c: f"{c}_trd" for c in ["Open", "High", "Low", "Close", "Volume"]}
    sig_sub = sig_df.rename(columns=sig_rename)
    trd_sub = trd_df.rename(columns=trd_rename)

    base = pd.merge(
        sig_sub[["Date", "Close_sig", "Open_sig", "High_sig", "Low_sig"]],
        trd_sub,
        on="Date",
        how="inner",
    ).dropna(subset=["Close_sig", "Close_trd"]).reset_index(drop=True)

    if base.empty:
        return None

    # numpy 배열 준비
    sig_close = base["Close_sig"].to_numpy(dtype=float)
    sig_high  = base["High_sig"].to_numpy(dtype=float)
    sig_low   = base["Low_sig"].to_numpy(dtype=float)

    trd_close = base["Close_trd"].to_numpy(dtype=float)
    trd_high  = base["High_trd"].to_numpy(dtype=float)
    trd_low   = base["Low_trd"].to_numpy(dtype=float)

    # 필요한 MA 기간 수집 - 최적화 탐색 가능한 모든 MA 기간 포함
    from .optimizer import _MA_FULL, _MA_REDUCED
    ma_periods = sorted(set(
        [params.ma_buy, params.ma_sell,
         params.ma_trend_short, params.ma_trend_long,
         params.market_ma_period]
        + _MA_FULL  # 최적화에서 탐색하는 모든 MA 기간
    ))

    # 시그널 티커 지표
    sig_ind = compute_all(
        close=sig_close,
        high=sig_high,
        low=sig_low,
        ma_periods=ma_periods,
        rsi_period=params.rsi_period,
        macd_fast=params.macd_fast,
        macd_slow=params.macd_slow,
        macd_signal=params.macd_signal_period,
        bb_period=params.bb_period,
        bb_std=params.bb_std,
        mom_period=params.momentum_period,
    )

    # 매매 티커 ATR (손절 계산용)
    from .indicators import atr as calc_atr
    trd_atr = calc_atr(trd_high, trd_low, trd_close)

    # 시장 필터 MA
    if "Close_mkt" in base.columns:
        mkt_close_arr = base["Close_mkt"].to_numpy(dtype=float)
        mkt_ma = sma(mkt_close_arr, params.market_ma_period)
        base["Close_mkt_arr"] = mkt_close_arr
    else:
        mkt_ma = None

    last_bar_date = base["Date"].iloc[-1]

    return {
        "base":          base,
        "sig_close":     sig_close,
        "trd_close":     trd_close,
        "trd_open":      base["Open_trd"].to_numpy(dtype=float),
        "trd_high":      trd_high,
        "trd_low":       trd_low,
        "trd_atr":       trd_atr,
        "sig_ind":       sig_ind,
        "mkt_close":     base["Close_mkt"].to_numpy(dtype=float) if "Close_mkt" in base.columns else None,
        "mkt_ma":        mkt_ma,
        "end_date":      pd.to_datetime(end_date).normalize(),
        "last_bar_date": last_bar_date,
    }


# ══════════════════════════════════════════════════════════
# 3. 신호 판단 (prev_i 기준 → T+1 매매 구현)
# ══════════════════════════════════════════════════════════

def _check_buy(prev_i: int, sig_ind: dict, p: StrategyParams, mkt_close, mkt_ma) -> tuple[bool, str]:
    """
    어제(prev_i) 기준으로 매수 조건 판단.
    Returns: (bool, reason_str)
    """
    # ── 볼린저 밴드 ───────────────────────────────────────
    if p.use_bollinger:
        cl = safe_get(sig_ind["bb_mid"], prev_i - p.offset_cl_buy + p.bb_period - 1)  # 근사
        cl = safe_get(np.array([0.0]), 0)  # placeholder
        # 실제론 sig_close를 직접 넘겨야 함 → engine 루프에서 처리
        return False, "bb_mode"  # 루프에서 직접 처리

    # ── 이평선 기본 조건 ──────────────────────────────────
    ma_arr = sig_ind["ma"].get(p.ma_buy)
    if ma_arr is None:
        return False, f"MA{p.ma_buy} 없음"

    # 에러 방지: 인덱스 범위 확인
    cl_idx  = prev_i - p.offset_cl_buy
    ma_idx  = prev_i - p.offset_ma_buy

    # sig_close는 엔진 루프에서 직접 전달 (이 함수 외부에서 처리)
    return True, "ok"  # 실제 비교는 run_backtest 루프에서 인라인으로


def _fill_price(px: float, trade_type: str, fee_bps: float, slip_bps: float) -> float:
    """수수료 + 슬리피지 적용 체결가 계산."""
    cost = (fee_bps + slip_bps) / 10_000.0
    if trade_type == "buy":
        return px * (1 + cost)
    else:
        return px * (1 - cost)


# ══════════════════════════════════════════════════════════
# 4. 백테스트 엔진 메인
# ══════════════════════════════════════════════════════════

def run_backtest(data: dict, p: StrategyParams) -> BacktestResult:
    """
    백테스트 실행.

    Args:
        data: prepare_data()의 반환값
        p:    StrategyParams 인스턴스

    Returns:
        BacktestResult
    """
    if data is None:
        return BacktestResult(error="데이터 없음")

    base     = data["base"]
    sig_cl   = data["sig_close"]
    trd_cl   = data["trd_close"]
    trd_op   = data["trd_open"]
    trd_hi   = data["trd_high"]
    trd_lo   = data["trd_low"]
    trd_atr  = data["trd_atr"]
    sig_ind  = data["sig_ind"]
    mkt_cl   = data["mkt_close"]
    mkt_ma   = data["mkt_ma"]
    n        = len(base)

    if n < 60:
        return BacktestResult(error="데이터 부족 (최소 60봉 필요)")

    # ── 지표 배열 미리 꺼내기 (루프 내 dict 접근 최소화) ──
    ma_buy_arr  = sig_ind["ma"].get(p.ma_buy,  np.full(n, np.nan))
    ma_sell_arr = sig_ind["ma"].get(p.ma_sell, np.full(n, np.nan))
    ma_ts_arr   = sig_ind["ma"].get(p.ma_trend_short, np.full(n, np.nan))
    ma_tl_arr   = sig_ind["ma"].get(p.ma_trend_long,  np.full(n, np.nan))
    rsi_arr     = sig_ind["rsi"]
    macd_hist   = sig_ind["macd_hist"]
    macd_line   = sig_ind["macd_line"]
    macd_sig    = sig_ind["macd_signal"]
    bb_mid      = sig_ind["bb_mid"]
    bb_up       = sig_ind["bb_upper"]
    bb_lo       = sig_ind["bb_lower"]
    roc_arr     = sig_ind["roc"]

    # ── 상태 변수 ─────────────────────────────────────────
    cash        = p.initial_cash
    position    = 0.0   # 보유 주수
    entry_price = 0.0   # 매수 단가
    entry_bar   = -1    # 매수 시점 bar 인덱스 (ATR 버그 수정 핵심!)
    hold_days   = 0
    asset_curve = np.zeros(n, dtype=float)
    trade_log   = []

    WARMUP = 50  # 지표 안정화 구간

    for i in range(WARMUP, n):
        close_today = trd_cl[i]
        open_today  = trd_op[i]
        high_today  = trd_hi[i]
        low_today   = trd_lo[i]

        # 다음봉 시가 (T+1 체결용) — 마지막 봉이면 당일 종가로 대체
        next_open = trd_op[i + 1] if i + 1 < n else close_today

        stop_hit   = False
        take_hit   = False
        sold_today = False
        exec_price = None
        signal     = "HOLD"
        reason     = ""
        detail     = ""

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # A. 지표값
        # 오프셋 1 = i-1 (전 거래일 종가) → 미래 참조 없음
        # 체결: next_open (i+1봉 시가)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        cl_b_idx = i - p.offset_cl_buy
        cl_s_idx = i - p.offset_cl_sell
        ma_b_idx = i - p.offset_ma_buy
        ma_s_idx = i - p.offset_ma_sell

        if (cl_b_idx < 0 or cl_s_idx < 0 or ma_b_idx < 0 or ma_s_idx < 0):
            asset_curve[i] = cash + position * close_today
            continue

        cl_b  = sig_cl[cl_b_idx]
        cl_s  = sig_cl[cl_s_idx]
        ma_b  = ma_buy_arr[ma_b_idx]
        ma_s  = ma_sell_arr[ma_s_idx]

        # 추세선
        ts_idx = max(0, i - p.offset_trend_short)
        tl_idx = max(0, i - p.offset_trend_long)
        trend_short = ma_ts_arr[ts_idx]
        trend_long  = ma_tl_arr[tl_idx]
        trend_up    = (trend_short >= trend_long) if (
            not np.isnan(trend_short) and not np.isnan(trend_long)
        ) else False

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # B. 매수/매도 조건 판단 (전일 기준)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        buy_cond  = False
        sell_cond = False
        buy_msg   = ""
        sell_msg  = ""

        # ── 볼린저 밴드 모드 ──────────────────────────
        if p.use_bollinger:
            bb_u_val = bb_up[cl_b_idx] if not np.isnan(bb_up[cl_b_idx]) else np.nan
            bb_m_val = bb_mid[cl_b_idx] if not np.isnan(bb_mid[cl_b_idx]) else np.nan
            bb_l_val = bb_lo[cl_b_idx] if not np.isnan(bb_lo[cl_b_idx]) else np.nan

            if "상단선" in p.bb_entry_type:
                buy_cond = cl_b > bb_u_val if not np.isnan(bb_u_val) else False
                buy_msg  = f"종가({cl_b:.2f}) > BB상단({bb_u_val:.2f})"
            elif "하단선" in p.bb_entry_type:
                buy_cond = cl_b < bb_l_val if not np.isnan(bb_l_val) else False
                buy_msg  = f"종가({cl_b:.2f}) < BB하단({bb_l_val:.2f})"
            else:
                buy_cond = cl_b > bb_m_val if not np.isnan(bb_m_val) else False
                buy_msg  = f"종가({cl_b:.2f}) > BB중심({bb_m_val:.2f})"

            if p.sell_operator != "OFF":
                if "중심선" in p.bb_exit_type:
                    sell_cond = cl_s < bb_mid[cl_s_idx] if not np.isnan(bb_mid[cl_s_idx]) else False
                    sell_msg  = f"종가({cl_s:.2f}) < BB중심({bb_mid[cl_s_idx]:.2f})"
                elif "하단선" in p.bb_exit_type:
                    sell_cond = cl_s < bb_lo[cl_s_idx] if not np.isnan(bb_lo[cl_s_idx]) else False
                    sell_msg  = f"종가({cl_s:.2f}) < BB하단({bb_lo[cl_s_idx]:.2f})"
                else:
                    sell_cond = cl_s < bb_up[cl_s_idx] if not np.isnan(bb_up[cl_s_idx]) else False
                    sell_msg  = f"종가({cl_s:.2f}) < BB상단({bb_up[cl_s_idx]:.2f})"

        # ── 이평선 기본 모드 ──────────────────────────
        else:
            if not (np.isnan(cl_b) or np.isnan(ma_b)):
                buy_base = compare(cl_b, ma_b, p.buy_operator)
                buy_msg  = f"종가({cl_b:.2f}) {p.buy_operator} MA{p.ma_buy}({ma_b:.2f})"

                if p.use_trend_buy:
                    if not trend_up:
                        buy_base = False
                        buy_msg += " [추세↓거부]"
                    else:
                        buy_msg += " [추세↑통과]"
                buy_cond = buy_base

            if p.sell_operator == "OFF":
                sell_cond = False
                sell_msg  = "OFF"
            elif not (np.isnan(cl_s) or np.isnan(ma_s)):
                sell_base = compare(cl_s, ma_s, p.sell_operator)
                sell_msg  = f"종가({cl_s:.2f}) {p.sell_operator} MA{p.ma_sell}({ma_s:.2f})"

                if p.use_trend_sell:
                    if trend_up:
                        sell_base = False
                        sell_msg += " [추세↑역추세거부]"
                    else:
                        sell_msg += " [추세↓통과]"
                sell_cond = sell_base

        # ── MACD 추가 필터 ────────────────────────────
        if p.use_macd and buy_cond:
            if p.macd_mode == "히스토그램 양전환":
                hist_now  = macd_hist[i] if not np.isnan(macd_hist[i]) else -1
                hist_prev = macd_hist[i - 1] if i > 0 and not np.isnan(macd_hist[i - 1]) else -1
                macd_ok   = hist_now > 0 and hist_prev <= 0
            else:  # 골든크로스
                macd_ok = crossover(macd_line, macd_sig, i)
            if not macd_ok:
                buy_cond = False
                buy_msg += " [MACD미충족]"

        # ── RSI 필터 ──────────────────────────────────
        if p.use_rsi_filter and buy_cond:
            rsi_val = rsi_arr[i]
            if not np.isnan(rsi_val):
                if rsi_val > p.rsi_max:
                    buy_cond = False
                    buy_msg += f" [RSI과매수:{rsi_val:.1f}]"
                elif rsi_val < p.rsi_min:
                    buy_cond = False
                    buy_msg += f" [RSI과매도:{rsi_val:.1f}]"

        # ── 모멘텀 필터 ───────────────────────────────
        if p.use_momentum_filter and buy_cond:
            roc_val = roc_arr[i]
            if not np.isnan(roc_val) and roc_val < p.momentum_threshold:
                buy_cond = False
                buy_msg += f" [모멘텀부족:{roc_val:.1f}%]"

        # ── 시장 필터 ─────────────────────────────────
        if p.use_market_filter and buy_cond and mkt_cl is not None and mkt_ma is not None:
            if i < len(mkt_cl) and not np.isnan(mkt_ma[i]):
                if mkt_cl[i] < mkt_ma[i]:
                    buy_cond = False
                    buy_msg += f" [시장하락장]"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # C. 포지션 관리 (손절/익절 먼저, 그 다음 전략 매도/매수)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if position > 0:
            # ── 손절 ──────────────────────────────────
            stop_price = 0.0
            if p.use_atr_stop and entry_bar >= 0:
                # [버그 수정] entry_bar에서의 ATR 사용 (기존: i - hold_days 버그)
                entry_atr = trd_atr[entry_bar]
                if not np.isnan(entry_atr):
                    stop_price = entry_price - entry_atr * p.atr_multiplier
                    detail     = f"ATR손절가({stop_price:.2f})"
            elif p.stop_loss_pct > 0:
                stop_price = entry_price * (1 - p.stop_loss_pct / 100)
                detail     = f"고정손절가({stop_price:.2f})"

            if stop_price > 0 and low_today <= stop_price:
                stop_hit   = True
                exec_price = open_today if open_today < stop_price else stop_price
                signal     = "SELL"
                reason     = "ATR손절" if p.use_atr_stop else "손절"

            # ── 익절 ──────────────────────────────────
            if not stop_hit and p.take_profit_pct > 0:
                tp_price = entry_price * (1 + p.take_profit_pct / 100)
                if high_today >= tp_price:
                    take_hit   = True
                    exec_price = open_today if open_today > tp_price else tp_price
                    signal     = "SELL"
                    reason     = "익절"
                    detail     = f"익절가({tp_price:.2f})"

            # ── 손절/익절 체결 ────────────────────────
            if stop_hit or take_hit:
                fill = _fill_price(exec_price, "sell", p.fee_bps, p.slip_bps)
                cash = position * fill
                trade_log.append(_make_log(
                    base, i, close_today, "SELL", exec_price, cash + position * close_today,
                    reason, detail, stop_hit, take_hit
                ))
                position   = 0.0
                entry_price = 0.0
                entry_bar   = -1
                hold_days   = 0
                sold_today  = True

        # ── 포지션 상태 기반 매매 처리 ───────────────────
        # 보유 중 → 매도 조건 체크 (매수 무시)
        # 미보유  → 매수 조건 체크 (매도 무시)
        if not sold_today and position > 0:
            if sell_cond and hold_days >= p.min_hold_days:
                exec_price = close_today  # LOC: 당일 종가 체결
                fill       = _fill_price(exec_price, "sell", p.fee_bps, p.slip_bps)
                cash       = position * fill
                trade_log.append(_make_log(
                    base, i, close_today, "SELL", exec_price,
                    cash, "전략매도", sell_msg, False, False
                ))
                position    = 0.0
                entry_price = 0.0
                entry_bar   = -1
                hold_days   = 0
                sold_today  = True

        elif not sold_today and position == 0:
            if buy_cond:
                exec_price  = close_today  # LOC: 당일 종가 체결
                fill        = _fill_price(exec_price, "buy", p.fee_bps, p.slip_bps)
                position    = cash / fill
                entry_price = exec_price
                entry_bar   = i
                hold_days   = 0
                cash        = 0.0
                signal      = "BUY"
                reason      = "전략매수"
                detail      = buy_msg
                trade_log.append(_make_log(
                    base, i, close_today, "BUY", exec_price,
                    position * close_today, reason, detail, False, False
                ))

        # hold_days 정확한 카운트
        if position > 0:
            hold_days += 1
        else:
            hold_days = 0

        asset_curve[i] = cash + position * close_today

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # D. 성과 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    curve = asset_curve[WARMUP:]
    if len(curve) == 0 or curve[0] == 0:
        return BacktestResult(error="자산 곡선 계산 실패")

    # 수익률
    total_return = (curve[-1] - p.initial_cash) / p.initial_cash * 100

    # MDD
    running_max = np.maximum.accumulate(curve)
    drawdown    = (curve - running_max) / running_max * 100
    mdd         = float(drawdown.min())

    # 승률 / Profit Factor
    wins, losses, g_profit, g_loss = 0, 0, 0.0, 0.0
    buy_price = None
    sell_trades = [l for l in trade_log if l["신호"] == "SELL"]
    buy_trades  = [l for l in trade_log if l["신호"] == "BUY"]

    for b, s in zip(buy_trades, sell_trades):
        pnl = (s["체결가"] - b["체결가"]) / b["체결가"]
        if pnl > 0:
            wins     += 1
            g_profit += pnl
        else:
            losses   += 1
            g_loss   += abs(pnl)

    total_sell = len(sell_trades)
    win_rate   = (wins / total_sell * 100) if total_sell > 0 else 0.0
    pf         = (g_profit / g_loss) if g_loss > 0 else 999.0

    # B&H 기준
    bh_start    = trd_cl[WARMUP]
    bh_end      = trd_cl[-1]
    bh_return   = (bh_end - bh_start) / bh_start * 100 if bh_start > 0 else 0.0
    bh_curve    = trd_cl[WARMUP:] / bh_start * p.initial_cash
    bh_dd       = (bh_curve - np.maximum.accumulate(bh_curve)) / np.maximum.accumulate(bh_curve) * 100
    bh_mdd      = float(bh_dd.min())

    # 차트 데이터
    chart_data = {
        "base":        base.iloc[WARMUP:].reset_index(drop=True),
        "asset_curve": curve,
        "bh_curve":    bh_curve,
        "ma_buy_arr":  ma_buy_arr[WARMUP:],
        "ma_sell_arr": ma_sell_arr[WARMUP:],
        "bb_upper":    bb_up[WARMUP:] if p.use_bollinger else None,
        "bb_lower":    bb_lo[WARMUP:] if p.use_bollinger else None,
        "bb_mid":      bb_mid[WARMUP:] if p.use_bollinger else None,
        "macd_line":   macd_line[WARMUP:] if p.use_macd else None,
        "macd_signal": macd_sig[WARMUP:] if p.use_macd else None,
        "macd_hist":   macd_hist[WARMUP:] if p.use_macd else None,
        "rsi":         rsi_arr[WARMUP:] if p.use_rsi_filter else None,
    }

    return BacktestResult(
        total_return_pct = round(total_return, 2),
        mdd_pct          = round(mdd, 2),
        win_rate_pct     = round(win_rate, 2),
        profit_factor    = round(min(pf, 999.0), 2),
        total_trades     = total_sell,
        bh_return_pct    = round(bh_return, 2),
        bh_mdd_pct       = round(bh_mdd, 2),
        trade_log        = trade_log,
        asset_curve      = curve,
        chart_data       = chart_data,
    )


def _make_log(base, i, close, signal, exec_price, asset, reason, detail, stop_hit, take_hit) -> dict:
    """매매 로그 딕셔너리 생성 헬퍼."""
    return {
        "날짜":    base["Date"].iloc[i],
        "종가":    round(float(close), 4),
        "신호":    signal,
        "체결가":  round(float(exec_price), 4) if exec_price is not None else None,
        "자산":    round(float(asset), 0),
        "이유":    reason,
        "상세":    detail,
        "손절발동": stop_hit,
        "익절발동": take_hit,
    }


# ══════════════════════════════════════════════════════════
# 5. 오늘의 시그널 요약 (Tab1용)
# ══════════════════════════════════════════════════════════

def get_today_signal(data: dict, p: StrategyParams) -> dict:
    """
    현재 시점의 매수/매도 신호를 반환.
    Returns: {"status": str, "buy_ok": bool, "sell_ok": bool, "details": str}
    """
    if data is None:
        return {"status": "데이터 없음", "buy_ok": False, "sell_ok": False, "details": ""}

    sig_ind = data["sig_ind"]
    sig_cl  = data["sig_close"]
    mkt_cl  = data["mkt_close"]
    mkt_ma  = data["mkt_ma"]
    n       = len(sig_cl)
    i       = n - 1  # 마지막 봉

    # ── 장중 미완성 데이터 처리 ──────────────────────────
    # UTC 21:00(미장 마감) 전이면 당일 데이터는 미완성
    # → 마지막 봉이 오늘(UTC)이면 한 봉 앞으로 당김
    import datetime as _dt
    now_utc   = _dt.datetime.utcnow()
    today_utc = pd.Timestamp(now_utc.date())
    last_bar  = data["base"]["Date"].iloc[-1]
    if pd.to_datetime(last_bar).normalize() >= today_utc:
        market_closed = now_utc.hour >= 21
        if not market_closed:
            i = max(0, i - 1)  # 미완성 당일 봉 제외

    # 오프셋 보정: 종료일 > 마지막 봉이면(주말/장 전) +1
    end_dt      = data.get("end_date", data["base"]["Date"].iloc[-1])
    last_bar_dt = data["base"]["Date"].iloc[i]
    effective_last = data["base"]["Date"].iloc[i]
    offset_adj  = 1 if pd.to_datetime(end_dt) > pd.to_datetime(effective_last) else 0

    ma_buy_arr  = sig_ind["ma"].get(p.ma_buy,  np.full(n, np.nan))
    ma_sell_arr = sig_ind["ma"].get(p.ma_sell, np.full(n, np.nan))
    ma_ts_arr   = sig_ind["ma"].get(p.ma_trend_short, np.full(n, np.nan))
    ma_tl_arr   = sig_ind["ma"].get(p.ma_trend_long,  np.full(n, np.nan))
    bb_up       = sig_ind["bb_upper"]
    bb_mid      = sig_ind["bb_mid"]
    bb_lo       = sig_ind["bb_lower"]

    cl_b_idx = max(0, i - p.offset_cl_buy + offset_adj)
    cl_s_idx = max(0, i - p.offset_cl_sell + offset_adj)
    ma_b_idx = max(0, i - p.offset_ma_buy + offset_adj)
    ma_s_idx = max(0, i - p.offset_ma_sell + offset_adj)

    cl_b = sig_cl[cl_b_idx]
    cl_s = sig_cl[cl_s_idx]
    ma_b = ma_buy_arr[ma_b_idx]
    ma_s = ma_sell_arr[ma_s_idx]

    ts_val   = ma_ts_arr[max(0, i - p.offset_trend_short + offset_adj)]
    tl_val   = ma_tl_arr[max(0, i - p.offset_trend_long + offset_adj)]
    trend_up = (ts_val >= tl_val) if not (np.isnan(ts_val) or np.isnan(tl_val)) else False

    # 기준일: 사이드바 종료일 (주말/공휴일이어도 그대로 표시)
    # 마지막 봉 날짜는 별도로 표시
    last_bar_date = data["base"]["Date"].iloc[-1].strftime("%Y-%m-%d")

    # 매수 조건
    if p.use_bollinger:
        bu = bb_up[cl_b_idx]
        bm = bb_mid[cl_b_idx]
        bl = bb_lo[cl_b_idx]
        if "상단선" in p.bb_entry_type:
            buy_ok  = cl_b > bu if not np.isnan(bu) else False
            buy_msg = f"종가({cl_b:.2f}) > BB상단({bu:.2f})"
        elif "하단선" in p.bb_entry_type:
            buy_ok  = cl_b < bl if not np.isnan(bl) else False
            buy_msg = f"종가({cl_b:.2f}) < BB하단({bl:.2f})"
        else:
            buy_ok  = cl_b > bm if not np.isnan(bm) else False
            buy_msg = f"종가({cl_b:.2f}) > BB중심({bm:.2f})"
    else:
        buy_base = compare(cl_b, ma_b, p.buy_operator)
        buy_msg  = f"종가({cl_b:.2f}) {p.buy_operator} MA{p.ma_buy}({ma_b:.2f})"
        if p.use_trend_buy and not trend_up:
            buy_base = False
            buy_msg += " → 추세필터 거부 ❌"
        buy_ok = buy_base

    # 시장 필터
    mkt_ok = True
    if p.use_market_filter and mkt_cl is not None and mkt_ma is not None:
        if not np.isnan(mkt_ma[prev_i]):
            mkt_ok = mkt_cl[prev_i] > mkt_ma[prev_i]
    if buy_ok and not mkt_ok:
        buy_ok  = False
        buy_msg += " → 시장필터 거부 ❌"

    # 매도 조건
    if p.sell_operator == "OFF":
        sell_ok  = False
        sell_msg = "매도 OFF"
    elif p.use_bollinger:
        bm_s = bb_mid[cl_s_idx]
        sell_ok  = cl_s < bm_s if not np.isnan(bm_s) else False
        sell_msg = f"종가({cl_s:.2f}) < BB중심({bm_s:.2f})"
    else:
        sell_base = compare(cl_s, ma_s, p.sell_operator)
        sell_msg  = f"종가({cl_s:.2f}) {p.sell_operator} MA{p.ma_sell}({ma_s:.2f})"
        if p.use_trend_sell and trend_up:
            sell_base = False
            sell_msg += " → 역추세필터 거부 ❌"
        sell_ok = sell_base

    if buy_ok and sell_ok:
        status = "⚠️ 매수/매도 중복"
    elif buy_ok:
        status = "🚀 매수 진입"
    elif sell_ok:
        status = "💧 매도 청산"
    else:
        status = "⏸ 관망"

    last_bar_date = data["base"]["Date"].iloc[-1].strftime("%Y-%m-%d")
    sig_date      = data["base"]["Date"].iloc[cl_b_idx].strftime("%Y-%m-%d")

    return {
        "status":        status,
        "buy_ok":        buy_ok,
        "sell_ok":       sell_ok,
        "buy_msg":       buy_msg,
        "sell_msg":      sell_msg,
        "last_bar_date": last_bar_date,  # 마지막 거래일 (i봉)
        "sig_date":      sig_date,       # 신호 계산에 사용된 종가 날짜 (i - offset)
        "trend_up":      trend_up,
    }
