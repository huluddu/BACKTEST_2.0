"""
portfolio_sim.py
================
공동계좌 선착순 포트폴리오 시뮬레이션.

설계 규칙:
- 전략 등록 순서로 동시 매수 신호 우선순위 결정
- 전략매도(종가 청산) 당일 다른 전략 매수 허용 (당일 종가 재사용)
- 익절/손절(장중 청산) 당일 다른 전략 매수 허용
- 현금 없어 건너뛴 거래는 skipped_log에 기록
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .engine import StrategyParams, prepare_data, _fill_price
from .indicators import compare, crossover, crossunder


# ══════════════════════════════════════════════════════════
# 1. 결과 dataclass
# ══════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    is_valid:        bool  = False
    error:           str   = ""

    # 전체 성과
    total_return_pct: float = 0.0
    cagr_pct:         float = 0.0
    mdd_pct:          float = 0.0
    sharpe:           float = 0.0
    calmar:           float = 0.0

    # 곡선
    dates:            list  = field(default_factory=list)
    equity_curve:     list  = field(default_factory=list)   # 전체 자산
    strategy_curves:  dict  = field(default_factory=dict)   # {전략명: [자산...]}
    weight_curves:    dict  = field(default_factory=dict)   # {전략명: [비중%...]}
    cash_curve:       list  = field(default_factory=list)   # 현금 곡선

    # 연도별 수익률
    yearly_returns:   pd.DataFrame = field(default_factory=pd.DataFrame)

    # 매매 로그
    trade_log:        list  = field(default_factory=list)   # 전체 체결 기록
    skipped_log:      list  = field(default_factory=list)   # 건너뛴 거래 기록

    # 겹침 분석
    overlap_periods:  list  = field(default_factory=list)   # 동시보유 구간
    overlap_days:     int   = 0

    # 현금 대기 비율
    cash_ratio_pct:   float = 0.0

    # 전략명 목록 (순서 보존)
    strategy_names:   list  = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 2. 신호 추출 (engine.py 루프 로직 재현)
# ══════════════════════════════════════════════════════════

def extract_signals(data: dict, p: StrategyParams) -> dict:
    """
    전략의 날짜별 매수/매도 신호를 추출.
    Returns:
        {
            "dates":     list[pd.Timestamp],
            "buy":       np.ndarray[bool],
            "sell":      np.ndarray[bool],
            "buy_msg":   list[str],
            "sell_msg":  list[str],
            "trd_close": np.ndarray,
            "trd_open":  np.ndarray,
            "trd_high":  np.ndarray,
            "trd_low":   np.ndarray,
            "trd_atr":   np.ndarray,
            "base":      DataFrame,
        }
    """
    base    = data["base"]
    sig_cl  = data["sig_close"]
    sig_ind = data["sig_ind"]
    mkt_cl  = data["mkt_close"]
    mkt_ma  = data["mkt_ma"]
    n       = len(base)

    trd_cl  = data["trd_close"]
    trd_op  = data["trd_open"]
    trd_hi  = data["trd_high"]
    trd_lo  = data["trd_low"]
    trd_atr = data["trd_atr"]

    # 지표 배열
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
    dc_upper    = sig_ind["dc_upper"]
    dc_lower    = sig_ind["dc_lower"]
    dc_exit_upper = sig_ind["dc_exit_upper"]
    dc_exit_lower = sig_ind["dc_exit_lower"]
    swing_res   = sig_ind["swing_resistance"]
    swing_sup   = sig_ind["swing_support"]
    vp_poc      = sig_ind["vp_poc"]
    vp_vah      = sig_ind["vp_vah"]
    vp_val      = sig_ind["vp_val"]

    WARMUP = 50
    buy_arr  = np.zeros(n, dtype=bool)
    sell_arr = np.zeros(n, dtype=bool)
    buy_msgs  = [""] * n
    sell_msgs = [""] * n

    for i in range(WARMUP, n):
        cl_b_idx = i - p.offset_cl_buy
        cl_s_idx = i - p.offset_cl_sell
        ma_b_idx = i - p.offset_ma_buy
        ma_s_idx = i - p.offset_ma_sell

        if cl_b_idx < 0 or cl_s_idx < 0 or ma_b_idx < 0 or ma_s_idx < 0:
            continue

        cl_b = sig_cl[cl_b_idx]
        cl_s = sig_cl[cl_s_idx]
        ma_b = ma_buy_arr[ma_b_idx]
        ma_s = ma_sell_arr[ma_s_idx]

        ts_idx     = max(0, i - p.offset_trend_short)
        tl_idx     = max(0, i - p.offset_trend_long)
        trend_short = ma_ts_arr[ts_idx]
        trend_long  = ma_tl_arr[tl_idx]
        trend_up    = (trend_short >= trend_long) if (
            not np.isnan(trend_short) and not np.isnan(trend_long)
        ) else False

        buy_cond  = False
        sell_cond = False
        buy_msg   = ""
        sell_msg  = ""

        # ── 도니챈 ───────────────────────────────────
        if p.use_donchian:
            dc_ref_b  = max(0, cl_b_idx - 1)
            dc_ref_s  = max(0, cl_s_idx - 1)
            dc_u_prev = dc_upper[dc_ref_b]
            dc_l_prev = dc_lower[dc_ref_b]
            dc_eu_prev = dc_exit_upper[dc_ref_s]
            dc_el_prev = dc_exit_lower[dc_ref_s]

            if p.dc_buy_mode == "상단 돌파":
                if not np.isnan(dc_u_prev) and not np.isnan(cl_b):
                    buy_cond = cl_b > dc_u_prev
                    buy_msg  = f"DC상단 돌파({cl_b:.2f}>{dc_u_prev:.2f})"
            else:
                if not np.isnan(dc_l_prev) and not np.isnan(cl_b):
                    buy_cond = cl_b < dc_l_prev
                    buy_msg  = f"DC하단 이탈({cl_b:.2f}<{dc_l_prev:.2f})"

            if p.dc_sell_mode == "OFF":
                sell_cond = False
            elif p.dc_sell_mode == "하단 이탈":
                if not np.isnan(dc_el_prev) and not np.isnan(cl_s):
                    sell_cond = cl_s < dc_el_prev
                    sell_msg  = f"DC하단 이탈({cl_s:.2f}<{dc_el_prev:.2f})"
            else:
                if not np.isnan(dc_eu_prev) and not np.isnan(cl_s):
                    sell_cond = cl_s > dc_eu_prev
                    sell_msg  = f"DC상단 이탈({cl_s:.2f}>{dc_eu_prev:.2f})"

        # ── 스윙 ─────────────────────────────────────
        elif p.use_swing:
            sw_res_val = swing_res[cl_b_idx]
            sw_sup_val = swing_sup[cl_b_idx]
            sw_res_s   = swing_res[cl_s_idx]
            sw_sup_s   = swing_sup[cl_s_idx]

            if p.swing_buy_mode == "저항 돌파":
                if not np.isnan(sw_res_val) and not np.isnan(cl_b):
                    buy_cond = cl_b > sw_res_val
                    buy_msg  = f"스윙저항 돌파({cl_b:.2f}>{sw_res_val:.2f})"
            else:
                if not np.isnan(sw_sup_val) and not np.isnan(cl_b):
                    near_sup = abs(cl_b - sw_sup_val) / sw_sup_val <= 0.01
                    buy_cond = near_sup and cl_b >= sw_sup_val
                    buy_msg  = f"스윙지지 반등({cl_b:.2f}≈{sw_sup_val:.2f})"

            if p.swing_sell_mode == "OFF":
                sell_cond = False
            elif p.swing_sell_mode == "지지 이탈":
                if not np.isnan(sw_sup_s) and not np.isnan(cl_s):
                    sell_cond = cl_s < sw_sup_s
                    sell_msg  = f"스윙지지 이탈({cl_s:.2f}<{sw_sup_s:.2f})"
            else:
                if not np.isnan(sw_res_s) and not np.isnan(cl_s):
                    sell_cond = cl_s >= sw_res_s
                    sell_msg  = f"스윙저항 도달({cl_s:.2f}>={sw_res_s:.2f})"

        # ── 볼린저 ───────────────────────────────────
        elif p.use_bollinger:
            bb_u_val = bb_up[cl_b_idx]
            bb_l_val = bb_lo[cl_b_idx]
            bb_m_val = bb_mid[cl_b_idx]

            if "상단선" in p.bb_entry_type:
                buy_cond = cl_b > bb_u_val if not np.isnan(bb_u_val) else False
                buy_msg  = f"BB상단 돌파({cl_b:.2f}>{bb_u_val:.2f})"
            elif "하단선" in p.bb_entry_type:
                buy_cond = cl_b < bb_l_val if not np.isnan(bb_l_val) else False
                buy_msg  = f"BB하단 이탈({cl_b:.2f}<{bb_l_val:.2f})"
            else:
                buy_cond = cl_b > bb_m_val if not np.isnan(bb_m_val) else False
                buy_msg  = f"BB중심 돌파({cl_b:.2f}>{bb_m_val:.2f})"

            if p.sell_operator != "OFF":
                if "중심선" in p.bb_exit_type:
                    sell_cond = cl_s < bb_mid[cl_s_idx] if not np.isnan(bb_mid[cl_s_idx]) else False
                    sell_msg  = f"BB중심 이탈({cl_s:.2f}<{bb_mid[cl_s_idx]:.2f})"
                elif "하단선" in p.bb_exit_type:
                    sell_cond = cl_s < bb_lo[cl_s_idx] if not np.isnan(bb_lo[cl_s_idx]) else False
                    sell_msg  = f"BB하단 이탈({cl_s:.2f}<{bb_lo[cl_s_idx]:.2f})"
                else:
                    sell_cond = cl_s < bb_up[cl_s_idx] if not np.isnan(bb_up[cl_s_idx]) else False
                    sell_msg  = f"BB상단 이탈({cl_s:.2f}<{bb_up[cl_s_idx]:.2f})"

        # ── VP ───────────────────────────────────────
        elif p.use_vp:
            vp_poc_b = vp_poc[cl_b_idx]
            vp_val_b = vp_val[cl_b_idx]
            vp_vah_b = vp_vah[cl_b_idx]
            vp_poc_s = vp_poc[cl_s_idx]
            vp_val_s = vp_val[cl_s_idx]

            if p.vp_buy_mode == "POC 상향 돌파":
                if not np.isnan(vp_poc_b):
                    buy_cond = cl_b > vp_poc_b
                    buy_msg  = f"VP POC 돌파({cl_b:.2f}>{vp_poc_b:.2f})"
            elif p.vp_buy_mode == "VAL 지지 반등":
                if not np.isnan(vp_val_b):
                    near_val = abs(cl_b - vp_val_b) / vp_val_b <= 0.01
                    buy_cond = near_val and cl_b >= vp_val_b
                    buy_msg  = f"VP VAL 반등({cl_b:.2f}≈{vp_val_b:.2f})"
            else:
                if not np.isnan(vp_vah_b):
                    buy_cond = cl_b > vp_vah_b
                    buy_msg  = f"VP VAH 돌파({cl_b:.2f}>{vp_vah_b:.2f})"

            if p.vp_sell_mode == "OFF":
                sell_cond = False
            elif p.vp_sell_mode == "POC 하향 이탈":
                if not np.isnan(vp_poc_s):
                    sell_cond = cl_s < vp_poc_s
                    sell_msg  = f"VP POC 이탈({cl_s:.2f}<{vp_poc_s:.2f})"
            else:
                if not np.isnan(vp_val_s):
                    sell_cond = cl_s < vp_val_s
                    sell_msg  = f"VP VAL 이탈({cl_s:.2f}<{vp_val_s:.2f})"

        # ── 이평선 기본 ──────────────────────────────
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

        # ── MACD 필터 ────────────────────────────────
        if p.use_macd and buy_cond:
            if p.macd_mode == "히스토그램 양전환":
                hist_now  = macd_hist[i] if not np.isnan(macd_hist[i]) else -1
                hist_prev = macd_hist[i-1] if i > 0 and not np.isnan(macd_hist[i-1]) else -1
                macd_ok   = hist_now > 0 and hist_prev <= 0
            else:
                macd_ok = crossover(macd_line, macd_sig, i)
            if not macd_ok:
                buy_cond = False
                buy_msg += " [MACD미충족]"

        # ── RSI 필터 ─────────────────────────────────
        if p.use_rsi_filter and buy_cond:
            rsi_val = rsi_arr[i]
            if not np.isnan(rsi_val):
                if rsi_val > p.rsi_max:
                    buy_cond = False
                    buy_msg += f" [RSI과매수:{rsi_val:.1f}]"
                elif rsi_val < p.rsi_min:
                    buy_cond = False
                    buy_msg += f" [RSI과매도:{rsi_val:.1f}]"

        # ── 시장 필터 ────────────────────────────────
        if p.use_market_filter and buy_cond and mkt_cl is not None and mkt_ma is not None:
            if i < len(mkt_cl) and not np.isnan(mkt_ma[i]):
                if mkt_cl[i] < mkt_ma[i]:
                    buy_cond = False
                    buy_msg += " [시장하락장]"

        buy_arr[i]   = buy_cond
        sell_arr[i]  = sell_cond
        buy_msgs[i]  = buy_msg
        sell_msgs[i] = sell_msg

    dates = list(base["Date"])
    return {
        "dates":     dates,
        "buy":       buy_arr,
        "sell":      sell_arr,
        "buy_msg":   buy_msgs,
        "sell_msg":  sell_msgs,
        "trd_close": trd_cl,
        "trd_open":  trd_op,
        "trd_high":  trd_hi,
        "trd_low":   trd_lo,
        "trd_atr":   trd_atr,
        "base":      base,
        "n":         n,
    }


# ══════════════════════════════════════════════════════════
# 3. 공동계좌 선착순 시뮬
# ══════════════════════════════════════════════════════════

def run_portfolio_sim(
    strategy_configs: list,   # [{"name": str, "data": dict, "params": StrategyParams}, ...]
    initial_cash: float = 10_000_000.0,
    progress_cb = None,
) -> PortfolioResult:
    """
    공동계좌 선착순 포트폴리오 시뮬레이션.

    Args:
        strategy_configs: 전략 목록 (등록 순서 = 우선순위)
        initial_cash:     초기 투자금
        progress_cb:      (step, total, msg) → None

    Returns:
        PortfolioResult
    """
    if not strategy_configs:
        return PortfolioResult(error="전략이 없습니다.")
    if len(strategy_configs) < 2:
        return PortfolioResult(error="전략을 2개 이상 선택하세요.")

    n_strat = len(strategy_configs)
    names   = [c["name"] for c in strategy_configs]

    # ── 1. 신호 추출 ─────────────────────────────────────
    if progress_cb:
        progress_cb(0, n_strat + 2, "신호 추출 중...")

    signals = []
    for idx, cfg in enumerate(strategy_configs):
        if progress_cb:
            progress_cb(idx, n_strat + 2, f"[{cfg['name']}] 신호 추출 중...")
        sig = extract_signals(cfg["data"], cfg["params"])
        sig["params"] = cfg["params"]
        signals.append(sig)

    # ── 2. 날짜 축 정렬 (공통 날짜만) ───────────────────
    date_sets = [set(s["dates"]) for s in signals]
    common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))
    if len(common_dates) < 10:
        return PortfolioResult(error="전략 간 공통 날짜가 너무 적습니다.")

    # 각 전략의 날짜 → 인덱스 맵
    date_to_idx = []
    for s in signals:
        d2i = {d: i for i, d in enumerate(s["dates"])}
        date_to_idx.append(d2i)

    T = len(common_dates)

    # ── 3. 공동계좌 루프 ─────────────────────────────────
    if progress_cb:
        progress_cb(n_strat, n_strat + 2, "포트폴리오 시뮬 중...")

    cash = float(initial_cash)

    # 전략별 상태
    positions    = [0.0]  * n_strat   # 보유 주수
    entry_prices = [0.0]  * n_strat   # 매수 단가
    entry_bars   = [-1]   * n_strat   # 매수 시점 bar (atr용)
    hold_days    = [0]    * n_strat

    # 기록용
    equity_arr   = np.zeros(T, dtype=float)
    cash_arr     = np.zeros(T, dtype=float)
    strat_vals   = [np.zeros(T, dtype=float) for _ in range(n_strat)]

    trade_log   = []
    skipped_log = []

    fee_bps  = strategy_configs[0]["params"].fee_bps
    slip_bps = strategy_configs[0]["params"].slip_bps

    for t, date in enumerate(common_dates):
        day_cash_freed = 0.0   # 당일 청산으로 확보된 현금
        sold_today     = set() # A단계에서 직접 추적 (당일 청산 전략 인덱스)

        # ── A. 포지션 보유 전략: 손절/익절/매도 처리 ──
        for s_idx in range(n_strat):
            if positions[s_idx] <= 0:
                continue

            sig   = signals[s_idx]
            p     = sig["params"]
            i     = date_to_idx[s_idx][date]
            trd_cl = sig["trd_close"]
            trd_op = sig["trd_open"]
            trd_hi = sig["trd_high"]
            trd_lo = sig["trd_low"]
            trd_atr = sig["trd_atr"]

            close_today = trd_cl[i]
            open_today  = trd_op[i]
            high_today  = trd_hi[i]
            low_today   = trd_lo[i]

            pos   = positions[s_idx]
            entry = entry_prices[s_idx]
            ebar  = entry_bars[s_idx]

            sold = False

            # ── 손절 ──────────────────────────────────
            stop_price = 0.0
            if p.use_atr_stop and ebar >= 0:
                entry_atr = trd_atr[ebar]
                if not np.isnan(entry_atr):
                    stop_price = entry - entry_atr * p.atr_multiplier
            elif p.stop_loss_pct > 0:
                stop_price = entry * (1 - p.stop_loss_pct / 100)

            if stop_price > 0 and low_today <= stop_price:
                exec_px = open_today if open_today < stop_price else stop_price
                fill    = _fill_price(exec_px, "sell", fee_bps, slip_bps)
                proceeds = pos * fill
                cash += proceeds
                day_cash_freed += proceeds
                trade_log.append({
                    "날짜":   date,
                    "전략":   names[s_idx],
                    "신호":   "SELL",
                    "이유":   "손절",
                    "체결가": round(exec_px, 4),
                    "수량":   round(pos, 6),
                    "금액":   round(proceeds, 0),
                    "자산":   round(cash + sum(
                        positions[k] * signals[k]["trd_close"][date_to_idx[k][date]]
                        for k in range(n_strat)
                        if positions[k] > 0 and k != s_idx
                    ), 0),
                })
                positions[s_idx]    = 0.0
                entry_prices[s_idx] = 0.0
                entry_bars[s_idx]   = -1
                hold_days[s_idx]    = 0
                sold = True
                sold_today.add(s_idx)

            # ── 익절 ──────────────────────────────────
            if not sold and p.take_profit_pct > 0:
                tp_price = entry * (1 + p.take_profit_pct / 100)
                if high_today >= tp_price:
                    exec_px = open_today if open_today > tp_price else tp_price
                    fill    = _fill_price(exec_px, "sell", fee_bps, slip_bps)
                    proceeds = pos * fill
                    cash += proceeds
                    day_cash_freed += proceeds
                    trade_log.append({
                        "날짜":   date,
                        "전략":   names[s_idx],
                        "신호":   "SELL",
                        "이유":   "익절",
                        "체결가": round(exec_px, 4),
                        "수량":   round(pos, 6),
                        "금액":   round(proceeds, 0),
                        "자산":   round(cash + sum(
                            positions[k] * signals[k]["trd_close"][date_to_idx[k][date]]
                            for k in range(n_strat)
                            if positions[k] > 0 and k != s_idx
                        ), 0),
                    })
                    positions[s_idx]    = 0.0
                    entry_prices[s_idx] = 0.0
                    entry_bars[s_idx]   = -1
                    hold_days[s_idx]    = 0
                    sold = True
                    sold_today.add(s_idx)

            # ── 전략매도 (신호 청산) ───────────────────
            if not sold and sig["sell"][i] and hold_days[s_idx] >= p.min_hold_days:
                exec_px  = close_today   # LOC 종가 체결
                fill     = _fill_price(exec_px, "sell", fee_bps, slip_bps)
                proceeds = pos * fill
                cash    += proceeds
                day_cash_freed += proceeds
                trade_log.append({
                    "날짜":   date,
                    "전략":   names[s_idx],
                    "신호":   "SELL",
                    "이유":   "전략매도",
                    "상세":   sig["sell_msg"][i],
                    "체결가": round(exec_px, 4),
                    "수량":   round(pos, 6),
                    "금액":   round(proceeds, 0),
                    "자산":   round(cash + sum(
                        positions[k] * signals[k]["trd_close"][date_to_idx[k][date]]
                        for k in range(n_strat)
                        if positions[k] > 0 and k != s_idx
                    ), 0),
                })
                positions[s_idx]    = 0.0
                entry_prices[s_idx] = 0.0
                entry_bars[s_idx]   = -1
                hold_days[s_idx]    = 0
                sold_today.add(s_idx)

        # ── B. 매수 신호 처리 ────────────────────────────────
        # 당일 청산된 전략이 먼저 재진입 (A안 우선권)
        # → 그 뒤 나머지 전략을 등록 순서대로 (선착순)
        # sold_today 는 A단계에서 직접 채워짐 (trade_log 파싱 불필요)
        buy_order = sorted(range(n_strat), key=lambda x: (0 if x in sold_today else 1, x))

        MIN_CASH = 1_000   # 1,000원 미만은 현금 없음으로 처리
        for s_idx in buy_order:
            if positions[s_idx] > 0:
                continue   # 이미 보유 중인 전략은 매수 불가

            sig  = signals[s_idx]
            p    = sig["params"]
            i    = date_to_idx[s_idx][date]

            if not sig["buy"][i]:
                continue

            if cash < MIN_CASH:
                skipped_log.append({
                    "날짜":     date,
                    "전략":     names[s_idx],
                    "이유":     "현금 부족",
                    "상세":     sig["buy_msg"][i],
                    "당시현금": round(cash, 0),
                })
                continue

            exec_px = sig["trd_close"][i]   # LOC 종가 체결
            fill    = _fill_price(exec_px, "buy", fee_bps, slip_bps)
            shares  = cash / fill
            cost    = shares * fill

            positions[s_idx]    = shares
            entry_prices[s_idx] = exec_px
            entry_bars[s_idx]   = i
            hold_days[s_idx]    = 0
            cash               -= cost   # 전액 투입

            trade_log.append({
                "날짜":   date,
                "전략":   names[s_idx],
                "신호":   "BUY",
                "이유":   "전략매수",
                "상세":   sig["buy_msg"][i],
                "체결가": round(exec_px, 4),
                "수량":   round(shares, 6),
                "금액":   round(cost, 0),
                "자산":   round(cash + sum(
                    positions[k] * signals[k]["trd_close"][date_to_idx[k][date]]
                    for k in range(n_strat) if positions[k] > 0
                ), 0),
            })
            # 선착순: 한 전략이 매수하면 당일 현금 소진 → 다음 전략은 현금 없음

        # ── C. 보유일수 카운트 & 자산 기록 ───────────
        for s_idx in range(n_strat):
            if positions[s_idx] > 0:
                hold_days[s_idx] += 1
                i = date_to_idx[s_idx][date]
                strat_vals[s_idx][t] = positions[s_idx] * signals[s_idx]["trd_close"][i]
            else:
                strat_vals[s_idx][t] = 0.0

        total_asset = cash + sum(strat_vals[s_idx][t] for s_idx in range(n_strat))
        equity_arr[t] = total_asset
        cash_arr[t]   = cash

    # ── 4. 성과 지표 계산 ────────────────────────────────
    if progress_cb:
        progress_cb(n_strat + 1, n_strat + 2, "성과 계산 중...")

    curve = equity_arr
    if curve[-1] <= 0:
        return PortfolioResult(error="시뮬레이션 실패: 자산이 0 이하")

    total_return = (curve[-1] - initial_cash) / initial_cash * 100

    # CAGR
    dates_pd = pd.to_datetime(common_dates)
    years    = (dates_pd[-1] - dates_pd[0]).days / 365.25
    cagr     = ((curve[-1] / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # MDD
    peak     = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak * 100
    mdd      = float(drawdown.min())

    # Sharpe (일간 수익률 기준, 무위험이자율 0%)
    daily_ret = np.diff(curve) / curve[:-1]
    sharpe    = (np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)
                 if np.std(daily_ret) > 0 else 0.0)

    # Calmar
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0.0

    # 현금 대기 비율
    cash_ratio = float(np.mean(cash_arr / np.maximum(equity_arr, 1))) * 100

    # ── 5. 연도별 수익률 ─────────────────────────────────
    df_curve = pd.DataFrame({
        "date":  dates_pd,
        "total": curve,
    })
    df_curve["year"] = df_curve["date"].dt.year

    # 매매 로그에서 전략별 연도별 실현손익 집계
    # - 합산(%): 연도 첫날 전체 자산 대비 마지막날 전체 자산 변화율
    # - 전략별(%): 그 해 BUY→SELL 실현 손익 합산 / 연초 전체 자산 (매매 없으면 표시 안 함)
    realized_pnl = {}   # {(year, name): 실현손익 합계}
    buy_cost_map  = {}  # {name: 가장 최근 BUY 금액}
    for rec in trade_log:
        yr_r = pd.Timestamp(rec["날짜"]).year
        nm   = rec["전략"]
        if rec["신호"] == "BUY":
            buy_cost_map[nm] = rec["금액"]
        elif rec["신호"] == "SELL":
            buy_cost = buy_cost_map.pop(nm, None)
            if buy_cost is not None:
                pnl = rec["금액"] - buy_cost
                key = (yr_r, nm)
                realized_pnl[key] = realized_pnl.get(key, 0.0) + pnl

    yearly_rows = []
    for yr, grp in df_curve.groupby("year"):
        row = {"연도": yr}
        start_total = grp["total"].iloc[0]
        end_total   = grp["total"].iloc[-1]
        if start_total > 0:
            row["합산(%)"] = round((end_total / start_total - 1) * 100, 1)
        else:
            row["합산(%)"] = None
        for name in names:
            pnl = realized_pnl.get((yr, name), None)
            if pnl is not None and start_total > 0:
                row[f"{name}(%)"] = round(pnl / start_total * 100, 1)
            else:
                row[f"{name}(%)"] = None   # 그 해 실현 거래 없음
        yearly_rows.append(row)
    yearly_df = pd.DataFrame(yearly_rows)

    # ── 6. 비중 추이 ─────────────────────────────────────
    weight_curves = {}
    for s_idx, name in enumerate(names):
        w = np.where(equity_arr > 0, strat_vals[s_idx] / equity_arr * 100, 0.0)
        weight_curves[name] = w.tolist()
    cash_weight = np.where(equity_arr > 0, cash_arr / equity_arr * 100, 100.0)

    # ── 7. 겹침 분석 ─────────────────────────────────────
    # 2개 이상 전략이 동시 보유 중인 날 → 포트폴리오 루프에서는 선착순이라
    # 실제로 겹침이 0이어야 하나, 확인용으로 남김
    holding_matrix = np.zeros((T, n_strat), dtype=bool)
    for t in range(T):
        for s_idx in range(n_strat):
            holding_matrix[t, s_idx] = strat_vals[s_idx][t] > 0

    overlap_days_count = int(np.sum(holding_matrix.sum(axis=1) >= 2))

    # 겹침 구간 목록
    overlap_periods = []
    in_overlap = False
    ov_start = None
    for t in range(T):
        n_holding = holding_matrix[t].sum()
        if n_holding >= 2 and not in_overlap:
            in_overlap = True
            ov_start   = common_dates[t]
        elif n_holding < 2 and in_overlap:
            in_overlap = False
            overlap_periods.append({
                "시작": ov_start,
                "종료": common_dates[t - 1],
                "일수": (pd.Timestamp(common_dates[t-1]) - pd.Timestamp(ov_start)).days + 1,
            })
    if in_overlap:
        overlap_periods.append({
            "시작": ov_start,
            "종료": common_dates[-1],
            "일수": (pd.Timestamp(common_dates[-1]) - pd.Timestamp(ov_start)).days + 1,
        })

    if progress_cb:
        progress_cb(n_strat + 2, n_strat + 2, "완료")

    return PortfolioResult(
        is_valid         = True,
        total_return_pct = round(total_return, 2),
        cagr_pct         = round(cagr, 2),
        mdd_pct          = round(mdd, 2),
        sharpe           = round(sharpe, 3),
        calmar           = round(calmar, 3),
        dates            = [str(d)[:10] for d in common_dates],
        equity_curve     = curve.tolist(),
        strategy_curves  = {names[s]: strat_vals[s].tolist() for s in range(n_strat)},
        weight_curves    = weight_curves,
        cash_curve       = cash_arr.tolist(),
        yearly_returns   = yearly_df,
        trade_log        = trade_log,
        skipped_log      = skipped_log,
        overlap_periods  = overlap_periods,
        overlap_days     = overlap_days_count,
        cash_ratio_pct   = round(cash_ratio, 1),
        strategy_names   = names,
    )


# ════════════════════════════════════════════════════════
# 4. 공동계좌 워크포워드 (롤링 윈도우) 분석
# ════════════════════════════════════════════════════════

@dataclass
class PortfolioWalkForwardResult:
    windows:        list  = field(default_factory=list)   # 각 구간 결과
    returns:        list  = field(default_factory=list)   # 구간별 수익률
    avg_return:     float = 0.0
    std_return:     float = 0.0
    win_rate:       float = 0.0
    total_return:   float = 0.0
    is_valid:       bool  = False


def run_portfolio_walk_forward(
    strategy_base_configs: list,  # [{"name": str, "signal_ticker": str, "trade_ticker": str, "market_ticker": str, "params": StrategyParams}]
    start_date,
    end_date,
    initial_cash:   float = 10_000_000.0,
    window_months:  int   = 12,
    step_months:    int   = 6,
    progress_cb     = None,
) -> PortfolioWalkForwardResult:
    """
    공동계좌 워크포워드 (롤링 윈도우) 분석.
    전체 기간을 window_months 구간으로 연속 실행하며
    각 구간에서 run_portfolio_sim() 실행.

    Args:
        strategy_base_configs: 전략 목록. 각 항목:
            - name: 전략명
            - signal_ticker, trade_ticker, market_ticker: 티커
            - params: StrategyParams
        start_date:    전체 시작일
        end_date:      전체 종료일
        initial_cash:  구간별 초기 투자금 (구간마다 동일)
        window_months: 구간 길이 (개월)
        step_months:   스텝 (개월)

    Returns:
        PortfolioWalkForwardResult
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    # 구간 생성
    windows = []
    cur = start_dt
    while True:
        win_end = cur + pd.DateOffset(months=window_months)
        if win_end > end_dt:
            break
        windows.append({
            "start": cur.date(),
            "end":   win_end.date(),
        })
        cur += pd.DateOffset(months=step_months)

    if not windows:
        return PortfolioWalkForwardResult(is_valid=False)

    total_w = len(windows)
    results = []

    for wi, w in enumerate(windows):
        if progress_cb:
            progress_cb(wi, total_w, f"구간 {wi+1}/{total_w}: {w['start']} ~ {w['end']}")

        try:
            # 각 구간마다 데이터 로드 후 run_portfolio_sim 실행
            configs = []
            data_ok = True
            for base_cfg in strategy_base_configs:
                try:
                    data_obj = prepare_data(
                        base_cfg["signal_ticker"],
                        base_cfg["trade_ticker"],
                        base_cfg["market_ticker"],
                        str(w["start"]), str(w["end"]),
                        base_cfg["params"],
                    )
                    if data_obj is None or len(data_obj["base"]) < 20:
                        data_ok = False
                        break
                    configs.append({
                        "name":   base_cfg["name"],
                        "data":   data_obj,
                        "params": base_cfg["params"],
                    })
                except Exception:
                    data_ok = False
                    break

            if not data_ok or len(configs) < 2:
                results.append({
                    **w,
                    "return": None, "mdd": None,
                    "trades": None, "cash_ratio": None,
                    "status": "데이터 부족",
                })
                continue

            ps_res = run_portfolio_sim(configs, initial_cash=initial_cash)
            if not ps_res.is_valid:
                results.append({
                    **w,
                    "return": None, "mdd": None,
                    "trades": None, "cash_ratio": None,
                    "status": f"실패: {ps_res.error}",
                })
                continue

            results.append({
                **w,
                "return":     round(ps_res.total_return_pct, 1),
                "cagr":       round(ps_res.cagr_pct, 1),
                "mdd":        round(ps_res.mdd_pct, 1),
                "sharpe":     round(ps_res.sharpe, 2),
                "trades":     len(ps_res.trade_log),
                "cash_ratio": round(ps_res.cash_ratio_pct, 1),
                "skipped":    len(ps_res.skipped_log),
                "status":     "완료",
            })

        except Exception as e:
            results.append({
                **w,
                "return": None, "mdd": None,
                "trades": None, "cash_ratio": None,
                "status": f"오류: {str(e)[:40]}",
            })

    if progress_cb:
        progress_cb(total_w, total_w, "완료")

    valid_returns = [r["return"] for r in results if r.get("return") is not None]
    if not valid_returns:
        return PortfolioWalkForwardResult(windows=results, is_valid=False)

    # 구간별 수익률 복리 합산 (독립 구간으로 각각 초기 투자금 기준)
    total_ret = 1.0
    for r in valid_returns:
        total_ret *= (1 + r / 100)
    total_ret = (total_ret - 1) * 100

    return PortfolioWalkForwardResult(
        windows      = results,
        returns      = valid_returns,
        avg_return   = round(float(np.mean(valid_returns)), 1),
        std_return   = round(float(np.std(valid_returns)), 1),
        win_rate     = round(sum(1 for r in valid_returns if r > 0) / len(valid_returns) * 100, 1),
        total_return = round(total_ret, 1),
        is_valid     = True,
    )
