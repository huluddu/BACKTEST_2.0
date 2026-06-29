"""
infinite_buy.py - 무한매수법 백테스트 엔진
라오어의 무한매수법을 일봉 데이터로 시뮬레이션

핵심 규칙:
- 원금 N등분 (기본 40), 매일 1회차씩 매수
- LOC 근사: 종가 < 평단이면 1회차, 종가 >= 평단이면 0.5회차
- 평단 +target_pct% 도달 시 전량 익절
- N회차 소진 시:
    평단 대비 -stop_pct% 이내 → 손절 후 새 사이클
    평단 대비 -stop_pct% 초과 → 동결(존버) + 새 원금으로 새 사이클 병렬 시작
- 동결 포지션은 익절가 도달 시 자동 청산
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class InfiniteBuyParams:
    initial_capital: float = 5_000_000.0  # 사이클당 원금
    n_splits:        int   = 40            # 분할 횟수
    target_pct:      float = 10.0          # 익절 목표 (%)
    stop_pct:        float = 10.0          # 손절 기준 (%)
    fee_bps:         float = 25.0          # 수수료 (bps)
    slip_bps:        float = 1.0           # 슬리피지 (bps)


@dataclass
class CycleResult:
    cycle_no:       int
    start_date:     str
    end_date:       str
    outcome:        str    # "익절", "손절", "동결", "진행중"
    return_pct:     float
    days:           int
    avg_price:      float
    exit_price:     float
    invested:       float  # 투입 자본


@dataclass
class InfiniteBuyResult:
    is_valid:        bool = False
    total_return_pct:float = 0.0
    total_profit:    float = 0.0
    win_rate_pct:    float = 0.0   # 익절 사이클 비율
    avg_cycle_days:  float = 0.0
    n_cycles_done:   int   = 0     # 완료 사이클 수
    n_win:           int   = 0
    n_loss:          int   = 0
    n_frozen:        int   = 0     # 존버 중 사이클 수
    total_invested:  float = 0.0   # 총 투입 자본
    mdd_pct:         float = 0.0
    asset_curve:     np.ndarray = field(default_factory=lambda: np.array([]))
    cycles:          list = field(default_factory=list)
    trade_log:       list = field(default_factory=list)


def _fill(price: float, side: str, fee_bps: float, slip_bps: float) -> float:
    bps = (fee_bps + slip_bps) / 10000
    return price * (1 + bps) if side == "buy" else price * (1 - bps)


def run_infinite_buy(
    df: pd.DataFrame,
    params: InfiniteBuyParams,
) -> InfiniteBuyResult:
    """
    df: 표준화된 일봉 DataFrame (Date, Open, High, Low, Close, Volume)
    """
    res = InfiniteBuyResult()
    if df is None or df.empty or len(df) < 5:
        return res

    dates  = df["Date"].values
    opens  = df["Open"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n      = len(df)

    unit_capital = params.initial_capital / params.n_splits
    fee_bps      = params.fee_bps
    slip_bps     = params.slip_bps

    # ── 상태 변수 ──────────────────────────────────────────
    # 활성 사이클 (병렬로 여러 개 가능 - 동결 포지션 포함)
    # 각 사이클: {shares, avg_price, invested, round, start_idx, capital}
    active_cycles = []
    frozen_cycles = []  # 동결된 포지션

    total_cash   = params.initial_capital  # 현재 운용 가능 현금
    realized_pnl = 0.0
    total_invested_ever = 0.0

    asset_curve  = np.zeros(n, dtype=float)
    trade_log    = []
    cycles_done  = []

    # 첫 사이클 시작
    def new_cycle(start_idx):
        return {
            "shares":    0.0,
            "avg_price": 0.0,
            "invested":  0.0,
            "round":     0,
            "start_idx": start_idx,
            "capital":   params.initial_capital,
            "unit":      params.initial_capital / params.n_splits,
        }

    current_cycle = new_cycle(0)

    for i in range(n):
        date_str   = str(dates[i])[:10]
        close      = closes[i]
        high       = highs[i]
        low        = lows[i]

        # ── 동결 포지션 익절 체크 ────────────────────────
        still_frozen = []
        for fc in frozen_cycles:
            tp_price = fc["avg_price"] * (1 + params.target_pct / 100)
            if high >= tp_price:
                exit_price = _fill(tp_price, "sell", fee_bps, slip_bps)
                profit = fc["shares"] * exit_price - fc["invested"]
                realized_pnl += profit
                total_cash   += fc["shares"] * exit_price
                trade_log.append({
                    "날짜": date_str, "구분": "동결익절",
                    "가격": round(exit_price, 2),
                    "수량": round(fc["shares"], 4),
                    "손익": round(profit, 0),
                })
                cycles_done.append(CycleResult(
                    cycle_no   = fc["cycle_no"],
                    start_date = fc["start_date"],
                    end_date   = date_str,
                    outcome    = "동결→익절",
                    return_pct = round(profit / fc["invested"] * 100, 2),
                    days       = i - fc["start_idx"],
                    avg_price  = fc["avg_price"],
                    exit_price = exit_price,
                    invested   = fc["invested"],
                ))
            else:
                still_frozen.append(fc)
        frozen_cycles = still_frozen

        # ── 현재 사이클 익절 체크 ────────────────────────
        if current_cycle["shares"] > 0:
            tp_price = current_cycle["avg_price"] * (1 + params.target_pct / 100)
            if high >= tp_price:
                exit_price = _fill(tp_price, "sell", fee_bps, slip_bps)
                proceeds   = current_cycle["shares"] * exit_price
                profit     = proceeds - current_cycle["invested"]
                realized_pnl += profit
                total_cash   += proceeds
                trade_log.append({
                    "날짜": date_str, "구분": "익절",
                    "가격": round(exit_price, 2),
                    "수량": round(current_cycle["shares"], 4),
                    "손익": round(profit, 0),
                })
                cycles_done.append(CycleResult(
                    cycle_no   = len(cycles_done) + 1,
                    start_date = str(dates[current_cycle["start_idx"]])[:10],
                    end_date   = date_str,
                    outcome    = "익절",
                    return_pct = round(profit / current_cycle["invested"] * 100, 2),
                    days       = i - current_cycle["start_idx"],
                    avg_price  = current_cycle["avg_price"],
                    exit_price = exit_price,
                    invested   = current_cycle["invested"],
                ))
                current_cycle = new_cycle(i + 1)
                asset_curve[i] = total_cash + sum(
                    fc["shares"] * close for fc in frozen_cycles
                )
                continue

        # ── 매수 로직 ─────────────────────────────────────
        if current_cycle["round"] < params.n_splits and total_cash >= current_cycle["unit"] * 0.5:
            avg  = current_cycle["avg_price"]

            # LOC 근사: 종가 < 평단 → 1회차, 종가 >= 평단 → 0.5회차
            if avg == 0 or close < avg:
                buy_amount = current_cycle["unit"]
            else:
                buy_amount = current_cycle["unit"] * 0.5

            buy_amount = min(buy_amount, total_cash)
            if buy_amount >= current_cycle["unit"] * 0.4:  # 최소 매수 금액 체크
                fill_price = _fill(close, "buy", fee_bps, slip_bps)
                shares     = buy_amount / fill_price
                prev_val   = current_cycle["shares"] * current_cycle["avg_price"]
                current_cycle["shares"]    += shares
                current_cycle["invested"]  += buy_amount
                current_cycle["avg_price"]  = (prev_val + shares * fill_price) / current_cycle["shares"]
                current_cycle["round"]     += 1
                total_cash                 -= buy_amount
                total_invested_ever        += buy_amount
                trade_log.append({
                    "날짜": date_str, "구분": "매수",
                    "가격": round(fill_price, 2),
                    "수량": round(shares, 4),
                    "회차": current_cycle["round"],
                })

        # ── 40회차 소진 처리 ─────────────────────────────
        if current_cycle["round"] >= params.n_splits and current_cycle["shares"] > 0:
            avg        = current_cycle["avg_price"]
            loss_pct   = (close - avg) / avg * 100

            if loss_pct >= -params.stop_pct:
                # 손절 (-10% 이내)
                exit_price = _fill(close, "sell", fee_bps, slip_bps)
                proceeds   = current_cycle["shares"] * exit_price
                profit     = proceeds - current_cycle["invested"]
                realized_pnl += profit
                total_cash   += proceeds
                trade_log.append({
                    "날짜": date_str, "구분": "손절",
                    "가격": round(exit_price, 2),
                    "수량": round(current_cycle["shares"], 4),
                    "손익": round(profit, 0),
                })
                cycles_done.append(CycleResult(
                    cycle_no   = len(cycles_done) + 1,
                    start_date = str(dates[current_cycle["start_idx"]])[:10],
                    end_date   = date_str,
                    outcome    = "손절",
                    return_pct = round(profit / current_cycle["invested"] * 100, 2),
                    days       = i - current_cycle["start_idx"],
                    avg_price  = avg,
                    exit_price = exit_price,
                    invested   = current_cycle["invested"],
                ))
                current_cycle = new_cycle(i + 1)
            else:
                # 동결 (존버) + 새 사이클 시작
                fc = dict(current_cycle)
                fc["cycle_no"]  = len(cycles_done) + len(frozen_cycles) + 1
                fc["start_date"] = str(dates[current_cycle["start_idx"]])[:10]
                frozen_cycles.append(fc)
                trade_log.append({
                    "날짜": date_str, "구분": "동결(존버)",
                    "가격": round(close, 2),
                    "수량": round(current_cycle["shares"], 4),
                    "손익": 0,
                })
                # 새 원금으로 새 사이클
                if total_cash >= params.initial_capital * 0.5:
                    current_cycle = new_cycle(i + 1)
                else:
                    current_cycle = new_cycle(i + 1)
                    current_cycle["unit"] = min(
                        current_cycle["unit"],
                        total_cash / params.n_splits
                    )

        # ── 자산 곡선 업데이트 ────────────────────────────
        live_val   = current_cycle["shares"] * close
        frozen_val = sum(fc["shares"] * close for fc in frozen_cycles)
        asset_curve[i] = total_cash + live_val + frozen_val

    # ── 미청산 포지션 처리 (백테스트 종료) ───────────────
    last_close = closes[-1]
    last_date  = str(dates[-1])[:10]

    if current_cycle["shares"] > 0:
        val    = current_cycle["shares"] * last_close
        profit = val - current_cycle["invested"]
        cycles_done.append(CycleResult(
            cycle_no   = len(cycles_done) + 1,
            start_date = str(dates[current_cycle["start_idx"]])[:10],
            end_date   = last_date,
            outcome    = "진행중",
            return_pct = round(profit / current_cycle["invested"] * 100, 2) if current_cycle["invested"] > 0 else 0,
            days       = n - current_cycle["start_idx"],
            avg_price  = current_cycle["avg_price"],
            exit_price = last_close,
            invested   = current_cycle["invested"],
        ))

    for fc in frozen_cycles:
        val    = fc["shares"] * last_close
        profit = val - fc["invested"]
        cycles_done.append(CycleResult(
            cycle_no   = fc.get("cycle_no", 0),
            start_date = fc.get("start_date", ""),
            end_date   = last_date,
            outcome    = "동결진행중",
            return_pct = round(profit / fc["invested"] * 100, 2) if fc["invested"] > 0 else 0,
            days       = n - fc["start_idx"],
            avg_price  = fc["avg_price"],
            exit_price = last_close,
            invested   = fc["invested"],
        ))

    # ── 결과 계산 ─────────────────────────────────────────
    final_assets = asset_curve[-1]
    total_profit = final_assets - params.initial_capital - sum(
        fc["invested"] for fc in frozen_cycles
    )

    done_cycles  = [c for c in cycles_done if c.outcome in ("익절", "손절", "동결→익절")]
    n_win        = sum(1 for c in done_cycles if "익절" in c.outcome)
    n_loss       = sum(1 for c in done_cycles if c.outcome == "손절")
    n_frozen_now = len(frozen_cycles)
    win_rate     = n_win / len(done_cycles) * 100 if done_cycles else 0

    avg_days = np.mean([c.days for c in done_cycles]) if done_cycles else 0

    # MDD
    peak = np.maximum.accumulate(asset_curve)
    mdd  = float(np.min((asset_curve - peak) / peak * 100)) if peak.max() > 0 else 0

    # 총 수익률 (투입 대비)
    total_ret = (final_assets - params.initial_capital) / params.initial_capital * 100

    res.is_valid         = True
    res.total_return_pct = round(total_ret, 2)
    res.total_profit     = round(total_profit, 0)
    res.win_rate_pct     = round(win_rate, 1)
    res.avg_cycle_days   = round(avg_days, 1)
    res.n_cycles_done    = len(done_cycles)
    res.n_win            = n_win
    res.n_loss           = n_loss
    res.n_frozen         = n_frozen_now
    res.total_invested   = round(total_invested_ever, 0)
    res.mdd_pct          = round(mdd, 2)
    res.asset_curve      = asset_curve
    res.cycles           = cycles_done
    res.trade_log        = trade_log

    return res
