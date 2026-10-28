"""
validation.py - 전략 검증 모듈
1. 구간별 유효성 검증 (현재 전략 파라미터 고정)
2. 파라미터 민감도 분석
"""
import copy, numpy as np, pandas as pd, datetime
from dataclasses import dataclass, field
from .engine import StrategyParams, prepare_data, run_backtest


# ══════════════════════════════════════════════════════════
# 1. 구간별 유효성 검증
# ══════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    windows:          list  = field(default_factory=list)
    oos_returns:      list  = field(default_factory=list)
    avg_oos_return:   float = 0.0
    std_oos_return:   float = 0.0
    win_rate:         float = 0.0
    total_oos_return: float = 0.0
    is_valid:         bool  = False


def run_walk_forward(
    signal_ticker: str,
    trade_ticker:  str,
    market_ticker: str,
    start_date,
    end_date,
    base_params:   StrategyParams,
    window_months: int = 12,   # 각 구간 길이 (개월)
    step_months:   int = 6,    # 스텝 (개월)
    progress_cb    = None,
) -> WalkForwardResult:
    """
    현재 파라미터를 고정한 채로 구간별 백테스트.
    "내 전략이 어느 시기에도 통하는가" 검증.
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
        return WalkForwardResult(is_valid=False)

    total = len(windows)
    results = []

    for wi, w in enumerate(windows):
        if progress_cb:
            progress_cb(wi, total, f"구간 {wi+1}/{total}: {w['start']} ~ {w['end']}")

        try:
            data = prepare_data(
                signal_ticker, trade_ticker, market_ticker,
                w["start"], w["end"], base_params
            )
            if data is None or len(data["base"]) < 20:
                results.append({**w, "return": None, "mdd": None,
                                "trades": None, "winrate": None, "status": "데이터 부족"})
                continue

            res = run_backtest(data, base_params)
            if not res.is_valid:
                results.append({**w, "return": None, "mdd": None,
                                "trades": None, "winrate": None, "status": "실패"})
                continue

            results.append({
                **w,
                "return":  round(res.total_return_pct, 1),
                "mdd":     round(res.mdd_pct, 1),
                "trades":  res.total_trades,
                "winrate": round(res.win_rate_pct, 1),
                "bh":      round(res.bh_return_pct, 1),
                "status":  "완료",
            })

        except Exception as e:
            results.append({**w, "return": None, "mdd": None,
                            "trades": None, "winrate": None, "status": f"오류"})

    if progress_cb:
        progress_cb(total, total, "완료")

    oos_returns = [r["return"] for r in results if r.get("return") is not None]
    if not oos_returns:
        return WalkForwardResult(windows=results, is_valid=False)

    total_ret = 1.0
    for r in oos_returns:
        total_ret *= (1 + r / 100)
    total_ret = (total_ret - 1) * 100

    return WalkForwardResult(
        windows          = results,
        oos_returns      = oos_returns,
        avg_oos_return   = round(float(np.mean(oos_returns)), 1),
        std_oos_return   = round(float(np.std(oos_returns)), 1),
        win_rate         = round(sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100, 1),
        total_oos_return = round(total_ret, 1),
        is_valid         = True,
    )


# ══════════════════════════════════════════════════════════
# 2. 파라미터 민감도 분석
# ══════════════════════════════════════════════════════════

@dataclass
class SensitivityResult:
    param_name:  str
    values:      list
    returns:     list
    mdds:        list
    win_rates:   list
    base_return: float
    sensitivity: float  # 표준편차 기준 민감도
    is_valid:    bool = False


def run_sensitivity_analysis(
    data_full:   dict,
    base_params: StrategyParams,
    params_to_test: list,  # [{"name": "ma_buy", "values": [30,40,50,60,70]}, ...]
    progress_cb  = None,
) -> list:
    """
    각 파라미터를 변화시키며 수익률/MDD/승률 변화를 분석.
    Returns: SensitivityResult 리스트
    """
    base_res    = run_backtest(data_full, base_params)
    base_return = base_res.total_return_pct if base_res.is_valid else 0.0

    results = []
    total   = sum(len(p["values"]) for p in params_to_test)
    done    = 0

    for param_cfg in params_to_test:
        param_name = param_cfg["name"]
        values     = param_cfg["values"]
        label      = param_cfg.get("label", param_name)

        returns   = []
        mdds      = []
        win_rates = []

        for val in values:
            done += 1
            if progress_cb:
                progress_cb(done, total, f"{label} = {val} 테스트 중...")

            p = copy.deepcopy(base_params)
            setattr(p, param_name, val)

            # 추세 필터 유효성 검사
            if hasattr(p, "ma_trend_short") and hasattr(p, "ma_trend_long"):
                if p.ma_trend_short >= p.ma_trend_long:
                    returns.append(None)
                    mdds.append(None)
                    win_rates.append(None)
                    continue

            res = run_backtest(data_full, p)
            if res.is_valid:
                returns.append(round(res.total_return_pct, 1))
                mdds.append(round(res.mdd_pct, 1))
                win_rates.append(round(res.win_rate_pct, 1))
            else:
                returns.append(None)
                mdds.append(None)
                win_rates.append(None)

        valid_returns = [r for r in returns if r is not None]
        sensitivity   = round(float(np.std(valid_returns)), 1) if len(valid_returns) > 1 else 0.0

        results.append(SensitivityResult(
            param_name  = param_name,
            values      = values,
            returns     = returns,
            mdds        = mdds,
            win_rates   = win_rates,
            base_return = base_return,
            sensitivity = sensitivity,
            is_valid    = bool(valid_returns),
        ))

    if progress_cb:
        progress_cb(total, total, "완료")

    return results


def make_sensitivity_configs(p: StrategyParams) -> list:
    """
    현재 파라미터 기준으로 민감도 테스트 범위 자동 생성.
    각 파라미터의 현재값 ± 범위로 후보 생성.
    활성화된 전략/필터에 해당하는 파라미터만 포함.
    """
    def _ma_range(center, half=20, step=5, lo_min=1, hi_max=400):
        lo = max(lo_min, center - half)
        hi = min(hi_max, center + half)
        return sorted(set(range(lo, hi + 1, step)) | {center})

    def _off_range(center, half=3, step=1, lo_min=0, hi_max=20):
        lo = max(lo_min, center - half)
        hi = min(hi_max, center + half)
        return sorted(set(range(lo, hi + 1, step)) | {center})

    def _int_range(center, half, step, lo_min=1, hi_max=9999):
        lo = max(lo_min, center - half)
        hi = min(hi_max, center + half)
        return sorted(set(range(lo, hi + 1, step)) | {center})

    def _float_range(center, half, step, lo_min=0.0, hi_max=9999.0):
        import numpy as _np
        vals = list(_np.arange(
            max(lo_min, round(center - half, 2)),
            min(hi_max, round(center + half + step * 0.01, 2)),
            step
        ))
        return sorted(set(round(v, 2) for v in vals) | {center})

    configs = []

    # ── 기본 MA / 오프셋 (항상 포함) ──────────────────────────
    if not (p.use_donchian or p.use_swing or p.use_vp or p.use_bollinger):
        configs += [
            {"name": "ma_buy",  "label": "매수 MA 기간",  "values": _ma_range(p.ma_buy)},
            {"name": "ma_sell", "label": "매도 MA 기간",  "values": _ma_range(p.ma_sell)},
        ]

    configs += [
        {"name": "offset_cl_buy",  "label": "종가 오프셋 (매수)", "values": _off_range(p.offset_cl_buy)},
        {"name": "offset_cl_sell", "label": "종가 오프셋 (매도)", "values": _off_range(p.offset_cl_sell)},
    ]

    # ── 손절 / 익절 ────────────────────────────────────────────
    if p.stop_loss_pct > 0 and not p.use_atr_stop:
        configs.append({"name": "stop_loss_pct", "label": "손절(%)",
                         "values": _float_range(p.stop_loss_pct, half=10, step=2, lo_min=2)})

    if p.use_atr_stop:
        configs.append({"name": "atr_multiplier", "label": "ATR 손절 배수",
                         "values": _float_range(p.atr_multiplier, half=1.5, step=0.5, lo_min=0.5, hi_max=6.0)})

    if p.take_profit_pct > 0:
        configs.append({"name": "take_profit_pct", "label": "익절(%)",
                         "values": _float_range(p.take_profit_pct, half=15, step=5, lo_min=5)})

    # ── 추세 필터 ──────────────────────────────────────────────
    if p.use_trend_buy or p.use_trend_sell:
        configs += [
            {"name": "ma_trend_short",        "label": "추세 단기 MA",          "values": _ma_range(p.ma_trend_short, half=15, step=5)},
            {"name": "ma_trend_long",         "label": "추세 장기 MA",          "values": _ma_range(p.ma_trend_long,  half=30, step=10, hi_max=400)},
            {"name": "offset_trend_short",    "label": "추세 단기 오프셋",      "values": _off_range(p.offset_trend_short)},
            {"name": "offset_trend_long",     "label": "추세 장기 오프셋",      "values": _off_range(p.offset_trend_long)},
        ]

    # ── RSI 필터 ───────────────────────────────────────────────
    if p.use_rsi_filter:
        configs += [
            {"name": "rsi_period", "label": "RSI 기간",    "values": _int_range(p.rsi_period, half=7, step=2, lo_min=2, hi_max=50)},
            {"name": "rsi_min",    "label": "RSI 하한",    "values": _float_range(p.rsi_min, half=15, step=5, lo_min=10, hi_max=50)},
            {"name": "rsi_max",    "label": "RSI 상한",    "values": _float_range(p.rsi_max, half=15, step=5, lo_min=50, hi_max=90)},
        ]

    # ── MACD ──────────────────────────────────────────────────
    if p.use_macd:
        configs += [
            {"name": "macd_fast",          "label": "MACD 단기",   "values": _int_range(p.macd_fast,          half=4, step=1, lo_min=2,  hi_max=30)},
            {"name": "macd_slow",          "label": "MACD 장기",   "values": _int_range(p.macd_slow,          half=8, step=2, lo_min=10, hi_max=60)},
            {"name": "macd_signal_period", "label": "MACD 시그널", "values": _int_range(p.macd_signal_period, half=4, step=1, lo_min=2,  hi_max=20)},
        ]

    # ── 볼린저 밴드 ────────────────────────────────────────────
    if p.use_bollinger:
        configs += [
            {"name": "bb_period", "label": "BB 기간",   "values": _int_range(p.bb_period, half=10, step=5, lo_min=5,   hi_max=60)},
            {"name": "bb_std",    "label": "BB 표준편차","values": _float_range(p.bb_std,  half=0.75, step=0.25, lo_min=1.0, hi_max=3.5)},
        ]

    # ── 도니챈 채널 ────────────────────────────────────────────
    if p.use_donchian:
        configs += [
            {"name": "dc_period_entry", "label": "도니챈 진입 기간", "values": _int_range(p.dc_period_entry, half=10, step=5, lo_min=5,  hi_max=60)},
            {"name": "dc_period_exit",  "label": "도니챈 청산 기간", "values": _int_range(p.dc_period_exit,  half=5,  step=2, lo_min=2,  hi_max=30)},
        ]

    # ── 스윙 하이/로우 ─────────────────────────────────────────
    if p.use_swing:
        configs.append({"name": "swing_n", "label": "스윙 N봉",
                         "values": _int_range(p.swing_n, half=4, step=1, lo_min=1, hi_max=15)})

    # ── 볼륨 프로파일 ──────────────────────────────────────────
    if p.use_vp:
        configs.append({"name": "vp_period", "label": "VP 롤링 기간",
                         "values": _int_range(p.vp_period, half=30, step=10, lo_min=20, hi_max=200)})

    # ── 시장 필터 ──────────────────────────────────────────────
    if p.use_market_filter:
        configs.append({"name": "market_ma_period", "label": "시장 필터 MA",
                         "values": _ma_range(p.market_ma_period, half=50, step=20, hi_max=500)})

    return configs
