"""
validation.py - 전략 검증 모듈
1. Walk-Forward Analysis
2. 파라미터 민감도 분석
"""
import copy, random, numpy as np, pandas as pd, optuna, datetime
from dataclasses import dataclass, field
from .engine import StrategyParams, prepare_data, run_backtest
from .optimizer import (
    OptimizeConstraints, _MA_REDUCED, _OFF_REDUCED,
    _build_params_from_trial, _params_from_trial_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 1. Walk-Forward Analysis
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
    ss_config:     dict,
    constraints:   OptimizeConstraints,
    is_months:     int = 24,
    oos_months:    int = 6,
    step_months:   int = 6,
    n_trials:      int = 100,
    n_seeds:       int = 2,
    progress_cb    = None,
) -> WalkForwardResult:

    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    windows = []
    cur = start_dt
    while True:
        is_end   = cur + pd.DateOffset(months=is_months)
        oos_end  = is_end + pd.DateOffset(months=oos_months)
        if oos_end > end_dt:
            break
        windows.append({
            "is_start":  cur.date(),
            "is_end":    is_end.date(),
            "oos_start": is_end.date(),
            "oos_end":   oos_end.date(),
        })
        cur += pd.DateOffset(months=step_months)

    if not windows:
        return WalkForwardResult(is_valid=False)

    total_steps = len(windows)
    results = []

    for wi, w in enumerate(windows):
        if progress_cb:
            progress_cb(wi, total_steps, f"윈도우 {wi+1}/{total_steps}: {w['is_start']} ~ {w['oos_end']}")

        data_is = prepare_data(
            signal_ticker, trade_ticker, market_ticker,
            w["is_start"], w["is_end"], base_params
        )
        if data_is is None or len(data_is["base"]) < 60:
            results.append({**w, "best_params": None, "is_return": None,
                            "oos_return": None, "status": "데이터 부족"})
            continue

        best_params = None
        best_score  = -999.0
        seeds = [random.randint(0, 99999) for _ in range(n_seeds)]

        for seed in seeds:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study   = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial, _data=data_is, _base=base_params):
                try:
                    p = _build_params_from_trial(
                        trial, _base, ss_config, False,
                        _MA_REDUCED, _OFF_REDUCED
                    )
                    res = run_backtest(_data, p)
                    if not res.is_valid or res.total_trades < constraints.min_trades:
                        raise optuna.TrialPruned()
                    if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd:
                        raise optuna.TrialPruned()
                    return float(res.total_return_pct or -999.0)
                except optuna.TrialPruned: raise
                except Exception: raise optuna.TrialPruned()

            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                best_t = max(completed, key=lambda t: t.value or -999.0)
                if (best_t.value or -999.0) > best_score:
                    best_score  = best_t.value or -999.0
                    best_params = _params_from_trial_params(best_t.params, base_params, ss_config)

        if best_params is None:
            results.append({**w, "best_params": None, "is_return": None,
                            "oos_return": None, "status": "최적화 실패"})
            continue

        is_res   = run_backtest(data_is, best_params)
        data_oos = prepare_data(
            signal_ticker, trade_ticker, market_ticker,
            w["oos_start"], w["oos_end"], best_params
        )
        if data_oos is None or len(data_oos["base"]) < 10:
            results.append({**w, "best_params": best_params,
                            "is_return": is_res.total_return_pct,
                            "oos_return": None, "status": "OOS 데이터 부족"})
            continue

        oos_res = run_backtest(data_oos, best_params)
        results.append({
            **w,
            "best_params": best_params,
            "is_return":   round(is_res.total_return_pct, 1),
            "is_mdd":      round(is_res.mdd_pct, 1),
            "is_trades":   is_res.total_trades,
            "oos_return":  round(oos_res.total_return_pct, 1),
            "oos_mdd":     round(oos_res.mdd_pct, 1),
            "oos_trades":  oos_res.total_trades,
            "oos_winrate": round(oos_res.win_rate_pct, 1),
            "status":      "완료",
            "ma_buy":      getattr(best_params, "ma_buy", "-"),
            "ma_sell":     getattr(best_params, "ma_sell", "-"),
            "buy_op":      getattr(best_params, "buy_operator", "-"),
            "sell_op":     getattr(best_params, "sell_operator", "-"),
        })

    if progress_cb:
        progress_cb(total_steps, total_steps, "완료")

    oos_returns = [r["oos_return"] for r in results if r.get("oos_return") is not None]
    if not oos_returns:
        return WalkForwardResult(windows=results, is_valid=False)

    total_oos = 1.0
    for r in oos_returns:
        total_oos *= (1 + r / 100)
    total_oos = (total_oos - 1) * 100

    return WalkForwardResult(
        windows          = results,
        oos_returns      = oos_returns,
        avg_oos_return   = round(float(np.mean(oos_returns)), 1),
        std_oos_return   = round(float(np.std(oos_returns)), 1),
        win_rate         = round(sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100, 1),
        total_oos_return = round(total_oos, 1),
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
    """
    def _ma_range(center, half=20, step=5):
        lo = max(1, center - half)
        hi = min(120, center + half)
        return sorted(set(range(lo, hi + 1, step)) | {center})

    def _off_range(center, half=10, step=5):
        lo = max(1, center - half)
        hi = min(60, center + half)
        return sorted(set(range(lo, hi + 1, step)) | {center})

    configs = [
        {"name": "ma_buy",  "label": "매수 MA",     "values": _ma_range(p.ma_buy)},
        {"name": "ma_sell", "label": "매도 MA",     "values": _ma_range(p.ma_sell)},
        {"name": "offset_cl_buy",  "label": "종가 오프셋 (매수)", "values": _off_range(p.offset_cl_buy)},
        {"name": "offset_cl_sell", "label": "종가 오프셋 (매도)", "values": _off_range(p.offset_cl_sell)},
    ]

    if p.stop_loss_pct > 0 and not p.use_atr_stop:
        sl_vals = sorted(set(range(
            max(5, int(p.stop_loss_pct) - 10),
            min(50, int(p.stop_loss_pct) + 15), 5
        )) | {int(p.stop_loss_pct)})
        configs.append({"name": "stop_loss_pct", "label": "손절(%)", "values": sl_vals})

    if p.take_profit_pct > 0:
        tp_vals = sorted(set(range(
            max(0, int(p.take_profit_pct) - 15),
            min(100, int(p.take_profit_pct) + 20), 5
        )) | {int(p.take_profit_pct)})
        configs.append({"name": "take_profit_pct", "label": "익절(%)", "values": tp_vals})

    if p.use_trend_buy or p.use_trend_sell:
        configs.append({"name": "ma_trend_short", "label": "추세 단기 MA", "values": _ma_range(p.ma_trend_short)})
        configs.append({"name": "ma_trend_long",  "label": "추세 장기 MA", "values": _ma_range(p.ma_trend_long)})

    return configs
