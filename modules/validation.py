"""
validation.py - 전략 검증 모듈
1. Walk-Forward Analysis
2. 다종목 교차 검증
"""
import copy, random, numpy as np, pandas as pd, optuna, datetime
from dataclasses import dataclass, field
from .engine import StrategyParams, prepare_data, run_backtest
from .optimizer import (
    OptimizeConstraints, _MA_REDUCED, _OFF_REDUCED, _MA_FULL, _OFF_FULL,
    _build_params_from_trial, _params_from_trial_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 1. Walk-Forward Analysis
# ══════════════════════════════════════════════════════════

@dataclass
class WalkForwardResult:
    windows:         list = field(default_factory=list)  # 각 윈도우 결과
    oos_returns:     list = field(default_factory=list)  # Out-of-Sample 수익률 모음
    avg_oos_return:  float = 0.0
    std_oos_return:  float = 0.0
    win_rate:        float = 0.0  # OOS 구간 중 수익 구간 비율
    total_oos_return:float = 0.0  # 전체 OOS 수익 (복리 합산)
    is_valid:        bool  = False


def run_walk_forward(
    signal_ticker: str,
    trade_ticker:  str,
    market_ticker: str,
    start_date,
    end_date,
    base_params:   StrategyParams,
    ss_config:     dict,
    constraints:   OptimizeConstraints,
    is_months:     int = 24,   # In-Sample 기간 (개월)
    oos_months:    int = 6,    # Out-of-Sample 기간 (개월)
    step_months:   int = 6,    # 스텝 (개월)
    n_trials:      int = 100,  # 최적화 횟수
    n_seeds:       int = 2,    # 시드 수
    progress_cb    = None,
) -> WalkForwardResult:

    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    # 윈도우 생성
    windows = []
    cur = start_dt
    while True:
        is_start = cur
        is_end   = cur + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end   = is_end + pd.DateOffset(months=oos_months)
        if oos_end > end_dt:
            break
        windows.append({
            "is_start":  is_start.date(),
            "is_end":    is_end.date(),
            "oos_start": oos_start.date(),
            "oos_end":   oos_end.date(),
        })
        cur = cur + pd.DateOffset(months=step_months)

    if not windows:
        return WalkForwardResult(is_valid=False)

    total_steps = len(windows)
    results = []

    for wi, w in enumerate(windows):
        if progress_cb:
            progress_cb(wi, total_steps, f"윈도우 {wi+1}/{total_steps}: {w['is_start']} ~ {w['oos_end']}")

        # ── In-Sample 최적화 ──────────────────────────────
        data_is = prepare_data(
            signal_ticker, trade_ticker, market_ticker,
            w["is_start"], w["is_end"], base_params
        )
        if data_is is None or len(data_is["base"]) < 60:
            results.append({**w, "best_params": None, "is_return": None, "oos_return": None, "status": "데이터 부족"})
            continue

        # 단순 단일시드 최적화 (속도 우선)
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
            results.append({**w, "best_params": None, "is_return": None, "oos_return": None, "status": "최적화 실패"})
            continue

        # IS 성과 확인
        is_res = run_backtest(data_is, best_params)

        # ── Out-of-Sample 검증 ────────────────────────────
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
            "best_params":  best_params,
            "is_return":    round(is_res.total_return_pct, 1),
            "is_mdd":       round(is_res.mdd_pct, 1),
            "is_trades":    is_res.total_trades,
            "oos_return":   round(oos_res.total_return_pct, 1),
            "oos_mdd":      round(oos_res.mdd_pct, 1),
            "oos_trades":   oos_res.total_trades,
            "oos_winrate":  round(oos_res.win_rate_pct, 1),
            "status":       "완료",
            # 최적 파라미터 요약
            "ma_buy":       getattr(best_params, "ma_buy", "-"),
            "ma_sell":      getattr(best_params, "ma_sell", "-"),
            "buy_op":       getattr(best_params, "buy_operator", "-"),
            "sell_op":      getattr(best_params, "sell_operator", "-"),
        })

    if progress_cb:
        progress_cb(total_steps, total_steps, "완료")

    # 결과 집계
    oos_returns = [r["oos_return"] for r in results if r.get("oos_return") is not None]
    if not oos_returns:
        return WalkForwardResult(windows=results, is_valid=False)

    # 복리 합산 수익률
    total_oos = 1.0
    for r in oos_returns:
        total_oos *= (1 + r / 100)
    total_oos = (total_oos - 1) * 100

    return WalkForwardResult(
        windows          = results,
        oos_returns      = oos_returns,
        avg_oos_return   = round(float(np.mean(oos_returns)), 1),
        std_oos_return   = round(float(np.std(oos_returns)),  1),
        win_rate         = round(sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100, 1),
        total_oos_return = round(total_oos, 1),
        is_valid         = True,
    )


# ══════════════════════════════════════════════════════════
# 2. 다종목 교차 검증
# ══════════════════════════════════════════════════════════

@dataclass
class CrossValidationResult:
    rows:     list = field(default_factory=list)
    is_valid: bool = False


def run_cross_validation(
    tickers:     list,
    start_date,
    end_date,
    base_params: StrategyParams,
    progress_cb  = None,
) -> CrossValidationResult:
    """
    현재 전략 파라미터를 여러 종목에 그대로 적용해서 성과 비교.
    signal_ticker = trade_ticker = 각 티커
    """
    rows = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, total, f"{ticker} 분석 중...")

        try:
            p = copy.deepcopy(base_params)
            p.signal_ticker = ticker
            p.trade_ticker  = ticker

            data = prepare_data(
                ticker, ticker, p.market_ticker,
                start_date, end_date, p
            )
            if data is None or len(data["base"]) < 60:
                rows.append({"티커": ticker, "상태": "데이터 부족"})
                continue

            res = run_backtest(data, p)
            if not res.is_valid:
                rows.append({"티커": ticker, "상태": "백테스트 실패"})
                continue

            rows.append({
                "티커":         ticker,
                "수익률(%)":    res.total_return_pct,
                "B&H(%)":       res.bh_return_pct,
                "MDD(%)":       res.mdd_pct,
                "승률(%)":      res.win_rate_pct,
                "PF":           res.profit_factor,
                "매매횟수":     res.total_trades,
                "샤프":         round(float(np.std(np.diff(res.asset_curve) / res.asset_curve[:-1]) * np.sqrt(252)) if len(res.asset_curve) > 1 else 0, 2),
                "상태":         "완료",
            })

        except Exception as e:
            rows.append({"티커": ticker, "상태": f"오류: {str(e)[:30]}"})

    if progress_cb:
        progress_cb(total, total, "완료")

    return CrossValidationResult(rows=rows, is_valid=bool(rows))
