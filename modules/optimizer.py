"""
optimizer.py - Optuna 베이지안 최적화
기존 optuna_objective 방식 기반으로 재작성:
- suggest_float(step=N) 사용 → 연속값 베이지안 최적화 효율 극대화
- bool은 categorical로 직접 사용
- 볼린저 제거
- Train/Test 없음 (전체 기간)
"""
from __future__ import annotations
import copy, numpy as np, pandas as pd, optuna, streamlit as st
from dataclasses import dataclass, field
from .engine import StrategyParams, prepare_data, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MA_LIST     = [1] + list(range(5, 121, 5))   # 1,5,10,...,120
_OFFSET_LIST = [1] + list(range(5, 61, 5))    # 1,5,10,...,60


@dataclass
class OptimizeConstraints:
    min_trades:   int   = 5
    min_win_rate: float = 0.0
    max_mdd:      float = 0.0  # 절대값 기준, 0 = 제한 없음


def _build_params_from_trial(trial, base_params, ss_config, disable_tp=False):
    """
    ss_config: 각 필터의 탐색 모드를 담은 dict
    {
      "trend_buy":  "off" | "fixed" | "search",
      "trend_sell": "off" | "fixed" | "search",
      "rsi":        "off" | "fixed" | "search",
      "macd":       "off" | "fixed" | "search",
      "atr":        "off" | "fixed" | "search",
    }
    """
    p = copy.deepcopy(base_params)

    # ── 이평선 / 오프셋 (categorical) ────────────────────
    p.ma_buy         = trial.suggest_categorical("ma_buy",      _MA_LIST)
    p.offset_ma_buy  = trial.suggest_categorical("off_ma_buy",  _OFFSET_LIST)
    p.offset_cl_buy  = trial.suggest_categorical("off_cl_buy",  _OFFSET_LIST)
    p.buy_operator   = trial.suggest_categorical("buy_op",      [">", "<"])

    p.ma_sell        = trial.suggest_categorical("ma_sell",     _MA_LIST)
    p.offset_ma_sell = trial.suggest_categorical("off_ma_sell", _OFFSET_LIST)
    p.offset_cl_sell = trial.suggest_categorical("off_cl_sell", _OFFSET_LIST)
    p.sell_operator  = trial.suggest_categorical("sell_op",     ["<", ">", "OFF"])

    # ── 추세 필터 ─────────────────────────────────────────
    trend_buy_mode  = ss_config.get("trend_buy",  "search")
    trend_sell_mode = ss_config.get("trend_sell", "search")

    if trend_buy_mode == "search":
        p.use_trend_buy = trial.suggest_categorical("use_trend_buy", [True, False])
    elif trend_buy_mode == "fixed":
        p.use_trend_buy = True
    else:
        p.use_trend_buy = False

    if trend_sell_mode == "search":
        p.use_trend_sell = trial.suggest_categorical("use_trend_sell", [True, False])
    elif trend_sell_mode == "fixed":
        p.use_trend_sell = True
    else:
        p.use_trend_sell = False

    if p.use_trend_buy or p.use_trend_sell:
        p.ma_trend_short     = trial.suggest_categorical("ma_ts",   _MA_LIST)
        p.ma_trend_long      = trial.suggest_categorical("ma_tl",   _MA_LIST)
        p.offset_trend_short = trial.suggest_categorical("off_ts",  _OFFSET_LIST)
        p.offset_trend_long  = trial.suggest_categorical("off_tl",  _OFFSET_LIST)
        if p.ma_trend_short >= p.ma_trend_long:
            raise optuna.TrialPruned()

    # ── 손절 / 익절 (float, step 활용) ───────────────────
    atr_mode = ss_config.get("atr", "search")
    if atr_mode == "search":
        p.use_atr_stop = trial.suggest_categorical("use_atr", [True, False])
    elif atr_mode == "fixed":
        p.use_atr_stop = True
    else:
        p.use_atr_stop = False

    if p.use_atr_stop:
        p.atr_multiplier = trial.suggest_float("atr_mult", 2.0, 4.0, step=0.5)
        p.stop_loss_pct  = 0.0
    else:
        p.stop_loss_pct  = trial.suggest_float("sl", 15.0, 35.0, step=5.0)

    if disable_tp:
        p.take_profit_pct = 0.0
    else:
        p.take_profit_pct = trial.suggest_float("tp", 0.0, 50.0, step=5.0)

    # ── RSI 필터 ─────────────────────────────────────────
    rsi_mode = ss_config.get("rsi", "off")
    if rsi_mode == "search":
        p.use_rsi_filter = trial.suggest_categorical("use_rsi", [True, False])
    elif rsi_mode == "fixed":
        p.use_rsi_filter = True
    else:
        p.use_rsi_filter = False

    if p.use_rsi_filter and rsi_mode == "search":
        p.rsi_period = trial.suggest_categorical("rsi_period", [7, 10, 14, 21])
        rsi_max      = trial.suggest_categorical("rsi_max",    [60, 65, 70, 75, 80])
        p.rsi_max    = rsi_max
        p.rsi_min    = 100 - rsi_max

    # ── MACD 필터 ────────────────────────────────────────
    macd_mode = ss_config.get("macd", "off")
    if macd_mode == "search":
        p.use_macd = trial.suggest_categorical("use_macd", [True, False])
    elif macd_mode == "fixed":
        p.use_macd = True
    else:
        p.use_macd = False

    return p


def _params_from_trial_params(tp, base_params):
    """완료된 trial의 params dict → StrategyParams"""
    p = copy.deepcopy(base_params)
    p.ma_buy             = tp.get("ma_buy", 50)
    p.offset_ma_buy      = tp.get("off_ma_buy", 1)
    p.offset_cl_buy      = tp.get("off_cl_buy", 1)
    p.buy_operator       = tp.get("buy_op", ">")
    p.ma_sell            = tp.get("ma_sell", 10)
    p.offset_ma_sell     = tp.get("off_ma_sell", 1)
    p.offset_cl_sell     = tp.get("off_cl_sell", 1)
    p.sell_operator      = tp.get("sell_op", "<")
    p.use_trend_buy      = tp.get("use_trend_buy", False)
    p.use_trend_sell     = tp.get("use_trend_sell", False)
    p.ma_trend_short     = tp.get("ma_ts", 20)
    p.ma_trend_long      = tp.get("ma_tl", 50)
    p.offset_trend_short = tp.get("off_ts", 1)
    p.offset_trend_long  = tp.get("off_tl", 1)
    p.use_atr_stop       = tp.get("use_atr", False)
    p.atr_multiplier     = tp.get("atr_mult", 2.0)
    p.stop_loss_pct      = tp.get("sl", 0.0)
    p.take_profit_pct    = tp.get("tp", 0.0)
    p.use_rsi_filter     = tp.get("use_rsi", False)
    p.rsi_period         = tp.get("rsi_period", 14)
    p.rsi_max            = tp.get("rsi_max", 70)
    p.rsi_min            = 100 - p.rsi_max
    p.use_macd           = tp.get("use_macd", False)
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0
    return p


def run_optimization(
    signal_ticker, trade_ticker, start_date, end_date,
    base_params, ss_config, constraints,
    n_trials=100, target="수익률 (%)", disable_tp=False,
    seed=None, progress_cb=None,
):
    """
    ss_config: 필터별 모드 dict
    {
      "trend_buy":  "off"|"fixed"|"search",
      "trend_sell": "off"|"fixed"|"search",
      "rsi":        "off"|"fixed"|"search",
      "macd":       "off"|"fixed"|"search",
      "atr":        "off"|"fixed"|"search",
    }
    """
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    data_full = prepare_data(
        signal_ticker, trade_ticker, base_params.market_ticker,
        start_date, end_date, base_params
    )
    if data_full is None:
        return pd.DataFrame(), None

    import random as _random
    actual_seed = seed if seed is not None else _random.randint(0, 99999)
    sampler = optuna.samplers.TPESampler(seed=actual_seed)
    if is_multi:
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        try:
            p   = _build_params_from_trial(trial, base_params, ss_config, disable_tp)
            res = run_backtest(data_full, p)

            if res is None or not res.is_valid:
                raise optuna.TrialPruned()
            if res.total_trades < constraints.min_trades:
                raise optuna.TrialPruned()
            if (res.win_rate_pct or 0) < constraints.min_win_rate:
                raise optuna.TrialPruned()
            if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd:
                raise optuna.TrialPruned()

            ret = float(res.total_return_pct or -999.0)
            mdd = float(abs(res.mdd_pct or 0))
            pf  = float(min(res.profit_factor or 0, 999.0))
            wr  = float(res.win_rate_pct or 0)

            if is_multi:
                return ret, mdd
            elif target == "Profit Factor":
                return pf
            elif target == "승률 (%)":
                return wr
            elif target == "MDD 최소화":
                return -mdd
            else:
                return ret

        except optuna.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()

    def _cb(study, trial):
        if progress_cb: progress_cb(trial.number + 1, n_trials)

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

    trials = study.best_trials if is_multi else [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    rows = []
    for t in trials:
        tp  = t.params
        p   = _params_from_trial_params(tp, base_params)
        res = run_backtest(data_full, p)
        if not res.is_valid: continue
        if res.total_trades < constraints.min_trades: continue
        if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd: continue
        rows.append({
            "수익률(%)":           res.total_return_pct,
            "MDD(%)":              res.mdd_pct,
            "승률(%)":             res.win_rate_pct,
            "PF":                  res.profit_factor,
            "매매횟수":            res.total_trades,
            "ma_buy":              tp.get("ma_buy"),
            "offset_cl_buy":       tp.get("off_cl_buy"),
            "offset_ma_buy":       tp.get("off_ma_buy"),
            "buy_operator":        tp.get("buy_op"),
            "ma_sell":             tp.get("ma_sell"),
            "offset_cl_sell":      tp.get("off_cl_sell"),
            "offset_ma_sell":      tp.get("off_ma_sell"),
            "sell_operator":       tp.get("sell_op"),
            "use_trend_buy":       tp.get("use_trend_buy", False),
            "use_trend_sell":      tp.get("use_trend_sell", False),
            "ma_trend_short":      tp.get("ma_ts"),
            "ma_trend_long":       tp.get("ma_tl"),
            "offset_trend_short":  tp.get("off_ts"),
            "offset_trend_long":   tp.get("off_tl"),
            "use_atr_stop":        tp.get("use_atr", False),
            "atr_multiplier":      tp.get("atr_mult"),
            "stop_loss_pct":       tp.get("sl"),
            "take_profit_pct":     tp.get("tp", 0.0),
            "use_rsi":             tp.get("use_rsi", False),
            "rsi_period":          tp.get("rsi_period"),
            "rsi_max":             tp.get("rsi_max"),
            "use_macd":            tp.get("use_macd", False),
        })

    if not rows:
        return pd.DataFrame(), study

    return pd.DataFrame(rows).sort_values("수익률(%)", ascending=False), study


def apply_optimal_params(row):
    p = StrategyParams()
    p.ma_buy             = int(row.get("ma_buy", 50))
    p.offset_cl_buy      = int(row.get("offset_cl_buy", 1))
    p.offset_ma_buy      = int(row.get("offset_ma_buy", 1))
    p.buy_operator       = str(row.get("buy_operator", ">"))
    p.ma_sell            = int(row.get("ma_sell", 10))
    p.offset_cl_sell     = int(row.get("offset_cl_sell", 1))
    p.offset_ma_sell     = int(row.get("offset_ma_sell", 1))
    p.sell_operator      = str(row.get("sell_operator", "<"))
    p.use_trend_buy      = bool(row.get("use_trend_buy", False))
    p.use_trend_sell     = bool(row.get("use_trend_sell", False))
    p.ma_trend_short     = int(row.get("ma_trend_short", 20))
    p.ma_trend_long      = int(row.get("ma_trend_long", 50))
    p.offset_trend_short = int(row.get("offset_trend_short", 1))
    p.offset_trend_long  = int(row.get("offset_trend_long", 1))
    p.use_atr_stop       = bool(row.get("use_atr_stop", False))
    p.atr_multiplier     = float(row.get("atr_multiplier", 2.0))
    p.stop_loss_pct      = float(row.get("stop_loss_pct", 0.0))
    p.take_profit_pct    = float(row.get("take_profit_pct", 0.0))
    return p
