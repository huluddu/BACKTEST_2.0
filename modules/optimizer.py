"""
optimizer.py - 기존 AI 지능형 풀옵션 + 베이지안 통합 업그레이드
"""
from __future__ import annotations
import copy, numpy as np, pandas as pd, optuna, streamlit as st
from dataclasses import dataclass, field
from .engine import StrategyParams, prepare_data, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MA_FULL     = [1] + list(range(5, 121, 5))
_OFFSET_FULL = [1] + list(range(5, 61,  5))
_MA_SIMPLE   = [5, 10, 20, 50, 60, 120, 200]
_OFF_SIMPLE  = [1, 5, 10, 20]

@dataclass
class SearchSpace:
    ma_buy_choices:         list = field(default_factory=lambda: _MA_SIMPLE)
    ma_sell_choices:        list = field(default_factory=lambda: _MA_SIMPLE)
    offset_cl_buy_choices:  list = field(default_factory=lambda: _OFF_SIMPLE)
    offset_ma_buy_choices:  list = field(default_factory=lambda: _OFF_SIMPLE)
    offset_cl_sell_choices: list = field(default_factory=lambda: _OFF_SIMPLE)
    offset_ma_sell_choices: list = field(default_factory=lambda: _OFF_SIMPLE)
    buy_operator_choices:   list = field(default_factory=lambda: [">", "<"])
    sell_operator_choices:  list = field(default_factory=lambda: ["<", ">", "OFF"])
    use_trend_buy_choices:  list = field(default_factory=lambda: [True, False])
    use_trend_sell_choices: list = field(default_factory=lambda: [True, False])
    ma_trend_short_choices: list = field(default_factory=lambda: [5, 10, 20, 50])
    ma_trend_long_choices:  list = field(default_factory=lambda: [20, 50, 60, 120, 200])
    stop_loss_choices:      list = field(default_factory=lambda: [0.0, 10.0, 15.0, 20.0, 25.0, 35.0])
    take_profit_choices:    list = field(default_factory=lambda: [0.0, 15.0, 25.0, 35.0, 50.0])
    use_atr_stop_choices:   list = field(default_factory=lambda: [True, False])
    atr_mult_choices:       list = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 4.0])
    # RSI 필터 탐색
    use_rsi_choices:        list = field(default_factory=lambda: [False])
    rsi_period_choices:     list = field(default_factory=lambda: [14])
    rsi_max_choices:        list = field(default_factory=lambda: [70])
    # 볼린저 탐색
    use_bb_choices:         list = field(default_factory=lambda: [False])
    bb_period_choices:      list = field(default_factory=lambda: [20])
    bb_std_choices:         list = field(default_factory=lambda: [2.0])
    # MACD 탐색
    use_macd_choices:       list = field(default_factory=lambda: [False])

def make_full_search_space(ma_choices=None, use_trend=True, use_atr=True,
                           use_rsi=False, use_bb=False, use_macd=False) -> SearchSpace:
    ma = ma_choices or _MA_FULL
    return SearchSpace(
        ma_buy_choices=ma, ma_sell_choices=ma,
        offset_cl_buy_choices=_OFFSET_FULL, offset_ma_buy_choices=_OFFSET_FULL,
        offset_cl_sell_choices=_OFFSET_FULL, offset_ma_sell_choices=_OFFSET_FULL,
        buy_operator_choices=[">", "<"], sell_operator_choices=["<", ">", "OFF"],
        use_trend_buy_choices=[True, False] if use_trend else [False],
        use_trend_sell_choices=[True, False] if use_trend else [False],
        ma_trend_short_choices=ma, ma_trend_long_choices=ma,
        stop_loss_choices=[15.0, 20.0, 25.0, 30.0, 35.0],
        take_profit_choices=[0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices=[True, False] if use_atr else [False],
        atr_mult_choices=[2.0, 2.5, 3.0, 4.0],
        # RSI: 사용 여부 + 기간 + 과매수 기준 탐색
        use_rsi_choices=[True, False] if use_rsi else [False],
        rsi_period_choices=[7, 10, 14, 21] if use_rsi else [14],
        rsi_max_choices=[60, 65, 70, 75, 80] if use_rsi else [70],
        # 볼린저: 사용 여부 + 기간 + 배수 탐색
        use_bb_choices=[True, False] if use_bb else [False],
        bb_period_choices=[10, 15, 20, 30] if use_bb else [20],
        bb_std_choices=[1.5, 2.0, 2.5] if use_bb else [2.0],
        # MACD: 사용 여부만 탐색
        use_macd_choices=[True, False] if use_macd else [False],
    )

def make_simple_search_space(base_params: StrategyParams) -> SearchSpace:
    return SearchSpace(
        ma_buy_choices=_MA_FULL, ma_sell_choices=_MA_FULL,   # [1]+range(5,121,5)
        offset_cl_buy_choices=_OFFSET_FULL, offset_ma_buy_choices=_OFFSET_FULL,
        offset_cl_sell_choices=_OFFSET_FULL, offset_ma_sell_choices=_OFFSET_FULL,
        buy_operator_choices=[base_params.buy_operator],
        sell_operator_choices=[base_params.sell_operator],
        use_trend_buy_choices=[base_params.use_trend_buy],
        use_trend_sell_choices=[base_params.use_trend_sell],
        ma_trend_short_choices=_MA_FULL,
        ma_trend_long_choices=_MA_FULL,
        stop_loss_choices=[0.0, 10.0, 15.0, 20.0, 25.0, 35.0],
        take_profit_choices=[0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices=[base_params.use_atr_stop],
        atr_mult_choices=[1.5, 2.0, 2.5, 3.0, 4.0],
    )

@dataclass
class OptimizeConstraints:
    min_trades:   int   = 5
    min_win_rate: float = 0.0
    max_mdd:      float = 0.0  # 절대값 기준, 0 = 제한 없음

def _build_params_from_trial(trial, ss, base_params, disable_tp=False):
    p = copy.deepcopy(base_params)
    p.ma_buy         = trial.suggest_categorical("ma_buy",      ss.ma_buy_choices)
    p.ma_sell        = trial.suggest_categorical("ma_sell",     ss.ma_sell_choices)
    p.offset_cl_buy  = trial.suggest_categorical("off_cl_buy",  ss.offset_cl_buy_choices)
    p.offset_ma_buy  = trial.suggest_categorical("off_ma_buy",  ss.offset_ma_buy_choices)
    p.offset_cl_sell = trial.suggest_categorical("off_cl_sell", ss.offset_cl_sell_choices)
    p.offset_ma_sell = trial.suggest_categorical("off_ma_sell", ss.offset_ma_sell_choices)
    p.buy_operator   = trial.suggest_categorical("buy_op",      ss.buy_operator_choices)
    p.sell_operator  = trial.suggest_categorical("sell_op",     ss.sell_operator_choices)
    p.use_trend_buy  = trial.suggest_categorical("use_trend_buy",  ss.use_trend_buy_choices)
    p.use_trend_sell = trial.suggest_categorical("use_trend_sell", ss.use_trend_sell_choices)
    p.ma_trend_short = trial.suggest_categorical("ma_ts",       ss.ma_trend_short_choices)
    p.ma_trend_long  = trial.suggest_categorical("ma_tl",       ss.ma_trend_long_choices)
    p.stop_loss_pct  = trial.suggest_categorical("sl",          ss.stop_loss_choices)
    p.use_atr_stop   = trial.suggest_categorical("use_atr",     ss.use_atr_stop_choices)
    p.atr_multiplier = trial.suggest_categorical("atr_mult",    ss.atr_mult_choices)
    p.take_profit_pct = 0.0 if disable_tp else trial.suggest_categorical("tp", ss.take_profit_choices)

    # RSI 필터 탐색
    p.use_rsi_filter = trial.suggest_categorical("use_rsi", ss.use_rsi_choices)
    if p.use_rsi_filter:
        p.rsi_period = trial.suggest_categorical("rsi_period", ss.rsi_period_choices)
        p.rsi_max    = trial.suggest_categorical("rsi_max",    ss.rsi_max_choices)
        p.rsi_min    = 100 - p.rsi_max  # 과매도 = 100 - 과매수 (대칭)

    # 볼린저 탐색
    p.use_bollinger = trial.suggest_categorical("use_bb", ss.use_bb_choices)
    if p.use_bollinger:
        p.bb_period = trial.suggest_categorical("bb_period", ss.bb_period_choices)
        p.bb_std    = trial.suggest_categorical("bb_std",    ss.bb_std_choices)

    # MACD 탐색
    p.use_macd = trial.suggest_categorical("use_macd", ss.use_macd_choices)

    if (p.use_trend_buy or p.use_trend_sell) and p.ma_trend_short >= p.ma_trend_long:
        raise optuna.TrialPruned()
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0
    return p

def _params_from_dict(tp, base_params):
    p = copy.deepcopy(base_params)
    p.ma_buy         = int(tp.get("ma_buy", 50))
    p.ma_sell        = int(tp.get("ma_sell", 10))
    p.offset_cl_buy  = int(tp.get("off_cl_buy", 1))
    p.offset_ma_buy  = int(tp.get("off_ma_buy", 1))
    p.offset_cl_sell = int(tp.get("off_cl_sell", 1))
    p.offset_ma_sell = int(tp.get("off_ma_sell", 1))
    p.buy_operator   = str(tp.get("buy_op", ">"))
    p.sell_operator  = str(tp.get("sell_op", "<"))
    p.use_trend_buy  = bool(tp.get("use_trend_buy", True))
    p.use_trend_sell = bool(tp.get("use_trend_sell", False))
    p.ma_trend_short = int(tp.get("ma_ts", 20))
    p.ma_trend_long  = int(tp.get("ma_tl", 50))
    p.stop_loss_pct  = float(tp.get("sl", 0.0))
    p.take_profit_pct= float(tp.get("tp", 0.0))
    p.use_atr_stop   = bool(tp.get("use_atr", False))
    p.atr_multiplier = float(tp.get("atr_mult", 2.0))
    if p.use_atr_stop: p.stop_loss_pct = 0.0
    return p

def _slice_data(data, start, end):
    sliced = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            sliced[k] = v[start:end]
        elif isinstance(v, pd.DataFrame):
            sliced[k] = v.iloc[start:end].reset_index(drop=True)
        elif isinstance(v, dict) and k == "sig_ind":
            sliced[k] = {
                ik: (iv[start:end] if isinstance(iv, np.ndarray)
                     else {pp: arr[start:end] for pp, arr in iv.items()})
                for ik, iv in v.items()
            }
        else:
            sliced[k] = v
    return sliced

def run_optimization(
    signal_ticker, trade_ticker, start_date, end_date,
    base_params, search_space, constraints,
    n_trials=100, target="수익률 (%)", disable_tp=False, progress_cb=None,
):
    """
    Optuna 최적화 실행 (Train/Test 분리 없음 - 전체 기간 최적화).
    target = "다중 목적 (수익률↑ + MDD↓)" 일 때 Pareto Front 반환.
    Returns: (DataFrame, study)
    """
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    data_full = prepare_data(
        signal_ticker, trade_ticker, base_params.market_ticker,
        start_date, end_date, base_params
    )
    if data_full is None:
        st.error("데이터 로드 실패")
        return pd.DataFrame(), None

    sampler = optuna.samplers.TPESampler(seed=42)
    if is_multi:
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        p = _build_params_from_trial(trial, search_space, base_params, disable_tp)

        res = run_backtest(data_full, p)
        if not res.is_valid or res.total_trades < constraints.min_trades:
            raise optuna.TrialPruned()
        if res.win_rate_pct < constraints.min_win_rate:
            raise optuna.TrialPruned()
        if constraints.max_mdd > 0 and abs(res.mdd_pct) > constraints.max_mdd:
            raise optuna.TrialPruned()

        if is_multi:
            return res.total_return_pct, abs(res.mdd_pct)
        elif target == "Profit Factor":
            return min(res.profit_factor, 999.0)
        elif target == "승률 (%)":
            return res.win_rate_pct
        elif target == "MDD 최소화":
            return -abs(res.mdd_pct)
        else:
            return res.total_return_pct

    def _cb(study, trial):
        if progress_cb: progress_cb(trial.number + 1, n_trials)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)
    except Exception as e:
        st.toast(f"⚠️ 최적화 오류: {e}", icon="⚠️")

    trials = study.best_trials if is_multi else [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    rows = []
    for t in trials:
        tp  = t.params
        p   = _params_from_dict(tp, base_params)
        res = run_backtest(data_full, p)
        if not res.is_valid: continue
        if res.total_trades < constraints.min_trades: continue
        if constraints.max_mdd > 0 and abs(res.mdd_pct) > constraints.max_mdd: continue
        rows.append({
            "수익률(%)":   res.total_return_pct,
            "MDD(%)":      res.mdd_pct,
            "승률(%)":     res.win_rate_pct,
            "PF":          res.profit_factor,
            "매매횟수":    res.total_trades,
            "ma_buy":          tp.get("ma_buy"),
            "ma_sell":         tp.get("ma_sell"),
            "offset_cl_buy":   tp.get("off_cl_buy"),
            "offset_ma_buy":   tp.get("off_ma_buy"),
            "offset_cl_sell":  tp.get("off_cl_sell"),
            "offset_ma_sell":  tp.get("off_ma_sell"),
            "buy_operator":    tp.get("buy_op"),
            "sell_operator":   tp.get("sell_op"),
            "use_trend_buy":   tp.get("use_trend_buy"),
            "use_trend_sell":  tp.get("use_trend_sell"),
            "ma_trend_short":  tp.get("ma_ts"),
            "ma_trend_long":   tp.get("ma_tl"),
            "stop_loss_pct":   tp.get("sl"),
            "take_profit_pct": tp.get("tp", 0.0),
            "use_atr_stop":    tp.get("use_atr"),
            "atr_multiplier":  tp.get("atr_mult"),
            "use_rsi":         tp.get("use_rsi", False),
            "rsi_period":      tp.get("rsi_period", 14),
            "rsi_max":         tp.get("rsi_max", 70),
            "use_bollinger":   tp.get("use_bb", False),
            "bb_period":       tp.get("bb_period", 20),
            "bb_std":          tp.get("bb_std", 2.0),
            "use_macd":        tp.get("use_macd", False),
        })

    if not rows:
        return pd.DataFrame(), study

    return pd.DataFrame(rows).sort_values("수익률(%)", ascending=False), study

def apply_optimal_params(row):
    p = StrategyParams()
    p.ma_buy         = int(row.get("ma_buy", 50))
    p.ma_sell        = int(row.get("ma_sell", 10))
    p.offset_cl_buy  = int(row.get("offset_cl_buy", 1))
    p.offset_ma_buy  = int(row.get("offset_ma_buy", 1))
    p.offset_cl_sell = int(row.get("offset_cl_sell", 1))
    p.offset_ma_sell = int(row.get("offset_ma_sell", 1))
    p.buy_operator   = str(row.get("buy_operator", ">"))
    p.sell_operator  = str(row.get("sell_operator", "<"))
    p.use_trend_buy  = bool(row.get("use_trend_buy", True))
    p.use_trend_sell = bool(row.get("use_trend_sell", False))
    p.ma_trend_short = int(row.get("ma_trend_short", 20))
    p.ma_trend_long  = int(row.get("ma_trend_long", 50))
    p.stop_loss_pct  = float(row.get("stop_loss_pct", 0.0))
    p.take_profit_pct= float(row.get("take_profit_pct", 0.0))
    p.use_atr_stop   = bool(row.get("use_atr_stop", False))
    p.atr_multiplier = float(row.get("atr_multiplier", 2.0))
    return p
