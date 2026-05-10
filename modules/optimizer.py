"""
optimizer.py
============
전략 파라미터 자동 최적화.

기존 대비 개선사항:
- auto_search: 랜덤 탐색 → Optuna 베이지안 최적화 실제 연동
- Train/Test 분리 검증으로 과적합 방지
- 다중 목적 최적화 지원 (수익률↑ + MDD↓)
- 사용자 정의 탐색 공간 (SearchSpace dataclass)
- 제약 조건 필터링 (최소 거래횟수, 승률, MDD 한계)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional

from .engine import StrategyParams, BacktestResult, prepare_data, run_backtest

# Optuna 로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 1. 탐색 공간 정의
# ══════════════════════════════════════════════════════════

@dataclass
class SearchSpace:
    """Optuna가 탐색할 파라미터 후보 목록."""

    # 이평선 기간
    ma_buy_choices:  list = field(default_factory=lambda: [5, 10, 20, 50, 60, 120])
    ma_sell_choices: list = field(default_factory=lambda: [5, 10, 20, 50, 60, 120])

    # 오프셋
    offset_cl_buy_choices:  list = field(default_factory=lambda: [1, 5, 10, 20])
    offset_ma_buy_choices:  list = field(default_factory=lambda: [1, 5, 10, 20])
    offset_cl_sell_choices: list = field(default_factory=lambda: [1, 5, 10, 20])
    offset_ma_sell_choices: list = field(default_factory=lambda: [1, 5, 10, 20])

    # 연산자
    buy_operator_choices:  list = field(default_factory=lambda: [">", "<"])
    sell_operator_choices: list = field(default_factory=lambda: ["<", ">", "OFF"])

    # 추세 필터
    use_trend_buy_choices:  list = field(default_factory=lambda: [True, False])
    use_trend_sell_choices: list = field(default_factory=lambda: [True, False])
    ma_trend_short_choices: list = field(default_factory=lambda: [5, 10, 20, 50])
    ma_trend_long_choices:  list = field(default_factory=lambda: [20, 50, 60, 120, 200])

    # 손절/익절
    stop_loss_choices:    list = field(default_factory=lambda: [0.0, 10.0, 15.0, 20.0, 25.0])
    take_profit_choices:  list = field(default_factory=lambda: [0.0, 15.0, 25.0, 35.0])
    use_atr_stop_choices: list = field(default_factory=lambda: [True, False])
    atr_mult_choices:     list = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])


@dataclass
class OptimizeConstraints:
    """최적화 결과 필터 조건."""
    min_trades:    int   = 5
    min_win_rate:  float = 0.0
    max_mdd:       float = 0.0    # 절대값 기준 (0 = 제한 없음)
    min_train_ret: float = -999.0
    min_test_ret:  float = -999.0


# ══════════════════════════════════════════════════════════
# 2. Optuna Objective 함수
# ══════════════════════════════════════════════════════════

def _build_params_from_trial(trial: optuna.Trial, ss: SearchSpace, base_params: StrategyParams) -> StrategyParams:
    """Trial에서 StrategyParams 생성."""
    import copy
    p = copy.deepcopy(base_params)

    p.ma_buy           = trial.suggest_categorical("ma_buy",  ss.ma_buy_choices)
    p.ma_sell          = trial.suggest_categorical("ma_sell", ss.ma_sell_choices)
    p.offset_cl_buy    = trial.suggest_categorical("off_cl_buy",  ss.offset_cl_buy_choices)
    p.offset_ma_buy    = trial.suggest_categorical("off_ma_buy",  ss.offset_ma_buy_choices)
    p.offset_cl_sell   = trial.suggest_categorical("off_cl_sell", ss.offset_cl_sell_choices)
    p.offset_ma_sell   = trial.suggest_categorical("off_ma_sell", ss.offset_ma_sell_choices)
    p.buy_operator     = trial.suggest_categorical("buy_op",  ss.buy_operator_choices)
    p.sell_operator    = trial.suggest_categorical("sell_op", ss.sell_operator_choices)
    p.use_trend_buy    = trial.suggest_categorical("use_trend_buy",  ss.use_trend_buy_choices)
    p.use_trend_sell   = trial.suggest_categorical("use_trend_sell", ss.use_trend_sell_choices)
    p.ma_trend_short   = trial.suggest_categorical("ma_ts", ss.ma_trend_short_choices)
    p.ma_trend_long    = trial.suggest_categorical("ma_tl", ss.ma_trend_long_choices)
    p.stop_loss_pct    = trial.suggest_categorical("sl",  ss.stop_loss_choices)
    p.take_profit_pct  = trial.suggest_categorical("tp",  ss.take_profit_choices)
    p.use_atr_stop     = trial.suggest_categorical("use_atr", ss.use_atr_stop_choices)
    p.atr_multiplier   = trial.suggest_categorical("atr_mult", ss.atr_mult_choices)

    # 논리 오류 즉시 Prune: 단기 > 장기 이평
    if p.use_trend_buy or p.use_trend_sell:
        if p.ma_trend_short >= p.ma_trend_long:
            raise optuna.TrialPruned()

    # 손절/ATR 중복 방지 (ATR 쓰면 고정손절 0)
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0

    return p


def _make_objective(
    data_full: dict,
    data_train: dict,
    data_test:  dict,
    ss: SearchSpace,
    base_params: StrategyParams,
    target: str,
    constraints: OptimizeConstraints,
):
    """Optuna objective 클로저 생성."""

    def objective(trial: optuna.Trial):
        p = _build_params_from_trial(trial, ss, base_params)

        # Train 구간 백테스트
        res_tr = run_backtest(data_train, p)
        if not res_tr.is_valid:
            raise optuna.TrialPruned()
        if res_tr.total_trades < constraints.min_trades:
            raise optuna.TrialPruned()
        if res_tr.total_return_pct < constraints.min_train_ret:
            raise optuna.TrialPruned()

        # Test 구간 백테스트 (과적합 방지)
        res_te = run_backtest(data_test, p)
        if res_te.total_return_pct < constraints.min_test_ret:
            raise optuna.TrialPruned()

        # 전체 구간 성과
        res_full = run_backtest(data_full, p)
        if not res_full.is_valid:
            raise optuna.TrialPruned()
        if res_full.win_rate_pct < constraints.min_win_rate:
            raise optuna.TrialPruned()
        if constraints.max_mdd > 0 and abs(res_full.mdd_pct) > constraints.max_mdd:
            raise optuna.TrialPruned()

        # 목표 점수 반환
        if target == "수익률":
            return res_full.total_return_pct
        elif target == "Profit Factor":
            return min(res_full.profit_factor, 999.0)
        elif target == "승률":
            return res_full.win_rate_pct
        elif target == "MDD 최소화":
            return -abs(res_full.mdd_pct)   # MDD는 최소화이므로 음수화
        else:
            return res_full.total_return_pct

    return objective


# ══════════════════════════════════════════════════════════
# 3. 메인 최적화 함수
# ══════════════════════════════════════════════════════════

def run_optimization(
    signal_ticker: str,
    trade_ticker:  str,
    start_date,
    end_date,
    split_ratio:   float,
    base_params:   StrategyParams,
    search_space:  SearchSpace,
    constraints:   OptimizeConstraints,
    n_trials:      int = 100,
    target:        str = "수익률",
    progress_cb=None,   # Streamlit progress bar 콜백
) -> pd.DataFrame:
    """
    Optuna 베이지안 최적화 실행.

    Args:
        split_ratio: Train 비율 (예: 0.6 → 앞 60%가 Train, 뒤 40%가 Test)
        target: "수익률", "Profit Factor", "승률", "MDD 최소화"
        progress_cb: (current, total) → None 형태의 콜백

    Returns:
        결과 DataFrame (상위 결과 포함, 빈 DF일 수도 있음)
    """
    # 전체 데이터 로드
    data_full = prepare_data(
        signal_ticker, trade_ticker, "", start_date, end_date, base_params
    )
    if data_full is None:
        st.error("데이터 로드 실패")
        return pd.DataFrame()

    base_df    = data_full["base"]
    n          = len(base_df)
    split_idx  = int(n * split_ratio)

    # Train / Test 데이터 분리
    def _slice_data(data: dict, start: int, end: int) -> dict:
        sliced = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                sliced[k] = v[start:end]
            elif isinstance(v, pd.DataFrame):
                sliced[k] = v.iloc[start:end].reset_index(drop=True)
            elif isinstance(v, dict) and k == "sig_ind":
                sliced[k] = {
                    ik: (iv[start:end] if isinstance(iv, np.ndarray) else {
                        p: arr[start:end] for p, arr in iv.items()
                    })
                    for ik, iv in v.items()
                }
            else:
                sliced[k] = v
        return sliced

    data_train = _slice_data(data_full, 0, split_idx)
    data_test  = _slice_data(data_full, split_idx, n)

    # Optuna Study 생성
    sampler = optuna.samplers.TPESampler(seed=42)   # 재현 가능한 결과
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    objective = _make_objective(
        data_full, data_train, data_test,
        search_space, base_params, target, constraints,
    )

    # 진행 콜백
    results_cache = []

    def _callback(study: optuna.Study, trial: optuna.Trial):
        if progress_cb:
            progress_cb(trial.number + 1, n_trials)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)
    except Exception as e:
        st.toast(f"⚠️ 최적화 중 오류: {e}", icon="⚠️")

    # ── 결과 수집 ─────────────────────────────────────────
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        tp = trial.params
        p  = _build_params_from_trial_params(tp, base_params)

        res_full = run_backtest(data_full,  p)
        res_tr   = run_backtest(data_train, p)
        res_te   = run_backtest(data_test,  p)

        if not res_full.is_valid:
            continue

        rows.append({
            "Full_수익률(%)":   res_full.total_return_pct,
            "Full_MDD(%)":      res_full.mdd_pct,
            "Full_승률(%)":     res_full.win_rate_pct,
            "Full_PF":          res_full.profit_factor,
            "Full_매매횟수":    res_full.total_trades,
            "Train_수익률(%)":  res_tr.total_return_pct,
            "Test_수익률(%)":   res_te.total_return_pct,
            "Test_MDD(%)":      res_te.mdd_pct,
            # 파라미터
            "ma_buy":           tp.get("ma_buy"),
            "ma_sell":          tp.get("ma_sell"),
            "offset_cl_buy":    tp.get("off_cl_buy"),
            "offset_ma_buy":    tp.get("off_ma_buy"),
            "offset_cl_sell":   tp.get("off_cl_sell"),
            "offset_ma_sell":   tp.get("off_ma_sell"),
            "buy_operator":     tp.get("buy_op"),
            "sell_operator":    tp.get("sell_op"),
            "use_trend_buy":    tp.get("use_trend_buy"),
            "use_trend_sell":   tp.get("use_trend_sell"),
            "ma_trend_short":   tp.get("ma_ts"),
            "ma_trend_long":    tp.get("ma_tl"),
            "stop_loss_pct":    tp.get("sl"),
            "take_profit_pct":  tp.get("tp"),
            "use_atr_stop":     tp.get("use_atr"),
            "atr_multiplier":   tp.get("atr_mult"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Full_수익률(%)", ascending=False)
    return df


def _build_params_from_trial_params(tp: dict, base_params: StrategyParams) -> StrategyParams:
    """Trial.params dict → StrategyParams 변환 (결과 수집용)."""
    import copy
    p = copy.deepcopy(base_params)
    p.ma_buy          = int(tp.get("ma_buy", 50))
    p.ma_sell         = int(tp.get("ma_sell", 10))
    p.offset_cl_buy   = int(tp.get("off_cl_buy", 1))
    p.offset_ma_buy   = int(tp.get("off_ma_buy", 1))
    p.offset_cl_sell  = int(tp.get("off_cl_sell", 1))
    p.offset_ma_sell  = int(tp.get("off_ma_sell", 1))
    p.buy_operator    = str(tp.get("buy_op", ">"))
    p.sell_operator   = str(tp.get("sell_op", "<"))
    p.use_trend_buy   = bool(tp.get("use_trend_buy", True))
    p.use_trend_sell  = bool(tp.get("use_trend_sell", False))
    p.ma_trend_short  = int(tp.get("ma_ts", 20))
    p.ma_trend_long   = int(tp.get("ma_tl", 50))
    p.stop_loss_pct   = float(tp.get("sl", 0.0))
    p.take_profit_pct = float(tp.get("tp", 0.0))
    p.use_atr_stop    = bool(tp.get("use_atr", False))
    p.atr_multiplier  = float(tp.get("atr_mult", 2.0))
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0
    return p


def apply_optimal_params(row: pd.Series) -> StrategyParams:
    """
    최적화 결과 행 → StrategyParams 변환.
    main.py에서 "적용하기" 버튼 클릭 시 사용.
    """
    import copy
    p = StrategyParams()
    p.ma_buy          = int(row.get("ma_buy", 50))
    p.ma_sell         = int(row.get("ma_sell", 10))
    p.offset_cl_buy   = int(row.get("offset_cl_buy", 1))
    p.offset_ma_buy   = int(row.get("offset_ma_buy", 1))
    p.offset_cl_sell  = int(row.get("offset_cl_sell", 1))
    p.offset_ma_sell  = int(row.get("offset_ma_sell", 1))
    p.buy_operator    = str(row.get("buy_operator", ">"))
    p.sell_operator   = str(row.get("sell_operator", "<"))
    p.use_trend_buy   = bool(row.get("use_trend_buy", True))
    p.use_trend_sell  = bool(row.get("use_trend_sell", False))
    p.ma_trend_short  = int(row.get("ma_trend_short", 20))
    p.ma_trend_long   = int(row.get("ma_trend_long", 50))
    p.stop_loss_pct   = float(row.get("stop_loss_pct", 0.0))
    p.take_profit_pct = float(row.get("take_profit_pct", 0.0))
    p.use_atr_stop    = bool(row.get("use_atr_stop", False))
    p.atr_multiplier  = float(row.get("atr_multiplier", 2.0))
    return p
