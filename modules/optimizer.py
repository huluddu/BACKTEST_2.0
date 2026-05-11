"""
optimizer.py
============
전략 파라미터 자동 최적화 (기존 AI 지능형 풀옵션 + 베이지안 통합)

기능:
- 단일 목적: 수익률 / PF / 승률 / MDD 최소화
- 다중 목적: 수익률↑ + MDD↓ 동시 최적화 (Pareto Front)
- 익절 탐색 끄기 옵션 (기존 disable_tp 기능 그대로)
- AI 풀옵션 모드: MA 1~120 전체 + 오프셋 1~60 탐색
- Train/Test 분리 검증으로 과적합 방지
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import optuna
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional

from .engine import StrategyParams, BacktestResult, prepare_data, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 1. 탐색 공간 정의
# ══════════════════════════════════════════════════════════

# AI 풀옵션용 전체 탐색 리스트 (기존 optuna_objective와 동일)
_MA_FULL    = [1] + list(range(5, 121, 5))          # 1, 5, 10, ..., 120
_OFFSET_FULL = [1] + list(range(5, 61, 5))           # 1, 5, 10, ..., 60
_MA_SIMPLE  = [5, 10, 20, 50, 60, 120, 200]          # 현재 설정 기반용
_OFF_SIMPLE = [1, 5, 10, 20]


@dataclass
class SearchSpace:
    """Optuna 탐색 공간. AI 풀옵션 여부에 따라 팩토리 함수로 생성 권장."""

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


def make_full_search_space(
    ma_choices: list = None,
    use_trend: bool  = True,
    use_atr: bool    = True,
) -> SearchSpace:
    """
    AI 풀옵션 탐색 공간 생성.
    기존 optuna_objective의 ma_list / offset_list와 동일한 범위.
    """
    ma = ma_choices or _MA_FULL
    return SearchSpace(
        ma_buy_choices         = ma,
        ma_sell_choices        = ma,
        offset_cl_buy_choices  = _OFFSET_FULL,
        offset_ma_buy_choices  = _OFFSET_FULL,
        offset_cl_sell_choices = _OFFSET_FULL,
        offset_ma_sell_choices = _OFFSET_FULL,
        buy_operator_choices   = [">", "<"],
        sell_operator_choices  = ["<", ">", "OFF"],
        use_trend_buy_choices  = [True, False] if use_trend else [False],
        use_trend_sell_choices = [True, False] if use_trend else [False],
        ma_trend_short_choices = ma,
        ma_trend_long_choices  = ma,
        stop_loss_choices      = [15.0, 20.0, 25.0, 30.0, 35.0],
        take_profit_choices    = [0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices   = [True, False] if use_atr else [False],
        atr_mult_choices       = [2.0, 2.5, 3.0, 4.0],
    )


def make_simple_search_space(base_params: StrategyParams) -> SearchSpace:
    """현재 사이드바 설정 기반 탐색 공간 (연산자/필터는 현재값 고정)."""
    return SearchSpace(
        ma_buy_choices         = _MA_SIMPLE,
        ma_sell_choices        = _MA_SIMPLE,
        offset_cl_buy_choices  = _OFF_SIMPLE,
        offset_ma_buy_choices  = _OFF_SIMPLE,
        offset_cl_sell_choices = _OFF_SIMPLE,
        offset_ma_sell_choices = _OFF_SIMPLE,
        buy_operator_choices   = [base_params.buy_operator],
        sell_operator_choices  = [base_params.sell_operator],
        use_trend_buy_choices  = [base_params.use_trend_buy],
        use_trend_sell_choices = [base_params.use_trend_sell],
        ma_trend_short_choices = [5, 10, 20, 50],
        ma_trend_long_choices  = [20, 50, 60, 120, 200],
        stop_loss_choices      = [0.0, 10.0, 15.0, 20.0, 25.0, 35.0],
        take_profit_choices    = [0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices   = [base_params.use_atr_stop],
        atr_mult_choices       = [1.5, 2.0, 2.5, 3.0, 4.0],
    )


@dataclass
class OptimizeConstraints:
    """최적화 결과 필터 조건."""
    min_trades:    int   = 5
    min_win_rate:  float = 0.0
    max_mdd:       float = 0.0    # 0 = 제한 없음
    min_train_ret: float = -999.0
    min_test_ret:  float = -999.0


# ══════════════════════════════════════════════════════════
# 2. Trial → StrategyParams 변환
# ══════════════════════════════════════════════════════════

def _build_params_from_trial(
    trial: optuna.Trial,
    ss: SearchSpace,
    base_params: StrategyParams,
    disable_tp: bool = False,
) -> StrategyParams:
    """Optuna Trial에서 StrategyParams 생성."""
    p = copy.deepcopy(base_params)

    p.ma_buy          = trial.suggest_categorical("ma_buy",       ss.ma_buy_choices)
    p.ma_sell         = trial.suggest_categorical("ma_sell",      ss.ma_sell_choices)
    p.offset_cl_buy   = trial.suggest_categorical("off_cl_buy",   ss.offset_cl_buy_choices)
    p.offset_ma_buy   = trial.suggest_categorical("off_ma_buy",   ss.offset_ma_buy_choices)
    p.offset_cl_sell  = trial.suggest_categorical("off_cl_sell",  ss.offset_cl_sell_choices)
    p.offset_ma_sell  = trial.suggest_categorical("off_ma_sell",  ss.offset_ma_sell_choices)
    p.buy_operator    = trial.suggest_categorical("buy_op",       ss.buy_operator_choices)
    p.sell_operator   = trial.suggest_categorical("sell_op",      ss.sell_operator_choices)
    p.use_trend_buy   = trial.suggest_categorical("use_trend_buy",  ss.use_trend_buy_choices)
    p.use_trend_sell  = trial.suggest_categorical("use_trend_sell", ss.use_trend_sell_choices)
    p.ma_trend_short  = trial.suggest_categorical("ma_ts",        ss.ma_trend_short_choices)
    p.ma_trend_long   = trial.suggest_categorical("ma_tl",        ss.ma_trend_long_choices)
    p.stop_loss_pct   = trial.suggest_categorical("sl",           ss.stop_loss_choices)
    p.use_atr_stop    = trial.suggest_categorical("use_atr",      ss.use_atr_stop_choices)
    p.atr_multiplier  = trial.suggest_categorical("atr_mult",     ss.atr_mult_choices)

    # 익절: disable_tp=True이면 무조건 0 (기존 disable_tp_checkbox 기능)
    if disable_tp:
        p.take_profit_pct = 0.0
    else:
        p.take_profit_pct = trial.suggest_categorical("tp", ss.take_profit_choices)

    # 논리 오류 Prune: 추세선 단기 >= 장기
    if p.use_trend_buy or p.use_trend_sell:
        if p.ma_trend_short >= p.ma_trend_long:
            raise optuna.TrialPruned()

    # ATR 손절 사용 시 고정 손절은 0으로
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0

    return p


def _params_from_dict(tp: dict, base_params: StrategyParams) -> StrategyParams:
    """Trial.params dict → StrategyParams (결과 수집용)."""
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


# ══════════════════════════════════════════════════════════
# 3. Objective 클로저
# ══════════════════════════════════════════════════════════

def _make_objective(
    data_full:   dict,
    data_train:  dict,
    data_test:   dict,
    ss:          SearchSpace,
    base_params: StrategyParams,
    target:      str,
    constraints: OptimizeConstraints,
    disable_tp:  bool = False,
):
    """단일/다중 목적 공용 objective 클로저."""

    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    def objective(trial: optuna.Trial):
        p = _build_params_from_trial(trial, ss, base_params, disable_tp)

        # Train 구간
        res_tr = run_backtest(data_train, p)
        if not res_tr.is_valid or res_tr.total_trades < constraints.min_trades:
            return (-999.0, 999.0) if is_multi else -999.0
        if res_tr.total_return_pct < constraints.min_train_ret:
            return (-999.0, 999.0) if is_multi else -999.0

        # Test 구간
        res_te = run_backtest(data_test, p)
        if res_te.total_return_pct < constraints.min_test_ret:
            return (-999.0, 999.0) if is_multi else -999.0

        # 전체 구간
        res_full = run_backtest(data_full, p)
        if not res_full.is_valid:
            return (-999.0, 999.0) if is_multi else -999.0
        if res_full.win_rate_pct < constraints.min_win_rate:
            return (-999.0, 999.0) if is_multi else -999.0
        if constraints.max_mdd > 0 and abs(res_full.mdd_pct) > constraints.max_mdd:
            return (-999.0, 999.0) if is_multi else -999.0

        # 점수 반환
        if is_multi:
            # 수익률 최대화 + MDD 절대값 최소화 (기존 다중 목적과 동일)
            return res_full.total_return_pct, abs(res_full.mdd_pct)
        elif target == "Profit Factor":
            return min(res_full.profit_factor, 999.0)
        elif target == "승률 (%)":
            return res_full.win_rate_pct
        elif target == "MDD 최소화":
            return -abs(res_full.mdd_pct)
        else:  # 수익률 (%)
            return res_full.total_return_pct

    return objective


# ══════════════════════════════════════════════════════════
# 4. 데이터 슬라이싱 헬퍼
# ══════════════════════════════════════════════════════════

def _slice_data(data: dict, start: int, end: int) -> dict:
    sliced = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            sliced[k] = v[start:end]
        elif isinstance(v, pd.DataFrame):
            sliced[k] = v.iloc[start:end].reset_index(drop=True)
        elif isinstance(v, dict) and k == "sig_ind":
            sliced[k] = {
                ik: (
                    iv[start:end] if isinstance(iv, np.ndarray)
                    else {pp: arr[start:end] for pp, arr in iv.items()}
                )
                for ik, iv in v.items()
            }
        else:
            sliced[k] = v
    return sliced


# ══════════════════════════════════════════════════════════
# 5. 메인 최적화 함수
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
    n_trials:      int  = 100,
    target:        str  = "수익률 (%)",
    disable_tp:    bool = False,
    progress_cb         = None,
) -> tuple[pd.DataFrame, object]:
    """
    Optuna 최적화 실행.

    Returns:
        (결과 DataFrame, study 객체)
        - 단일 목적: DataFrame은 수익률 내림차순 정렬
        - 다중 목적: DataFrame은 Pareto Front 전체 포함, study도 반환
    """
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    # 데이터 준비
    data_full = prepare_data(
        signal_ticker, trade_ticker, base_params.market_ticker,
        start_date, end_date, base_params
    )
    if data_full is None:
        st.error("데이터 로드 실패")
        return pd.DataFrame(), None

    n         = len(data_full["base"])
    split_idx = int(n * split_ratio)
    data_train = _slice_data(data_full, 0, split_idx)
    data_test  = _slice_data(data_full, split_idx, n)

    # Study 생성
    sampler = optuna.samplers.TPESampler(seed=42)
    if is_multi:
        study = optuna.create_study(
            directions=["maximize", "minimize"],  # 수익률↑, MDD↓
            sampler=sampler,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = _make_objective(
        data_full, data_train, data_test,
        search_space, base_params, target, constraints, disable_tp,
    )

    def _cb(study, trial):
        if progress_cb:
            progress_cb(trial.number + 1, n_trials)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)
    except Exception as e:
        st.toast(f"⚠️ 최적화 중 오류: {e}", icon="⚠️")

    # ── 결과 수집 ─────────────────────────────────────────
    if is_multi:
        trials = study.best_trials  # Pareto Front만
    else:
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    rows = []
    for t in trials:
        tp = t.params
        p  = _params_from_dict(tp, base_params)

        res_full = run_backtest(data_full,  p)
        res_tr   = run_backtest(data_train, p)
        res_te   = run_backtest(data_test,  p)

        if not res_full.is_valid:
            continue

        row = {
            "Full_수익률(%)":  res_full.total_return_pct,
            "Full_MDD(%)":     res_full.mdd_pct,
            "Full_승률(%)":    res_full.win_rate_pct,
            "Full_PF":         res_full.profit_factor,
            "Full_매매횟수":   res_full.total_trades,
            "Train_수익률(%)": res_tr.total_return_pct,
            "Test_수익률(%)":  res_te.total_return_pct,
            "Test_MDD(%)":     res_te.mdd_pct,
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
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(), study

    df = pd.DataFrame(rows)
    if is_multi:
        df = df.sort_values("Full_수익률(%)", ascending=False)
    else:
        df = df.sort_values("Full_수익률(%)", ascending=False)

    return df, study


# ══════════════════════════════════════════════════════════
# 6. 결과 적용 헬퍼
# ══════════════════════════════════════════════════════════

def apply_optimal_params(row: pd.Series) -> StrategyParams:
    """최적화 결과 행 → StrategyParams."""
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
"""
optimizer.py
============
전략 파라미터 자동 최적화 (기존 AI 지능형 풀옵션 + 베이지안 통합)

기능:
- 단일 목적: 수익률 / PF / 승률 / MDD 최소화
- 다중 목적: 수익률↑ + MDD↓ 동시 최적화 (Pareto Front)
- 익절 탐색 끄기 옵션 (기존 disable_tp 기능 그대로)
- AI 풀옵션 모드: MA 1~120 전체 + 오프셋 1~60 탐색
- Train/Test 분리 검증으로 과적합 방지
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import optuna
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional

from .engine import StrategyParams, BacktestResult, prepare_data, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════
# 1. 탐색 공간 정의
# ══════════════════════════════════════════════════════════

# AI 풀옵션용 전체 탐색 리스트 (기존 optuna_objective와 동일)
_MA_FULL    = [1] + list(range(5, 121, 5))          # 1, 5, 10, ..., 120
_OFFSET_FULL = [1] + list(range(5, 61, 5))           # 1, 5, 10, ..., 60
_MA_SIMPLE  = [5, 10, 20, 50, 60, 120, 200]          # 현재 설정 기반용
_OFF_SIMPLE = [1, 5, 10, 20]


@dataclass
class SearchSpace:
    """Optuna 탐색 공간. AI 풀옵션 여부에 따라 팩토리 함수로 생성 권장."""

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


def make_full_search_space(
    ma_choices: list = None,
    use_trend: bool  = True,
    use_atr: bool    = True,
) -> SearchSpace:
    """
    AI 풀옵션 탐색 공간 생성.
    기존 optuna_objective의 ma_list / offset_list와 동일한 범위.
    """
    ma = ma_choices or _MA_FULL
    return SearchSpace(
        ma_buy_choices         = ma,
        ma_sell_choices        = ma,
        offset_cl_buy_choices  = _OFFSET_FULL,
        offset_ma_buy_choices  = _OFFSET_FULL,
        offset_cl_sell_choices = _OFFSET_FULL,
        offset_ma_sell_choices = _OFFSET_FULL,
        buy_operator_choices   = [">", "<"],
        sell_operator_choices  = ["<", ">", "OFF"],
        use_trend_buy_choices  = [True, False] if use_trend else [False],
        use_trend_sell_choices = [True, False] if use_trend else [False],
        ma_trend_short_choices = ma,
        ma_trend_long_choices  = ma,
        stop_loss_choices      = [15.0, 20.0, 25.0, 30.0, 35.0],
        take_profit_choices    = [0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices   = [True, False] if use_atr else [False],
        atr_mult_choices       = [2.0, 2.5, 3.0, 4.0],
    )


def make_simple_search_space(base_params: StrategyParams) -> SearchSpace:
    """현재 사이드바 설정 기반 탐색 공간 (연산자/필터는 현재값 고정)."""
    return SearchSpace(
        ma_buy_choices         = _MA_SIMPLE,
        ma_sell_choices        = _MA_SIMPLE,
        offset_cl_buy_choices  = _OFF_SIMPLE,
        offset_ma_buy_choices  = _OFF_SIMPLE,
        offset_cl_sell_choices = _OFF_SIMPLE,
        offset_ma_sell_choices = _OFF_SIMPLE,
        buy_operator_choices   = [base_params.buy_operator],
        sell_operator_choices  = [base_params.sell_operator],
        use_trend_buy_choices  = [base_params.use_trend_buy],
        use_trend_sell_choices = [base_params.use_trend_sell],
        ma_trend_short_choices = [5, 10, 20, 50],
        ma_trend_long_choices  = [20, 50, 60, 120, 200],
        stop_loss_choices      = [0.0, 10.0, 15.0, 20.0, 25.0, 35.0],
        take_profit_choices    = [0.0, 15.0, 25.0, 35.0, 50.0],
        use_atr_stop_choices   = [base_params.use_atr_stop],
        atr_mult_choices       = [1.5, 2.0, 2.5, 3.0, 4.0],
    )


@dataclass
class OptimizeConstraints:
    """최적화 결과 필터 조건."""
    min_trades:    int   = 5
    min_win_rate:  float = 0.0
    max_mdd:       float = 0.0    # 0 = 제한 없음
    min_train_ret: float = -999.0
    min_test_ret:  float = -999.0


# ══════════════════════════════════════════════════════════
# 2. Trial → StrategyParams 변환
# ══════════════════════════════════════════════════════════

def _build_params_from_trial(
    trial: optuna.Trial,
    ss: SearchSpace,
    base_params: StrategyParams,
    disable_tp: bool = False,
) -> StrategyParams:
    """Optuna Trial에서 StrategyParams 생성."""
    p = copy.deepcopy(base_params)

    p.ma_buy          = trial.suggest_categorical("ma_buy",       ss.ma_buy_choices)
    p.ma_sell         = trial.suggest_categorical("ma_sell",      ss.ma_sell_choices)
    p.offset_cl_buy   = trial.suggest_categorical("off_cl_buy",   ss.offset_cl_buy_choices)
    p.offset_ma_buy   = trial.suggest_categorical("off_ma_buy",   ss.offset_ma_buy_choices)
    p.offset_cl_sell  = trial.suggest_categorical("off_cl_sell",  ss.offset_cl_sell_choices)
    p.offset_ma_sell  = trial.suggest_categorical("off_ma_sell",  ss.offset_ma_sell_choices)
    p.buy_operator    = trial.suggest_categorical("buy_op",       ss.buy_operator_choices)
    p.sell_operator   = trial.suggest_categorical("sell_op",      ss.sell_operator_choices)
    p.use_trend_buy   = trial.suggest_categorical("use_trend_buy",  ss.use_trend_buy_choices)
    p.use_trend_sell  = trial.suggest_categorical("use_trend_sell", ss.use_trend_sell_choices)
    p.ma_trend_short  = trial.suggest_categorical("ma_ts",        ss.ma_trend_short_choices)
    p.ma_trend_long   = trial.suggest_categorical("ma_tl",        ss.ma_trend_long_choices)
    p.stop_loss_pct   = trial.suggest_categorical("sl",           ss.stop_loss_choices)
    p.use_atr_stop    = trial.suggest_categorical("use_atr",      ss.use_atr_stop_choices)
    p.atr_multiplier  = trial.suggest_categorical("atr_mult",     ss.atr_mult_choices)

    # 익절: disable_tp=True이면 무조건 0 (기존 disable_tp_checkbox 기능)
    if disable_tp:
        p.take_profit_pct = 0.0
    else:
        p.take_profit_pct = trial.suggest_categorical("tp", ss.take_profit_choices)

    # 논리 오류 Prune: 추세선 단기 >= 장기
    if p.use_trend_buy or p.use_trend_sell:
        if p.ma_trend_short >= p.ma_trend_long:
            raise optuna.TrialPruned()

    # ATR 손절 사용 시 고정 손절은 0으로
    if p.use_atr_stop:
        p.stop_loss_pct = 0.0

    return p


def _params_from_dict(tp: dict, base_params: StrategyParams) -> StrategyParams:
    """Trial.params dict → StrategyParams (결과 수집용)."""
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


# ══════════════════════════════════════════════════════════
# 3. Objective 클로저
# ══════════════════════════════════════════════════════════

def _make_objective(
    data_full:   dict,
    data_train:  dict,
    data_test:   dict,
    ss:          SearchSpace,
    base_params: StrategyParams,
    target:      str,
    constraints: OptimizeConstraints,
    disable_tp:  bool = False,
):
    """단일/다중 목적 공용 objective 클로저."""

    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    def objective(trial: optuna.Trial):
        p = _build_params_from_trial(trial, ss, base_params, disable_tp)

        # Train 구간
        res_tr = run_backtest(data_train, p)
        if not res_tr.is_valid or res_tr.total_trades < constraints.min_trades:
            return (-999.0, 999.0) if is_multi else -999.0
        if res_tr.total_return_pct < constraints.min_train_ret:
            return (-999.0, 999.0) if is_multi else -999.0

        # Test 구간
        res_te = run_backtest(data_test, p)
        if res_te.total_return_pct < constraints.min_test_ret:
            return (-999.0, 999.0) if is_multi else -999.0

        # 전체 구간
        res_full = run_backtest(data_full, p)
        if not res_full.is_valid:
            return (-999.0, 999.0) if is_multi else -999.0
        if res_full.win_rate_pct < constraints.min_win_rate:
            return (-999.0, 999.0) if is_multi else -999.0
        if constraints.max_mdd > 0 and abs(res_full.mdd_pct) > constraints.max_mdd:
            return (-999.0, 999.0) if is_multi else -999.0

        # 점수 반환
        if is_multi:
            # 수익률 최대화 + MDD 절대값 최소화 (기존 다중 목적과 동일)
            return res_full.total_return_pct, abs(res_full.mdd_pct)
        elif target == "Profit Factor":
            return min(res_full.profit_factor, 999.0)
        elif target == "승률 (%)":
            return res_full.win_rate_pct
        elif target == "MDD 최소화":
            return -abs(res_full.mdd_pct)
        else:  # 수익률 (%)
            return res_full.total_return_pct

    return objective


# ══════════════════════════════════════════════════════════
# 4. 데이터 슬라이싱 헬퍼
# ══════════════════════════════════════════════════════════

def _slice_data(data: dict, start: int, end: int) -> dict:
    sliced = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            sliced[k] = v[start:end]
        elif isinstance(v, pd.DataFrame):
            sliced[k] = v.iloc[start:end].reset_index(drop=True)
        elif isinstance(v, dict) and k == "sig_ind":
            sliced[k] = {
                ik: (
                    iv[start:end] if isinstance(iv, np.ndarray)
                    else {pp: arr[start:end] for pp, arr in iv.items()}
                )
                for ik, iv in v.items()
            }
        else:
            sliced[k] = v
    return sliced


# ══════════════════════════════════════════════════════════
# 5. 메인 최적화 함수
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
    n_trials:      int  = 100,
    target:        str  = "수익률 (%)",
    disable_tp:    bool = False,
    progress_cb         = None,
) -> tuple[pd.DataFrame, object]:
    """
    Optuna 최적화 실행.

    Returns:
        (결과 DataFrame, study 객체)
        - 단일 목적: DataFrame은 수익률 내림차순 정렬
        - 다중 목적: DataFrame은 Pareto Front 전체 포함, study도 반환
    """
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    # 데이터 준비
    data_full = prepare_data(
        signal_ticker, trade_ticker, base_params.market_ticker,
        start_date, end_date, base_params
    )
    if data_full is None:
        st.error("데이터 로드 실패")
        return pd.DataFrame(), None

    n         = len(data_full["base"])
    split_idx = int(n * split_ratio)
    data_train = _slice_data(data_full, 0, split_idx)
    data_test  = _slice_data(data_full, split_idx, n)

    # Study 생성
    sampler = optuna.samplers.TPESampler(seed=42)
    if is_multi:
        study = optuna.create_study(
            directions=["maximize", "minimize"],  # 수익률↑, MDD↓
            sampler=sampler,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = _make_objective(
        data_full, data_train, data_test,
        search_space, base_params, target, constraints, disable_tp,
    )

    def _cb(study, trial):
        if progress_cb:
            progress_cb(trial.number + 1, n_trials)

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)
    except Exception as e:
        st.toast(f"⚠️ 최적화 중 오류: {e}", icon="⚠️")

    # ── 결과 수집 ─────────────────────────────────────────
    if is_multi:
        trials = study.best_trials  # Pareto Front만
    else:
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    rows = []
    for t in trials:
        tp = t.params
        p  = _params_from_dict(tp, base_params)

        res_full = run_backtest(data_full,  p)
        res_tr   = run_backtest(data_train, p)
        res_te   = run_backtest(data_test,  p)

        if not res_full.is_valid:
            continue

        row = {
            "Full_수익률(%)":  res_full.total_return_pct,
            "Full_MDD(%)":     res_full.mdd_pct,
            "Full_승률(%)":    res_full.win_rate_pct,
            "Full_PF":         res_full.profit_factor,
            "Full_매매횟수":   res_full.total_trades,
            "Train_수익률(%)": res_tr.total_return_pct,
            "Test_수익률(%)":  res_te.total_return_pct,
            "Test_MDD(%)":     res_te.mdd_pct,
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
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(), study

    df = pd.DataFrame(rows)
    if is_multi:
        df = df.sort_values("Full_수익률(%)", ascending=False)
    else:
        df = df.sort_values("Full_수익률(%)", ascending=False)

    return df, study


# ══════════════════════════════════════════════════════════
# 6. 결과 적용 헬퍼
# ══════════════════════════════════════════════════════════

def apply_optimal_params(row: pd.Series) -> StrategyParams:
    """최적화 결과 행 → StrategyParams."""
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
