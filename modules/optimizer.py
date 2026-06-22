"""
optimizer.py - 2단계 멀티시드 최적화
1단계: 축소 탐색 공간 × N시드 → 유망 파라미터 범위 추출
2단계: 좁혀진 공간 × N시드 → 정밀 탐색
결과 합산 → 중복 제거 → 정렬
"""
import copy, random, numpy as np, pandas as pd, optuna
from dataclasses import dataclass
from .engine import StrategyParams, prepare_data, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 탐색 후보 리스트 ──────────────────────────────────────
_MA_FULL     = [1] + list(range(5, 121, 5))    # 25개
_MA_REDUCED  = [5, 10, 20, 30, 50, 60, 80, 100, 120]  # 9개 (1단계용)
_OFF_FULL    = [1] + list(range(5, 61, 5))     # 13개
_OFF_REDUCED = [1, 5, 10, 20, 30, 45, 60]  # 7개 (1단계용, 전체 범위 커버)


@dataclass
class OptimizeConstraints:
    min_trades:   int   = 5
    min_win_rate: float = 0.0
    max_mdd:      float = 0.0  # 절대값 기준, 0 = 제한 없음


def _make_ma_list(center, half_range=15, step=5):
    """2단계용: 중심값 ± half_range 범위의 MA 후보 생성"""
    lo = max(1, center - half_range)
    hi = min(120, center + half_range)
    candidates = sorted(set(
        [lo, center, hi] +
        list(range(max(5, (lo // step) * step), hi + 1, step))
    ))
    return [c for c in candidates if 1 <= c <= 120] or [center]


def _make_off_list(center, half_range=10, step=5):
    """2단계용: 중심값 ± half_range 범위의 오프셋 후보 생성"""
    lo = max(1, center - half_range)
    hi = min(60, center + half_range)
    candidates = sorted(set(
        [lo, center, hi] +
        list(range(max(1, (lo // step) * step), hi + 1, step))
    ))
    return [c for c in candidates if 1 <= c <= 60] or [center]


def _build_params_from_trial(trial, base_params, ss_config, disable_tp,
                              ma_list, off_list):
    p = copy.deepcopy(base_params)

    p.ma_buy         = trial.suggest_categorical("ma_buy",      ma_list)
    p.offset_ma_buy  = trial.suggest_categorical("off_ma_buy",  off_list)
    p.offset_cl_buy  = trial.suggest_categorical("off_cl_buy",  off_list)
    p.buy_operator   = trial.suggest_categorical("buy_op",      [">", "<"])

    p.ma_sell        = trial.suggest_categorical("ma_sell",     ma_list)
    p.offset_ma_sell = trial.suggest_categorical("off_ma_sell", off_list)
    p.offset_cl_sell = trial.suggest_categorical("off_cl_sell", off_list)
    p.sell_operator  = trial.suggest_categorical("sell_op",     ["<", ">", "OFF"])

    # 추세 필터
    trend_buy_mode  = ss_config.get("trend_buy",  "search")
    trend_sell_mode = ss_config.get("trend_sell", "search")
    if trend_buy_mode == "search":
        p.use_trend_buy = True   # 탐색 포함 = 항상 ON, MA만 탐색
    elif trend_buy_mode == "fixed": p.use_trend_buy = True
    else: p.use_trend_buy = False

    if trend_sell_mode == "search":
        p.use_trend_sell = True  # 탐색 포함 = 항상 ON, MA만 탐색
    elif trend_sell_mode == "fixed": p.use_trend_sell = True
    else: p.use_trend_sell = False

    if p.use_trend_buy or p.use_trend_sell:
        p.ma_trend_short     = trial.suggest_categorical("ma_ts",  ma_list)
        p.ma_trend_long      = trial.suggest_categorical("ma_tl",  ma_list)
        p.offset_trend_short = trial.suggest_categorical("off_ts", off_list)
        p.offset_trend_long  = trial.suggest_categorical("off_tl", off_list)
        if p.ma_trend_short >= p.ma_trend_long:
            raise optuna.TrialPruned()

    # 손절/익절
    atr_mode = ss_config.get("atr", "search")
    if atr_mode == "search":
        p.use_atr_stop = trial.suggest_categorical("use_atr", [True, False])
    elif atr_mode == "fixed": p.use_atr_stop = True
    else: p.use_atr_stop = False

    if p.use_atr_stop:
        p.atr_multiplier = trial.suggest_float("atr_mult", 2.0, 4.0, step=0.5)
        p.stop_loss_pct  = 0.0
    else:
        p.stop_loss_pct  = trial.suggest_float("sl", 15.0, 35.0, step=5.0)

    p.take_profit_pct = 0.0 if disable_tp else trial.suggest_float("tp", 0.0, 50.0, step=5.0)

    # RSI
    rsi_mode = ss_config.get("rsi", "off")
    if rsi_mode == "search":
        p.use_rsi_filter = trial.suggest_categorical("use_rsi", [True, False])
    elif rsi_mode == "fixed": p.use_rsi_filter = True
    else: p.use_rsi_filter = False
    if p.use_rsi_filter and rsi_mode == "search":
        p.rsi_period = trial.suggest_categorical("rsi_period", [7, 10, 14, 21])
        rsi_max      = trial.suggest_categorical("rsi_max",    [60, 65, 70, 75, 80])
        p.rsi_max, p.rsi_min = rsi_max, 100 - rsi_max

    # MACD
    macd_mode = ss_config.get("macd", "off")
    if macd_mode == "search":
        p.use_macd = trial.suggest_categorical("use_macd", [True, False])
    elif macd_mode == "fixed": p.use_macd = True
    else: p.use_macd = False

    return p


def _params_from_trial_params(tp, base_params, ss_config=None):
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

    # use_trend: ss_config 기반으로 결정
    if ss_config:
        tbm = ss_config.get("trend_buy",  "off")
        tsm = ss_config.get("trend_sell", "off")
        p.use_trend_buy  = (tbm  in ("search", "fixed"))
        p.use_trend_sell = (tsm in ("search", "fixed"))
    else:
        p.use_trend_buy  = tp.get("use_trend_buy", False)
        p.use_trend_sell = tp.get("use_trend_sell", False)

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
    if p.use_atr_stop: p.stop_loss_pct = 0.0
    return p


def _run_single_study(data_full, base_params, ss_config, constraints,
                      n_trials, target, disable_tp, seed, ma_list, off_list,
                      progress_cb=None, progress_offset=0, progress_total=1):
    """단일 시드 × 단일 탐색 공간 최적화"""
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")
    sampler  = optuna.samplers.TPESampler(seed=seed)
    if is_multi:
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        try:
            p   = _build_params_from_trial(trial, base_params, ss_config,
                                           disable_tp, ma_list, off_list)
            res = run_backtest(data_full, p)
            if res is None or not res.is_valid: raise optuna.TrialPruned()
            if res.total_trades < constraints.min_trades: raise optuna.TrialPruned()
            if (res.win_rate_pct or 0) < constraints.min_win_rate: raise optuna.TrialPruned()
            if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd:
                raise optuna.TrialPruned()
            ret = float(res.total_return_pct or -999.0)
            mdd = float(abs(res.mdd_pct or 0))
            if is_multi: return ret, mdd
            elif target == "Profit Factor": return float(min(res.profit_factor or 0, 999.0))
            elif target == "승률 (%)":      return float(res.win_rate_pct or 0)
            elif target == "MDD 최소화":    return -mdd
            else: return ret
        except optuna.TrialPruned: raise
        except Exception: raise optuna.TrialPruned()

    def _cb(study, trial):
        if progress_cb:
            done = progress_offset + trial.number + 1
            progress_cb(done, progress_total)

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)
    return study


def _collect_rows(study, data_full, base_params, constraints, target, ss_config):
    """study에서 결과 rows 수집"""
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")
    trials   = study.best_trials if is_multi else [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    rows = []
    for t in trials:
        tp  = t.params
        p   = _params_from_trial_params(tp, base_params, ss_config)
        res = run_backtest(data_full, p)
        if not res.is_valid: continue
        if res.total_trades < constraints.min_trades: continue
        if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd: continue
        rows.append({
            "수익률(%)": res.total_return_pct, "MDD(%)": res.mdd_pct,
            "승률(%)": res.win_rate_pct, "PF": res.profit_factor,
            "매매횟수": res.total_trades,
            "ma_buy": tp.get("ma_buy"), "offset_cl_buy": tp.get("off_cl_buy"),
            "offset_ma_buy": tp.get("off_ma_buy"), "buy_operator": tp.get("buy_op"),
            "ma_sell": tp.get("ma_sell"), "offset_cl_sell": tp.get("off_cl_sell"),
            "offset_ma_sell": tp.get("off_ma_sell"), "sell_operator": tp.get("sell_op"),
            "use_trend_buy":  p.use_trend_buy,
            "use_trend_sell": p.use_trend_sell,
            "ma_trend_short": tp.get("ma_ts"), "ma_trend_long": tp.get("ma_tl"),
            "offset_trend_short": tp.get("off_ts"), "offset_trend_long": tp.get("off_tl"),
            "use_atr_stop": tp.get("use_atr", False), "atr_multiplier": tp.get("atr_mult"),
            "stop_loss_pct": tp.get("sl"), "take_profit_pct": tp.get("tp", 0.0),
            "use_rsi": tp.get("use_rsi", False), "rsi_period": tp.get("rsi_period"),
            "rsi_max": tp.get("rsi_max"), "use_macd": tp.get("use_macd", False),
        })
    return rows


def run_optimization(
    signal_ticker, trade_ticker, start_date, end_date,
    base_params, ss_config, constraints,
    # 1단계 설정
    stage1_trials=200, stage1_seeds=3,
    # 2단계 설정
    stage2_trials=100, stage2_seeds=3,
    target="수익률 (%)", disable_tp=False,
    progress_cb=None,
):
    """
    2단계 멀티시드 최적화.
    1단계: 축소 공간 × stage1_seeds개 시드
    2단계: 좁혀진 공간 × stage2_seeds개 시드
    """
    data_full = prepare_data(
        signal_ticker, trade_ticker, base_params.market_ticker,
        start_date, end_date, base_params
    )
    if data_full is None:
        return pd.DataFrame(), None

    total_trials = stage1_trials * stage1_seeds + stage2_trials * stage2_seeds
    progress_done = [0]

    def _prog(done, total):
        progress_done[0] = done
        if progress_cb: progress_cb(done, total_trials)

    # ════════════════════════════════════════
    # 1단계: 축소 공간으로 넓게 탐색
    # ════════════════════════════════════════
    seeds1 = [random.randint(0, 99999) for _ in range(stage1_seeds)]
    stage1_rows = []

    for i, seed in enumerate(seeds1):
        study = _run_single_study(
            data_full, base_params, ss_config, constraints,
            stage1_trials, target, disable_tp, seed,
            ma_list=_MA_REDUCED, off_list=_OFF_REDUCED,
            progress_cb=_prog,
            progress_offset=i * stage1_trials,
            progress_total=total_trials,
        )
        stage1_rows.extend(_collect_rows(study, data_full, base_params, constraints, target, ss_config))

    if not stage1_rows:
        return pd.DataFrame(), None

    # 1단계 결과에서 유망 파라미터 범위 추출 (상위 20%)
    df1 = pd.DataFrame(stage1_rows).sort_values("수익률(%)", ascending=False)
    top_n = max(3, len(df1) // 5)
    top   = df1.head(top_n)

    # 상위 결과의 MA/오프셋 중심값 계산
    def _mode_or_median(series):
        try: return int(series.mode().iloc[0])
        except: return int(series.median())

    ma_buy_center  = _mode_or_median(top["ma_buy"].dropna())
    ma_sell_center = _mode_or_median(top["ma_sell"].dropna())
    off_center     = _mode_or_median(
        pd.concat([top["offset_cl_buy"], top["offset_ma_buy"],
                   top["offset_cl_sell"], top["offset_ma_sell"]]).dropna()
    )

    # 2단계용 좁혀진 탐색 공간
    ma2  = sorted(set(_make_ma_list(ma_buy_center) + _make_ma_list(ma_sell_center)))
    off2 = _make_off_list(off_center)

    # ════════════════════════════════════════
    # 2단계: 좁혀진 공간으로 정밀 탐색
    # ════════════════════════════════════════
    seeds2 = [random.randint(0, 99999) for _ in range(stage2_seeds)]
    stage2_rows = []

    for i, seed in enumerate(seeds2):
        study = _run_single_study(
            data_full, base_params, ss_config, constraints,
            stage2_trials, target, disable_tp, seed,
            ma_list=ma2, off_list=off2,
            progress_cb=_prog,
            progress_offset=stage1_trials * stage1_seeds + i * stage2_trials,
            progress_total=total_trials,
        )
        stage2_rows.extend(_collect_rows(study, data_full, base_params, constraints, target, ss_config))

    # 1단계 + 2단계 합산, 중복 제거 (파라미터 기준), 정렬
    all_rows = stage1_rows + stage2_rows
    if not all_rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(all_rows)

    # 중복 제거: 핵심 파라미터가 동일한 행 제거
    key_cols = ["ma_buy", "ma_sell", "offset_cl_buy", "offset_ma_buy",
                "offset_cl_sell", "offset_ma_sell", "buy_operator", "sell_operator"]
    existing = [c for c in key_cols if c in df.columns]
    df = df.drop_duplicates(subset=existing)
    df = df.sort_values("수익률(%)", ascending=False).reset_index(drop=True)

    # 단계 정보 추가
    n1 = len(pd.DataFrame(stage1_rows).drop_duplicates(subset=existing)) if stage1_rows else 0
    df["단계"] = ["1단계" if i < n1 else "2단계" for i in range(len(df))]

    return df, None


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
    p.ma_trend_short     = int(row.get("ma_trend_short", 20) or 20)
    p.ma_trend_long      = int(row.get("ma_trend_long", 50) or 50)
    p.offset_trend_short = int(row.get("offset_trend_short", 1) or 1)
    p.offset_trend_long  = int(row.get("offset_trend_long", 1) or 1)
    p.use_atr_stop       = bool(row.get("use_atr_stop", False))
    p.atr_multiplier     = float(row.get("atr_multiplier", 2.0) or 2.0)
    p.stop_loss_pct      = float(row.get("stop_loss_pct", 0.0) or 0.0)
    p.take_profit_pct    = float(row.get("take_profit_pct", 0.0) or 0.0)
    return p


def _make_neighbor_list(center: int, half: int, step: int, lo: int, hi: int) -> list:
    """중심값 ± half 범위의 후보 리스트 생성"""
    vals = sorted(set(
        [center] +
        list(range(max(lo, center - half), min(hi, center + half) + 1, step))
    ))
    return [v for v in vals if lo <= v <= hi] or [center]


def run_preset_optimization(
    preset_params: StrategyParams,
    start_date, end_date,
    constraints: OptimizeConstraints,
    ma_half: int   = 15,
    off_half: int  = 10,
    sl_range: bool = True,
    tp_range: bool = True,
    n_trials: int  = 200,
    n_seeds: int   = 3,
    target: str    = "수익률 (%)",
    disable_tp: bool = False,
    progress_cb    = None,
) -> tuple:
    """
    등록된 전략 파라미터 근처에서 최적점 탐색.
    Returns: (결과 DataFrame, 현재 전략 BacktestResult)
    """
    p = preset_params
    is_multi = (target == "다중 목적 (수익률↑ + MDD↓)")

    data_full = prepare_data(
        p.signal_ticker, p.trade_ticker, p.market_ticker,
        start_date, end_date, p
    )
    if data_full is None:
        return pd.DataFrame(), None

    current_result = run_backtest(data_full, p)

    ma_buy_list  = _make_neighbor_list(p.ma_buy,  ma_half, 5, 1, 120)
    ma_sell_list = _make_neighbor_list(p.ma_sell, ma_half, 5, 1, 120)
    off_center   = int((p.offset_cl_buy + p.offset_ma_buy + p.offset_cl_sell + p.offset_ma_sell) / 4)
    off_list     = _make_neighbor_list(off_center, off_half, 5, 1, 60)

    total_trials = n_trials * n_seeds
    all_rows = []
    seeds = [random.randint(0, 99999) for _ in range(n_seeds)]

    for i, seed in enumerate(seeds):
        sampler = optuna.samplers.TPESampler(seed=seed)
        if is_multi:
            study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial, _p=p, _mb=ma_buy_list, _ms=ma_sell_list, _off=off_list):
            try:
                q = copy.deepcopy(_p)
                q.ma_buy         = trial.suggest_categorical("ma_buy",     _mb)
                q.offset_cl_buy  = trial.suggest_categorical("off_cl_buy", _off)
                q.offset_ma_buy  = trial.suggest_categorical("off_ma_buy", _off)
                q.buy_operator   = trial.suggest_categorical("buy_op",     list(set([_p.buy_operator, ">", "<"])))
                q.ma_sell        = trial.suggest_categorical("ma_sell",    _ms)
                q.offset_cl_sell = trial.suggest_categorical("off_cl_sell",_off)
                q.offset_ma_sell = trial.suggest_categorical("off_ma_sell",_off)
                q.sell_operator  = trial.suggest_categorical("sell_op",    list(set([_p.sell_operator, "<", ">", "OFF"])))

                if _p.use_trend_buy or _p.use_trend_sell:
                    ma_ts = _make_neighbor_list(_p.ma_trend_short, ma_half, 5, 1, 120)
                    ma_tl = _make_neighbor_list(_p.ma_trend_long,  ma_half, 5, 1, 120)
                    off_t = _make_neighbor_list(int((_p.offset_trend_short + _p.offset_trend_long)/2), off_half, 5, 1, 60)
                    q.ma_trend_short     = trial.suggest_categorical("ma_ts",  ma_ts)
                    q.ma_trend_long      = trial.suggest_categorical("ma_tl",  ma_tl)
                    q.offset_trend_short = trial.suggest_categorical("off_ts", off_t)
                    q.offset_trend_long  = trial.suggest_categorical("off_tl", off_t)
                    if q.ma_trend_short >= q.ma_trend_long:
                        raise optuna.TrialPruned()

                if sl_range and not _p.use_atr_stop and _p.stop_loss_pct > 0:
                    sl_lo = max(5.0,  _p.stop_loss_pct - 10)
                    sl_hi = min(50.0, _p.stop_loss_pct + 10)
                    q.stop_loss_pct = trial.suggest_float("sl", sl_lo, sl_hi, step=5.0)
                if tp_range and not disable_tp and _p.take_profit_pct > 0:
                    tp_lo = max(0.0,   _p.take_profit_pct - 15)
                    tp_hi = min(100.0, _p.take_profit_pct + 15)
                    q.take_profit_pct = trial.suggest_float("tp", tp_lo, tp_hi, step=5.0)
                if disable_tp: q.take_profit_pct = 0.0

                res = run_backtest(data_full, q)
                if res is None or not res.is_valid: raise optuna.TrialPruned()
                if res.total_trades < constraints.min_trades: raise optuna.TrialPruned()
                if (res.win_rate_pct or 0) < constraints.min_win_rate: raise optuna.TrialPruned()
                if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd:
                    raise optuna.TrialPruned()

                ret = float(res.total_return_pct or -999.0)
                mdd = float(abs(res.mdd_pct or 0))
                if is_multi:   return ret, mdd
                elif target == "Profit Factor": return float(min(res.profit_factor or 0, 999.0))
                elif target == "승률 (%)":      return float(res.win_rate_pct or 0)
                elif target == "MDD 최소화":    return -mdd
                else: return ret

            except optuna.TrialPruned: raise
            except Exception: raise optuna.TrialPruned()

        def _cb(study, trial, _off=i * n_trials):
            if progress_cb: progress_cb(_off + trial.number + 1, total_trials)

        study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

        trials = study.best_trials if is_multi else [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for t in trials:
            tp_ = t.params
            q   = copy.deepcopy(p)
            q.ma_buy         = tp_.get("ma_buy",     p.ma_buy)
            q.offset_cl_buy  = tp_.get("off_cl_buy", p.offset_cl_buy)
            q.offset_ma_buy  = tp_.get("off_ma_buy", p.offset_ma_buy)
            q.buy_operator   = tp_.get("buy_op",     p.buy_operator)
            q.ma_sell        = tp_.get("ma_sell",    p.ma_sell)
            q.offset_cl_sell = tp_.get("off_cl_sell",p.offset_cl_sell)
            q.offset_ma_sell = tp_.get("off_ma_sell",p.offset_ma_sell)
            q.sell_operator  = tp_.get("sell_op",    p.sell_operator)
            if p.use_trend_buy or p.use_trend_sell:
                q.ma_trend_short     = tp_.get("ma_ts",  p.ma_trend_short)
                q.ma_trend_long      = tp_.get("ma_tl",  p.ma_trend_long)
                q.offset_trend_short = tp_.get("off_ts", p.offset_trend_short)
                q.offset_trend_long  = tp_.get("off_tl", p.offset_trend_long)
            q.stop_loss_pct   = tp_.get("sl", p.stop_loss_pct)
            q.take_profit_pct = tp_.get("tp", p.take_profit_pct)
            if disable_tp: q.take_profit_pct = 0.0

            res = run_backtest(data_full, q)
            if not res.is_valid: continue
            if res.total_trades < constraints.min_trades: continue
            if constraints.max_mdd > 0 and abs(res.mdd_pct or 0) > constraints.max_mdd: continue
            all_rows.append({
                "수익률(%)":       res.total_return_pct,
                "MDD(%)":          res.mdd_pct,
                "승률(%)":         res.win_rate_pct,
                "PF":              res.profit_factor,
                "매매횟수":        res.total_trades,
                "ma_buy":          tp_.get("ma_buy"),
                "offset_cl_buy":   tp_.get("off_cl_buy"),
                "offset_ma_buy":   tp_.get("off_ma_buy"),
                "buy_operator":    tp_.get("buy_op"),
                "ma_sell":         tp_.get("ma_sell"),
                "offset_cl_sell":  tp_.get("off_cl_sell"),
                "offset_ma_sell":  tp_.get("off_ma_sell"),
                "sell_operator":   tp_.get("sell_op"),
                "ma_trend_short":  tp_.get("ma_ts"),
                "ma_trend_long":   tp_.get("ma_tl"),
                "stop_loss_pct":   tp_.get("sl"),
                "take_profit_pct": tp_.get("tp", 0.0),
            })

    if not all_rows:
        return pd.DataFrame(), current_result

    df = pd.DataFrame(all_rows)
    key_cols = ["ma_buy", "ma_sell", "offset_cl_buy", "offset_ma_buy",
                "offset_cl_sell", "offset_ma_sell", "buy_operator", "sell_operator"]
    df = df.drop_duplicates(subset=[c for c in key_cols if c in df.columns])
    df = df.sort_values("수익률(%)", ascending=False).reset_index(drop=True)
    return df, current_result
