"""
portfolio.py
============
저장된 전략(프리셋) 전체를 일괄 백테스트하고 비교하는 모듈.
Tab2 (PRESETS) 기능 담당.
"""

from __future__ import annotations

import datetime
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

from .engine import StrategyParams, BacktestResult, prepare_data, run_backtest, get_today_signal
from .data_loader import get_data


# ══════════════════════════════════════════════════════════
# 1. 프리셋 dict → StrategyParams 변환
# ══════════════════════════════════════════════════════════

def preset_to_params(p: dict) -> StrategyParams:
    """저장된 전략 dict를 StrategyParams로 안전하게 변환."""

    def _int(key, default):
        try:
            return int(float(p.get(key, default) or default))
        except (TypeError, ValueError):
            return default

    def _float(key, default):
        try:
            return float(p.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _bool(key, default):
        v = p.get(key, default)
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ["true", "1", "t", "y"]

    def _str(key, default):
        return str(p.get(key, default) or default)

    # 티커 키 이름 호환 (구버전 키도 지원)
    sig = _str("signal_ticker_input", _str("signal_ticker", "SOXL"))
    trd = _str("trade_ticker_input",  _str("trade_ticker",  "SOXL"))
    mkt = _str("market_ticker_input", _str("market_ticker", "SPY"))

    sp = StrategyParams(
        signal_ticker   = sig,
        trade_ticker    = trd,
        market_ticker   = mkt,
        ma_buy          = _int("ma_buy", 50),
        offset_ma_buy   = _int("offset_ma_buy", 1),
        offset_cl_buy   = _int("offset_cl_buy", 1),
        buy_operator    = _str("buy_operator", ">"),
        ma_sell         = _int("ma_sell", 10),
        offset_ma_sell  = _int("offset_ma_sell", 1),
        offset_cl_sell  = _int("offset_cl_sell", 1),
        sell_operator   = _str("sell_operator", "<"),
        use_trend_buy   = _bool("use_trend_in_buy", True),
        use_trend_sell  = _bool("use_trend_in_sell", False),
        ma_trend_short  = _int("ma_compare_short", 20),
        ma_trend_long   = _int("ma_compare_long", 50),
        offset_trend_short = _int("offset_compare_short", 1),
        offset_trend_long  = _int("offset_compare_long", 1),
        use_bollinger   = _bool("use_bollinger", False),
        bb_period       = _int("bb_period", 20),
        bb_std          = _float("bb_std", 2.0),
        bb_entry_type   = _str("bb_entry_type", "상단선 돌파 (추세)"),
        bb_exit_type    = _str("bb_exit_type", "중심선(MA) 이탈"),
        use_rsi_filter  = _bool("use_rsi_filter", False),
        rsi_period      = _int("rsi_period", 14),
        rsi_max         = _float("rsi_max", 70.0),
        use_market_filter  = _bool("use_market_filter", False),
        market_ma_period   = _int("market_ma_period", 200),
        use_atr_stop    = _bool("use_atr_stop", False),
        atr_multiplier  = _float("atr_multiplier", 2.0),
        stop_loss_pct   = _float("stop_loss_pct", 0.0),
        take_profit_pct = _float("take_profit_pct", 0.0),
        min_hold_days   = _int("min_hold_days", 0),
        fee_bps         = _float("fee_bps", 25.0),
        slip_bps        = _float("slip_bps", 1.0),
    )
    return sp


# ══════════════════════════════════════════════════════════
# 2. 현재 설정 기준 일괄 분석 (Sub-tab 1)
# ══════════════════════════════════════════════════════════

def run_portfolio_scan(
    presets: dict,
    start_date,
    end_date,
    progress_placeholder=None,
) -> pd.DataFrame:
    """
    모든 프리셋에 대해 백테스트 + 오늘 시그널 + 보유 상태를 한 번에 분석.

    Returns: 결과 DataFrame
    """
    rows = []
    total = len(presets)

    for idx, (name, preset_dict) in enumerate(presets.items()):
        if progress_placeholder:
            progress_placeholder.progress(
                int(idx / total * 100),
                text=f"분석 중: {name} ({idx+1}/{total})"
            )

        try:
            p = preset_to_params(preset_dict)

            # 데이터 준비
            data = prepare_data(
                p.signal_ticker, p.trade_ticker, p.market_ticker,
                start_date, end_date, p
            )
            if data is None:
                rows.append(_error_row(name, p.trade_ticker, "데이터 없음"))
                continue

            # 백테스트
            result = run_backtest(data, p)

            # 오늘 시그널
            today_sig = get_today_signal(data, p)

            # 보유 상태 판단 (마지막 매매 로그 기준)
            hold_status    = "⚪ 미보유"
            buy_price_disp = "-"

            if result.trade_log:
                last = result.trade_log[-1]
                if last["신호"] == "BUY":
                    date_str    = pd.to_datetime(last["날짜"]).strftime("%Y-%m-%d")
                    hold_status = f"🟢 보유중 ({date_str})"
                    buy_price_disp = f"${last['체결가']:,.2f}"

            rows.append({
                "전략명":     name,
                "티커":       p.trade_ticker,
                "보유여부":   hold_status,
                "매수가":     buy_price_disp,
                "오늘신호":   today_sig["status"],
                "수익률(%)":  f"{result.total_return_pct}%",
                "B&H(%)":     f"{result.bh_return_pct}%",
                "MDD(%)":     f"{result.mdd_pct}%",
                "승률(%)":    f"{result.win_rate_pct}%",
                "PF":         result.profit_factor,
                "매매횟수":   result.total_trades,
            })

        except Exception as e:
            rows.append(_error_row(name, "?", str(e)))

    if progress_placeholder:
        progress_placeholder.empty()

    return pd.DataFrame(rows)


def _error_row(name: str, ticker: str, error: str) -> dict:
    return {
        "전략명": name, "티커": ticker,
        "보유여부": "❌ 에러", "매수가": "-", "오늘신호": error,
        "수익률(%)": "-", "B&H(%)": "-", "MDD(%)": "-",
        "승률(%)": "-", "PF": "-", "매매횟수": 0,
    }


# ══════════════════════════════════════════════════════════
# 3. 5/10/15/20년 구간 분석 (Sub-tab 2)
# ══════════════════════════════════════════════════════════

def run_period_stress_test(
    presets: dict,
    year_list: list = [5, 10, 15, 20],
    progress_placeholder=None,
) -> pd.DataFrame:
    """
    각 전략을 5/10/15/20년 구간별로 백테스트해서 멀티인덱스 DataFrame 반환.
    """
    today      = datetime.date.today()
    data_list  = []
    total_steps = len(presets) * len(year_list)
    step        = 0

    for name, preset_dict in presets.items():
        p        = preset_to_params(preset_dict)
        strategy_id = f"{name} ({p.trade_ticker})"
        row_data = {}

        for yr in year_list:
            step += 1
            if progress_placeholder:
                progress_placeholder.progress(
                    int(step / total_steps * 100),
                    text=f"[{name}] {yr}년 분석 중..."
                )

            start_d = today - datetime.timedelta(days=365 * yr)

            try:
                data = prepare_data(
                    p.signal_ticker, p.trade_ticker, p.market_ticker,
                    start_d, today, p
                )
                if data is None:
                    for cat in ["수익률", "MDD", "승률", "매매횟수"]:
                        row_data[(cat, f"{yr}년")] = "-"
                    continue

                result = run_backtest(data, p)

                # 실제 데이터 시작일 확인 (데이터가 yr년치에 못 미칠 수 있음)
                real_start  = data["base"]["Date"].iloc[0].date()
                years_avail = round((today - real_start).days / 365, 1)
                suffix      = f" ({years_avail}y)" if years_avail < (yr - 0.5) else ""

                row_data[("수익률", f"{yr}년")] = f"{result.total_return_pct}%{suffix}"
                row_data[("MDD",    f"{yr}년")] = f"{result.mdd_pct}%"
                row_data[("승률",   f"{yr}년")] = f"{result.win_rate_pct}%"
                row_data[("매매횟수", f"{yr}년")] = f"{result.total_trades}회"

            except Exception as e:
                for cat in ["수익률", "MDD", "승률", "매매횟수"]:
                    row_data[(cat, f"{yr}년")] = "Err"

        row_data[("전략", "이름")] = strategy_id
        data_list.append(row_data)

    if progress_placeholder:
        progress_placeholder.empty()

    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list)
    if ("전략", "이름") in df.columns:
        df = df.set_index(("전략", "이름"))
        df.index.name = "전략명 (매매종목)"

    # 컬럼 순서 정리
    desired = []
    for cat in ["수익률", "MDD", "승률", "매매횟수"]:
        for yr in year_list:
            desired.append((cat, f"{yr}년"))
    final_cols = [c for c in desired if c in df.columns]

    return df[final_cols] if final_cols else df
