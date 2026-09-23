"""
macro_data.py - 거시경제 지표 데이터 로더
FRED API로 TIPS 실질금리, CAPE, ECY 가져오기
"""
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import date, timedelta


FRED_BASE = "https://api.fred.stlouisfed.org/series/observations"


def _get_fred_api_key() -> str | None:
    """Streamlit secrets에서 FRED API 키 가져오기"""
    try:
        return st.secrets["fred_api_key"]
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fred_series(series_id: str, start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    FRED에서 시계열 데이터 가져오기.
    Returns: DataFrame with columns [date, value]
    """
    api_key = _get_fred_api_key()
    if not api_key:
        return pd.DataFrame()

    try:
        resp = requests.get(FRED_BASE, params={
            "series_id":       series_id,
            "api_key":         api_key,
            "file_type":       "json",
            "observation_start": start_date,
            "observation_end": date.today().strftime("%Y-%m-%d"),
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        df = pd.DataFrame(data)[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_shiller_cape(start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    Shiller CAPE 데이터 가져오기.
    FRED 시리즈: CAPE (Cyclically Adjusted Price-to-Earnings Ratio)
    """
    # FRED에 Shiller CAPE가 없으면 직접 계산
    # FRED: 'MULTPL/SHILLER_PE_RATIO_MONTH' 는 Quandl
    # 대신 FRED의 'CAPE' 시리즈 또는 계산
    df = fetch_fred_series("CAPE", start_date)
    if df.empty:
        # fallback: S&P500 실질주가 / 10년 실질이익으로 근사
        # FRED에 직접 CAPE 시리즈가 없으므로 Yale 데이터 사용
        try:
            url = "https://shiller-data.vercel.app/api/cape"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                raw = resp.json()
                records = raw.get("data", raw) if isinstance(raw, dict) else raw
                df = pd.DataFrame(records)
                if "date" in df.columns and "cape" in df.columns:
                    df = df[["date", "cape"]].rename(columns={"cape": "value"})
                    df["date"]  = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna().sort_values("date")
                    df = df[df["date"] >= pd.to_datetime(start_date)]
                    return df.reset_index(drop=True)
        except Exception:
            pass
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_macro(start_date: str = "1990-01-01") -> dict:
    """
    모든 거시지표 한 번에 가져오기.
    Returns: {
        "tips":  DataFrame (date, value) - 10년 TIPS 실질금리 %
        "cape":  DataFrame (date, value) - Shiller CAPE
        "ecy":   DataFrame (date, value) - Excess CAPE Yield %
    }
    """
    # TIPS 실질금리 (일간)
    tips = fetch_fred_series("DFII10", start_date)

    # CAPE (월간) - FRED 직접 또는 계산
    cape = fetch_shiller_cape(start_date)

    # ECY = (1/CAPE) - TIPS/100
    ecy = pd.DataFrame()
    if not cape.empty and not tips.empty:
        try:
            # CAPE를 월간 → 일간으로 forward fill
            cape_daily = cape.set_index("date").reindex(
                pd.date_range(cape["date"].min(), date.today(), freq="D")
            ).ffill().reset_index().rename(columns={"index": "date"})

            tips_idx = tips.set_index("date")["value"]
            ecy_rows = []
            for _, row in cape_daily.iterrows():
                d = row["date"]
                if d in tips_idx.index:
                    tips_val = tips_idx[d]
                    cape_val = row["value"]
                    if pd.notna(cape_val) and cape_val > 0:
                        ecy_val = (1 / cape_val * 100) - tips_val
                        ecy_rows.append({"date": d, "value": round(ecy_val, 3)})

            ecy = pd.DataFrame(ecy_rows) if ecy_rows else pd.DataFrame()
        except Exception:
            pass

    return {"tips": tips, "cape": cape, "ecy": ecy}


def get_macro_value_at(df: pd.DataFrame, target_date) -> float | None:
    """특정 날짜의 지표값 (없으면 가장 최근 이전값)"""
    if df is None or df.empty:
        return None
    try:
        target_date = pd.to_datetime(target_date)
        past = df[df["date"] <= target_date]
        if past.empty:
            return None
        return float(past.iloc[-1]["value"])
    except Exception:
        return None


def compute_macro_ma(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """이동평균 계산 (일간 데이터 기준)"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df = df.copy().sort_values("date")
        df["ma"] = df["value"].rolling(window=window_days, min_periods=1).mean()
        return df
    except Exception:
        return df


def build_macro_filter_series(
    macro_data: dict,
    tips_cfg:  dict | None = None,
    cape_cfg:  dict | None = None,
    ecy_cfg:   dict | None = None,
) -> pd.Series:
    """
    날짜별 매매 허용 여부 시리즈 생성.
    각 cfg: {
        "enabled": bool,
        "mode": "value" | "ma_cross",  # value: 절대값 비교, ma_cross: 이평선 크로스
        "operator": ">" | "<",
        "threshold": float,             # value 모드
        "ma_period": int,               # ma_cross 모드
        "ma_operator": ">" | "<",       # ma_cross: value > MA면 허용
    }
    Returns: pd.Series(index=date, value=True/False)
              True = 매매 허용, False = 매매 금지
    """
    # 전체 날짜 범위
    all_dates = set()
    for key in ["tips", "cape", "ecy"]:
        df = macro_data.get(key)
        if df is not None and not df.empty:
            all_dates.update(df["date"].dt.date.tolist())

    if not all_dates:
        return pd.Series(dtype=bool)

    date_range = pd.date_range(min(all_dates), max(all_dates), freq="D")
    result = pd.Series(True, index=date_range)

    def _apply_filter(df, cfg):
        if df is None or df.empty or not cfg.get("enabled"): return
        df = df.copy().sort_values("date")
        df = df.set_index("date").reindex(date_range).ffill()

        mode = cfg.get("mode", "value")
        op   = cfg.get("operator", ">")

        if mode == "value":
            thr = cfg.get("threshold", 0.0)
            if op == ">":
                mask = df["value"] > thr
            else:
                mask = df["value"] < thr
            result[mask == False] = False

        elif mode == "ma_cross":
            ma_p = cfg.get("ma_period", 12)
            ma_op = cfg.get("ma_operator", ">")
            df["ma"] = df["value"].rolling(window=ma_p, min_periods=1).mean()
            if ma_op == ">":
                mask = df["value"] > df["ma"]
            else:
                mask = df["value"] < df["ma"]
            result[mask == False] = False

    _apply_filter(macro_data.get("tips"), tips_cfg)
    _apply_filter(macro_data.get("cape"), cape_cfg)
    _apply_filter(macro_data.get("ecy"),  ecy_cfg)

    return result
