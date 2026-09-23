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
        key = st.secrets.get("fred_api_key", None)
        if key:
            return str(key).strip()
        # 섹션 없이 직접 접근 시도
        for k in ["fred_api_key", "FRED_API_KEY", "fred_key"]:
            try:
                val = getattr(st.secrets, k, None)
                if val:
                    return str(val).strip()
            except Exception:
                pass
        return None
    except Exception:
        return None


def fetch_fred_series(series_id: str, start_date: str = "1990-01-01") -> pd.DataFrame:
    api_key = _get_fred_api_key()
    if not api_key:
        return pd.DataFrame({"error": ["API 키 없음"]})

    try:
        resp = requests.get(FRED_BASE, params={
            "series_id":         series_id,
            "api_key":           api_key,
            "file_type":         "json",
            "observation_start": start_date,
            "observation_end":   date.today().strftime("%Y-%m-%d"),
        }, timeout=15)
        
        if resp.status_code != 200:
            return pd.DataFrame({"error": [f"HTTP {resp.status_code}: {resp.text[:100]}"]})
        
        json_data = resp.json()
        observations = json_data.get("observations", [])
        
        if not observations:
            error_msg = json_data.get("error_message", "observations 없음")
            return pd.DataFrame({"error": [error_msg]})
        
        df = pd.DataFrame(observations)[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_shiller_cape(start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    Shiller CAPE 데이터 가져오기.
    Yale 교수 공식 Excel 파일에서 직접 파싱.
    """
    try:
        # Yale Shiller 공식 데이터 (Excel)
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        df_raw = pd.read_excel(url, sheet_name="Data", header=7)
        # 컬럼: Date, P, D, E, CPI, Date Fraction, Long Interest Rate, Real Price, Real Dividend, Real Total Return Price, Real Earnings, Real TR Scaled Earnings, CAPE, TR CAPE, Excess CAPE Yield, Total Return Bond, Real Total Return Bond, 10-Year Annualized Stock Real Return, 10-Year Annualized Bond Real Return, Real 10-year Excess Annualized Returns
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        # 날짜 컬럼 찾기
        date_col = df_raw.columns[0]
        cape_col = "CAPE" if "CAPE" in df_raw.columns else [c for c in df_raw.columns if "CAPE" in str(c)][0]

        df = df_raw[[date_col, cape_col]].copy()
        df.columns = ["date_raw", "value"]
        df = df.dropna(subset=["value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # 날짜 변환 (1871.01 형식)
        def _parse_shiller_date(d):
            try:
                d = float(d)
                year = int(d)
                month = round((d - year) * 100)
                if month == 0: month = 1
                if month > 12: month = 12
                return pd.Timestamp(year=year, month=month, day=1)
            except Exception:
                return pd.NaT

        df["date"] = df["date_raw"].apply(_parse_shiller_date)
        df = df.dropna(subset=["date"])
        df = df[["date", "value"]].sort_values("date")
        df = df[df["date"] >= pd.to_datetime(start_date)]
        return df.reset_index(drop=True)

    except Exception:
        # fallback: FRED 시도
        return fetch_fred_series("CAPE", start_date)


def fetch_all_macro(start_date: str = "1990-01-01") -> dict:
    """모든 거시지표 한 번에 가져오기."""
    tips = fetch_fred_series("DFII10", start_date)
    cape = fetch_shiller_cape(start_date)

    ecy = pd.DataFrame()
    if not cape.empty and not tips.empty:
        try:
            cape_idx = cape.set_index("date")["value"]
            date_range = pd.date_range(
                max(cape["date"].min(), tips["date"].min()),
                min(cape["date"].max(), tips["date"].max()),
                freq="D"
            )
            cape_daily = cape_idx.reindex(date_range).ffill()
            tips_daily = tips.set_index("date")["value"].reindex(date_range).ffill()
            ecy_vals = (1.0 / cape_daily * 100) - tips_daily
            ecy_vals = ecy_vals.dropna()
            ecy = pd.DataFrame({
                "date":  ecy_vals.index,
                "value": ecy_vals.round(3).values
            }).reset_index(drop=True)
        except Exception:
            ecy = pd.DataFrame()

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
