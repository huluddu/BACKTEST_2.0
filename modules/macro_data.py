"""
macro_data.py - 거시경제 지표 데이터 로더
CAPE: 번들 CSV → Yale → multpl.com
TIPS: yfinance ^TNX - 2%
ECY:  CAPE + TIPS로 계산
"""
import os, time, io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

# 모듈 레벨 캐시 (st.cache_data/session_state 문제 회피)
_cache = {}
_cache_time = {}
_CACHE_TTL = 3600  # 1시간


def _get_fred_api_key():
    try:
        key = st.secrets.get("fred_api_key", None)
        if key: return str(key).strip()
        val = getattr(st.secrets, "fred_api_key", None)
        if val: return str(val).strip()
    except Exception:
        pass
    return None


def _load_cape_csv() -> pd.DataFrame:
    """번들 CSV에서 CAPE 로드"""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "cape_data.csv"),
        "/mount/src/backtest_2.0/data/cape_data.csv",
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["date"]  = pd.to_datetime(df["date"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame()


def fetch_shiller_cape(start_date: str = "1990-01-01") -> pd.DataFrame:
    cache_key = f"cape_{start_date}"
    if cache_key in _cache and (time.time() - _cache_time.get(cache_key, 0)) < _CACHE_TTL:
        return _cache[cache_key]

    df = _load_cape_csv()
    if df.empty:
        # Yale 시도
        try:
            import urllib.request
            req = urllib.request.urlopen(
                "http://www.econ.yale.edu/~shiller/data/ie_data.xls", timeout=20)
            df_raw = pd.read_excel(io.BytesIO(req.read()), sheet_name="Data", header=7)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            date_col = df_raw.columns[0]
            cape_col = [c for c in df_raw.columns if "CAPE" in str(c).upper()][0]
            tmp = df_raw[[date_col, cape_col]].copy()
            tmp.columns = ["date_raw", "value"]
            tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
            tmp = tmp.dropna(subset=["value"])
            def _p(d):
                try:
                    d=float(d); y=int(d); m=max(1,min(12,round((d-y)*100) or 1))
                    return pd.Timestamp(year=y,month=m,day=1)
                except: return pd.NaT
            tmp["date"] = tmp["date_raw"].apply(_p)
            tmp = tmp.dropna(subset=["date"])
            df = tmp[["date","value"]].sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame()

    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    _cache[cache_key] = df
    _cache_time[cache_key] = time.time()
    return df


def fetch_tips_via_yfinance(start_date: str = "2003-01-01") -> pd.DataFrame:
    cache_key = f"tips_{start_date}"
    if cache_key in _cache and (time.time() - _cache_time.get(cache_key, 0)) < _CACHE_TTL:
        return _cache[cache_key]

    try:
        import yfinance as yf
        tnx = yf.download("^TNX", start=start_date, progress=False, auto_adjust=True)
        if tnx is None or tnx.empty:
            return pd.DataFrame()
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx.columns = tnx.columns.get_level_values(0)
        tnx = tnx.reset_index()
        date_col  = "Date" if "Date" in tnx.columns else tnx.columns[0]
        close_col = "Close" if "Close" in tnx.columns else "close"
        df = pd.DataFrame({
            "date":  pd.to_datetime(tnx[date_col]),
            "value": pd.to_numeric(tnx[close_col], errors="coerce") - 2.0,
        })
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if not df.empty:
            _cache[cache_key] = df
            _cache_time[cache_key] = time.time()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_all_macro(start_date: str = "1990-01-01") -> dict:
    cape = fetch_shiller_cape(start_date)
    tips = fetch_tips_via_yfinance(start_date)

    ecy = pd.DataFrame()
    if not cape.empty and not tips.empty:
        try:
            date_range = pd.date_range(
                max(cape["date"].min(), tips["date"].min()),
                min(cape["date"].max(), tips["date"].max()),
                freq="D"
            )
            cape_d = cape.set_index("date")["value"].reindex(date_range).ffill()
            tips_d = tips.set_index("date")["value"].reindex(date_range).ffill()
            ecy_v  = (1.0 / cape_d * 100) - tips_d
            ecy_v  = ecy_v.dropna()
            ecy = pd.DataFrame({"date": ecy_v.index, "value": ecy_v.round(3).values})
        except Exception:
            pass

    return {"tips": tips, "cape": cape, "ecy": ecy}


def get_macro_value_at(df: pd.DataFrame, target_date) -> float | None:
    if df is None or df.empty: return None
    try:
        target_date = pd.to_datetime(target_date)
        past = df[df["date"] <= target_date]
        return float(past.iloc[-1]["value"]) if not past.empty else None
    except Exception:
        return None


def build_macro_filter_series(macro_data, tips_cfg=None, cape_cfg=None, ecy_cfg=None):
    all_dates = set()
    for key in ["tips", "cape", "ecy"]:
        df = macro_data.get(key)
        if df is not None and not df.empty and "date" in df.columns:
            all_dates.update(df["date"].dt.date.tolist())

    if not all_dates:
        return pd.Series(dtype=bool)

    date_range = pd.date_range(min(all_dates), max(all_dates), freq="D")
    result = pd.Series(True, index=date_range)

    def _apply(df, cfg):
        if df is None or df.empty or not cfg or not cfg.get("enabled"):
            return
        if "date" not in df.columns:
            return
        s = df.set_index("date")["value"].reindex(date_range).ffill()
        mode = cfg.get("mode", "value")
        op   = cfg.get("operator", ">")
        if mode == "value":
            thr = cfg.get("threshold", 0.0)
            mask = (s > thr) if op == ">" else (s < thr)
        else:
            ma_p = cfg.get("ma_period", 12)
            ma_op = cfg.get("ma_operator", ">")
            ma = s.rolling(window=ma_p, min_periods=1).mean()
            mask = (s > ma) if ma_op == ">" else (s < ma)
        result[~mask.fillna(False)] = False

    _apply(macro_data.get("tips"), tips_cfg)
    _apply(macro_data.get("cape"), cape_cfg)
    _apply(macro_data.get("ecy"),  ecy_cfg)

    return result
