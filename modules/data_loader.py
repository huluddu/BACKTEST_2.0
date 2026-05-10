"""
data_loader.py
==============
주가 데이터 및 펀더멘털 정보 로더.

개선사항 (vs 기존):
- except: pass → 명시적 에러 로깅 (st.toast)
- FDR/yfinance 이중 백업 구조 유지
- 숫자 컬럼 dtype 강제 변환 (float64)
- 빈 DF 반환 시 항상 표준 컬럼 구조 보장
- get_fundamental_info: 반환값 타입 보장 (None → 0)
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ── 표준 컬럼 구조 ──────────────────────────────────────
REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
EMPTY_DF = pd.DataFrame(columns=REQUIRED_COLS)


# ══════════════════════════════════════════════════════════
# 내부 헬퍼
# ══════════════════════════════════════════════════════════

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명을 표준화하고 타입을 강제 변환.
    실패 시 EMPTY_DF 반환.
    """
    try:
        # 날짜 컬럼 찾기 & 이름 통일
        df = df.copy()
        col_lower = {c.lower(): c for c in df.columns}

        if "date" in col_lower:
            df.rename(columns={col_lower["date"]: "Date"}, inplace=True)
        elif df.index.name and "date" in df.index.name.lower():
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        else:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        # OHLCV 컬럼 통일 (대소문자 무관)
        col_lower = {c.lower(): c for c in df.columns}
        for std in ["Open", "High", "Low", "Close", "Volume"]:
            if std.lower() in col_lower and col_lower[std.lower()] != std:
                df.rename(columns={col_lower[std.lower()]: std}, inplace=True)

        # Close 없으면 포기
        if "Close" not in df.columns:
            return EMPTY_DF

        # 없는 OHLC는 Close로 채움
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0

        # 타입 변환
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(float)

        # 정리
        df = df.dropna(subset=["Date", "Close"])
        df = df[~df["Date"].dt.dayofweek.isin([5, 6])]   # 주말 제거
        df = df.drop_duplicates(subset=["Date"], keep="last")
        df = df.sort_values("Date").reset_index(drop=True)

        return df[REQUIRED_COLS]

    except Exception as e:
        st.toast(f"⚠️ 데이터 표준화 실패: {e}", icon="⚠️")
        return EMPTY_DF


def _clip_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """시작/종료일 기준으로 데이터를 자름."""
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# 메인 데이터 로더
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=600)
def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """
    주가 데이터 로드.
    FDR → yfinance 순서로 시도. 둘 다 실패하면 EMPTY_DF 반환.

    Args:
        ticker: 종목 코드 (예: "AAPL", "005930", "SOXL")
        start_date: 시작일 (date 또는 str)
        end_date: 종료일 (date 또는 str)

    Returns:
        표준화된 DataFrame (Date, Open, High, Low, Close, Volume)
        실패 시 빈 DataFrame (컬럼은 동일하게 유지)
    """
    if not ticker or not str(ticker).strip():
        return EMPTY_DF

    ticker = str(ticker).strip().upper()

    # yfinance end는 exclusive라 +1일 필요
    # 단, 오늘 이후 날짜는 의미없으므로 min(end_date+1, 오늘+1)로 제한
    # → 장중에 미완성 당일 데이터가 포함되는 것을 방지
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_plus1 = (pd.to_datetime(end_date) + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_adj   = min(end_plus1, tomorrow)
    start_str = str(start_date)

    # ── 1차 시도: FinanceDataReader ──────────────────────
    if FDR_AVAILABLE:
        try:
            df = fdr.DataReader(ticker, start_str, end_adj)
            if df is not None and not df.empty:
                df = _standardize(df)
                if not df.empty:
                    return _clip_dates(df, start_date, end_date)
        except Exception as e:
            st.toast(f"FDR 실패 ({ticker}): {e} → yfinance 시도", icon="ℹ️")

    # ── 2차 시도: yfinance ───────────────────────────────
    if YF_AVAILABLE:
        try:
            # 한국 종목 코드 처리
            yf_code = f"{ticker}.KS" if ticker.isdigit() else ticker
            df = yf.download(
                yf_code,
                start=start_str,
                end=end_adj,
                progress=False,
                auto_adjust=True,
            )
            if df is not None and not df.empty:
                # yfinance MultiIndex 컬럼 처리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df = _standardize(df)
                if not df.empty:
                    return _clip_dates(df, start_date, end_date)
        except Exception as e:
            st.toast(f"yfinance 실패 ({ticker}): {e}", icon="⚠️")

    st.toast(f"❌ 데이터 로드 완전 실패: {ticker}", icon="❌")
    return EMPTY_DF


# ══════════════════════════════════════════════════════════
# 펀더멘털 정보
# ══════════════════════════════════════════════════════════

def _safe_float(val, default=0.0) -> float:
    """None/NaN/문자열을 안전하게 float으로 변환."""
    try:
        result = float(val)
        return default if (np.isnan(result) or np.isinf(result)) else result
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False, ttl=3600)
def get_fundamental_info(ticker: str) -> dict:
    """
    yfinance 기반 기업 기본정보 로드.
    실패 시 기본값 dict 반환 (항상 동일 구조 보장).
    """
    default = {
        "Name": ticker,
        "Symbol": ticker,
        "Sector": "N/A",
        "Industry": "N/A",
        "MarketCap": 0,
        "Beta": 0.0,
        "PER": 0.0,
        "PBR": 0.0,
        "ROE": 0.0,
        "NetIncome": 0,
        "DividendYield": 0.0,
        "52W_High": 0.0,
        "52W_Low": 0.0,
        "AvgVolume": 0,
        "Description": "정보를 불러올 수 없습니다.",
    }

    if not YF_AVAILABLE:
        return default

    try:
        yf_code = f"{ticker}.KS" if str(ticker).isdigit() else ticker
        info = yf.Ticker(yf_code).info
        if not info:
            return default

        return {
            "Name":          info.get("longName", ticker),
            "Symbol":        info.get("symbol", ticker),
            "Sector":        info.get("sector", "N/A") or "N/A",
            "Industry":      info.get("industry", "N/A") or "N/A",
            "MarketCap":     int(info.get("marketCap", 0) or 0),
            "Beta":          _safe_float(info.get("beta")),
            "PER":           _safe_float(info.get("trailingPE")),
            "PBR":           _safe_float(info.get("priceToBook")),
            "ROE":           _safe_float(info.get("returnOnEquity")),
            "NetIncome":     int(info.get("netIncomeToCommon", 0) or 0),
            "DividendYield": _safe_float(info.get("dividendYield")),
            "52W_High":      _safe_float(info.get("fiftyTwoWeekHigh")),
            "52W_Low":       _safe_float(info.get("fiftyTwoWeekLow")),
            "AvgVolume":     int(info.get("averageVolume", 0) or 0),
            "Description":   info.get("longBusinessSummary", "정보 없음") or "정보 없음",
        }

    except Exception as e:
        st.toast(f"⚠️ 펀더멘털 로드 실패 ({ticker}): {e}", icon="⚠️")
        return default
