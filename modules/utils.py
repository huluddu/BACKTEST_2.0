"""
utils.py
========
구글 시트 연동 및 공통 유틸리티.

개선사항 (vs 기존):
- raise e 제거 → 앱 전체가 뻗는 버그 수정
- 저장/로드 실패 시 조용히 return (UX 개선)
- 컬럼 타입 안전 변환 보장
- 차트 공통 함수 추가 (engine과 분리)
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Optional

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


# ══════════════════════════════════════════════════════════
# 1. 구글 시트 연결
# ══════════════════════════════════════════════════════════

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

def _get_gc():
    """gspread 클라이언트 반환. secrets 없으면 None."""
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.toast(f"⚠️ 구글 인증 실패: {e}", icon="⚠️")
        return None


def _get_worksheet(sheet_name: str, tab_name: str):
    """특정 시트의 워크시트 반환. 실패 시 None."""
    gc = _get_gc()
    if gc is None:
        return None
    try:
        sh = gc.open(sheet_name)
        try:
            return sh.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=tab_name, rows=500, cols=50)
    except Exception as e:
        st.toast(f"⚠️ 시트 접근 실패 ({sheet_name}/{tab_name}): {e}", icon="⚠️")
        return None


# ══════════════════════════════════════════════════════════
# 2. 전략 저장 / 로드
# ══════════════════════════════════════════════════════════

def save_strategy(sheet_name: str, tab_name: str, strategy_name: str, params: dict) -> bool:
    """
    전략 파라미터를 구글 시트에 저장.
    기존 동일 이름 전략은 덮어씀.

    Returns: True(성공) / False(실패)
    """
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return False

    try:
        all_data = ws.get_all_values()

        # 헤더 없으면 생성
        if not all_data:
            headers = ["strategy_name"] + sorted(params.keys())
            ws.append_row(headers)
            all_data = [headers]

        headers    = all_data[0]
        new_row    = [strategy_name] + [str(params.get(h, "")) for h in headers[1:]]

        # 기존 행 찾기 → 업데이트 or 추가
        name_col   = 1
        found_row  = None
        for i, row in enumerate(all_data[1:], start=2):
            if row and row[0] == strategy_name:
                found_row = i
                break

        if found_row:
            ws.delete_rows(found_row)
            ws.insert_row(new_row, found_row)
        else:
            ws.append_row(new_row)

        st.toast(f"✅ '{strategy_name}' 저장 완료", icon="✅")
        return True

    except Exception as e:
        st.toast(f"❌ 저장 실패: {e}", icon="❌")
        return False   # [버그 수정] 기존: raise e → 앱 전체가 에러 화면으로 전환되던 문제 수정


def load_strategies(sheet_name: str, tab_name: str) -> dict:
    """
    구글 시트에서 모든 전략 로드.

    Returns: {strategy_name: {param_key: value}, ...}
    실패 시 빈 dict 반환 (앱 계속 작동)
    """
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return {}

    try:
        all_data = ws.get_all_values()
        if len(all_data) < 2:
            return {}

        headers  = all_data[0]
        result   = {}

        for row in all_data[1:]:
            if not row or not row[0]:
                continue
            name   = row[0]
            params = {}
            for i, h in enumerate(headers[1:], start=1):
                params[h] = row[i] if i < len(row) else ""
            result[name] = params

        return result

    except Exception as e:
        st.toast(f"⚠️ 전략 로드 실패: {e}", icon="⚠️")
        return {}


def delete_strategy(sheet_name: str, tab_name: str, strategy_name: str) -> bool:
    """
    전략 1개 삭제.

    Returns: True(성공) / False(실패)
    """
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return False

    try:
        cell = ws.find(strategy_name, in_column=1)
        if cell:
            ws.delete_rows(cell.row)
            st.toast(f"🗑️ '{strategy_name}' 삭제 완료", icon="🗑️")
            return True
        else:
            st.toast(f"⚠️ '{strategy_name}' 찾을 수 없음", icon="⚠️")
            return False

    except Exception as e:
        st.toast(f"❌ 삭제 실패: {e}", icon="❌")
        return False


def get_strategy_names(sheet_name: str, tab_name: str) -> list:
    """전략 이름 목록만 빠르게 조회."""
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return []
    try:
        col = ws.col_values(1)
        return [v for v in col[1:] if v]  # 헤더 제외
    except Exception:
        return []


# ══════════════════════════════════════════════════════════
# 3. 매매일지 저장 / 로드
# ══════════════════════════════════════════════════════════

JOURNAL_COLS = [
    "날짜", "종목", "신호", "체결가", "수량",
    "매수금액", "현재가", "평가손익(%)", "메모"
]

def save_journal_row(sheet_name: str, tab_name: str, row: dict) -> bool:
    """매매일지 한 행 추가."""
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return False

    try:
        existing = ws.get_all_values()
        if not existing:
            ws.append_row(JOURNAL_COLS)

        new_row = [str(row.get(c, "")) for c in JOURNAL_COLS]
        ws.append_row(new_row)
        st.toast("✅ 매매일지 저장 완료", icon="✅")
        return True

    except Exception as e:
        st.toast(f"❌ 매매일지 저장 실패: {e}", icon="❌")
        return False


def load_journal(sheet_name: str, tab_name: str) -> pd.DataFrame:
    """매매일지 전체 로드."""
    ws = _get_worksheet(sheet_name, tab_name)
    if ws is None:
        return pd.DataFrame(columns=JOURNAL_COLS)

    try:
        data = ws.get_all_values()
        if len(data) < 2:
            return pd.DataFrame(columns=JOURNAL_COLS)

        df = pd.DataFrame(data[1:], columns=data[0])
        return df

    except Exception as e:
        st.toast(f"⚠️ 매매일지 로드 실패: {e}", icon="⚠️")
        return pd.DataFrame(columns=JOURNAL_COLS)


# ══════════════════════════════════════════════════════════
# 4. 공통 성과 지표 계산
# ══════════════════════════════════════════════════════════

def calc_monthly_returns(asset_curve: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """
    월별 수익률 계산 → 히트맵용 DataFrame 반환.

    Returns:
        index: 연도, columns: 월(1~12), values: 월 수익률(%)
    """
    if len(asset_curve) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date":  pd.to_datetime(dates).values,
        "asset": asset_curve,
    }).set_index("date")

    monthly = df["asset"].resample("ME").last()
    monthly_ret = monthly.pct_change() * 100

    result = {}
    for date, ret in monthly_ret.items():
        yr  = date.year
        mo  = date.month
        if yr not in result:
            result[yr] = {}
        result[yr][mo] = round(float(ret), 2) if not np.isnan(ret) else None

    df_out = pd.DataFrame(result).T
    df_out.columns = [f"{int(c)}월" for c in df_out.columns]
    df_out.index.name = "연도"
    return df_out


def calc_annual_returns(asset_curve: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """연간 수익률 계산."""
    if len(asset_curve) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date":  pd.to_datetime(dates).values,
        "asset": asset_curve,
    }).set_index("date")

    yearly = df["asset"].resample("YE").last()
    yearly_ret = yearly.pct_change() * 100

    return pd.DataFrame({
        "연도":     [d.year for d in yearly_ret.index],
        "수익률(%)": [round(float(v), 2) if not np.isnan(v) else None for v in yearly_ret.values],
    }).dropna().set_index("연도")


def format_result_metric(value: float, suffix: str = "%", positive_green: bool = True) -> str:
    """
    수익률/MDD 등 지표를 색상 있는 문자열로 포맷.
    Streamlit st.metric delta 값으로도 사용 가능.
    """
    sign   = "+" if value > 0 else ""
    return f"{sign}{value:.2f}{suffix}"


def sharpe_ratio(asset_curve: np.ndarray, risk_free: float = 0.04) -> float:
    """
    연율화 샤프 비율 계산.
    risk_free: 연 무위험 수익률 (기본 4%)
    """
    if len(asset_curve) < 2:
        return 0.0

    daily_ret = np.diff(asset_curve) / asset_curve[:-1]
    excess    = daily_ret - (risk_free / 252)
    std       = np.std(excess)

    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def calmar_ratio(total_return_pct: float, mdd_pct: float) -> float:
    """Calmar Ratio = 연간수익률 / |MDD|"""
    if mdd_pct == 0:
        return 0.0
    return round(abs(total_return_pct) / abs(mdd_pct), 2)


# ══════════════════════════════════════════════════════════
# 5. 세션 상태 관리 헬퍼
# ══════════════════════════════════════════════════════════

def init_session_state(defaults: dict):
    """세션 상태 초기화 (없는 키만 설정)."""
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_state(key: str, default=None):
    """세션 상태 값 안전 조회."""
    return st.session_state.get(key, default)


def set_state(key: str, value):
    """세션 상태 값 설정."""
    st.session_state[key] = value
