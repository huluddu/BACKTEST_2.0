"""
indicators.py
=============
모든 보조지표 계산을 담당하는 모듈.

설계 원칙:
- 입력/출력 모두 numpy ndarray (pandas 의존 최소화 → 속도)
- 각 함수는 독립적으로 테스트 가능
- NaN은 계산 불가 구간에만 사용 (앞쪽 워밍업 구간)
- 인덱스 의미: result[i]는 i번째 봉(bar) 기준값
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════
# 1. 이동평균 (Moving Average)
# ══════════════════════════════════════════════════════════

def sma(x: np.ndarray, period: int) -> np.ndarray:
    """
    단순이동평균 (Simple Moving Average)
    - period=1이면 원본 그대로 반환
    - np.convolve 기반으로 pandas rolling보다 ~3배 빠름
    """
    if period <= 1:
        return x.astype(float)
    if len(x) < period:
        return np.full(len(x), np.nan)

    result = np.full(len(x), np.nan, dtype=float)
    kernel = np.ones(period, dtype=float) / period
    conv = np.convolve(x.astype(float), kernel, mode="valid")
    result[period - 1:] = conv
    return result


def ema(x: np.ndarray, period: int) -> np.ndarray:
    """
    지수이동평균 (Exponential Moving Average)
    - MACD, Stochastic 등에 사용
    """
    result = np.full(len(x), np.nan, dtype=float)
    if len(x) < period:
        return result

    k = 2.0 / (period + 1)
    # 첫 EMA = 첫 period개의 SMA
    result[period - 1] = np.mean(x[:period])
    for i in range(period, len(x)):
        result[i] = x[i] * k + result[i - 1] * (1 - k)
    return result


# ══════════════════════════════════════════════════════════
# 2. 볼린저 밴드 (Bollinger Bands)
# ══════════════════════════════════════════════════════════

def bollinger_bands(x: np.ndarray, period: int, std_mult: float) -> tuple:
    """
    볼린저 밴드 계산
    Returns: (mid, upper, lower) — 모두 numpy ndarray
    """
    mid = sma(x, period)
    # 워밍업 구간은 NaN이므로 nanstd 기반 rolling 처리
    std = np.full(len(x), np.nan, dtype=float)
    for i in range(period - 1, len(x)):
        std[i] = np.std(x[i - period + 1: i + 1], ddof=0)

    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


# ══════════════════════════════════════════════════════════
# 3. RSI (Relative Strength Index)
# ══════════════════════════════════════════════════════════

def rsi(x: np.ndarray, period: int) -> np.ndarray:
    """
    RSI 계산 (Wilder's Smoothing 방식)
    - 과매수/과매도 필터에 사용
    """
    result = np.full(len(x), np.nan, dtype=float)
    if len(x) < period + 1:
        return result

    delta = np.diff(x.astype(float))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # 첫 평균 (단순평균)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


# ══════════════════════════════════════════════════════════
# 4. MACD (Moving Average Convergence Divergence)
# ══════════════════════════════════════════════════════════

def macd(x: np.ndarray,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> tuple:
    """
    MACD 계산
    Returns:
        macd_line   : EMA(fast) - EMA(slow)
        signal_line : EMA(macd_line, signal)
        histogram   : macd_line - signal_line

    매수 신호: histogram > 0 (MACD가 시그널선 위)
    매도 신호: histogram < 0
    골든크로스: macd_line이 signal_line을 상향 돌파
    """
    fast_ema = ema(x, fast)
    slow_ema = ema(x, slow)

    macd_line = fast_ema - slow_ema
    # NaN 구간 처리 후 signal 계산
    valid_start = slow - 1
    macd_valid = macd_line.copy()
    macd_valid[:valid_start] = np.nan

    signal_line = np.full(len(x), np.nan, dtype=float)
    # signal EMA는 MACD의 유효값부터 시작
    non_nan = np.where(~np.isnan(macd_valid))[0]
    if len(non_nan) >= signal:
        start = non_nan[signal - 1]
        signal_line[start] = np.nanmean(macd_valid[non_nan[:signal]])
        k = 2.0 / (signal + 1)
        for i in range(start + 1, len(x)):
            if not np.isnan(macd_valid[i]):
                signal_line[i] = macd_valid[i] * k + signal_line[i - 1] * (1 - k)
            else:
                signal_line[i] = np.nan

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ══════════════════════════════════════════════════════════
# 5. Stochastic Oscillator
# ══════════════════════════════════════════════════════════

def stochastic(high: np.ndarray,
               low: np.ndarray,
               close: np.ndarray,
               k_period: int = 14,
               d_period: int = 3) -> tuple:
    """
    스토캐스틱 계산
    Returns: (k_line, d_line)
    - k_line: (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - d_line: SMA(k_line, d_period)

    80 이상: 과매수, 20 이하: 과매도
    """
    n = len(close)
    k_line = np.full(n, np.nan, dtype=float)

    for i in range(k_period - 1, n):
        low_min = np.min(low[i - k_period + 1: i + 1])
        high_max = np.max(high[i - k_period + 1: i + 1])
        denom = high_max - low_min
        if denom == 0:
            k_line[i] = 50.0
        else:
            k_line[i] = (close[i] - low_min) / denom * 100.0

    d_line = sma(k_line, d_period)
    return k_line, d_line


# ══════════════════════════════════════════════════════════
# 6. ATR (Average True Range)
# ══════════════════════════════════════════════════════════

def atr(high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    ATR 계산 (Wilder's Smoothing)
    - 변동성 기반 손절 계산에 사용
    - result[i] = i번째 봉 기준 ATR값
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=float)
    if n < 2:
        return result

    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    if n < period:
        return result

    # 첫 ATR = period개의 TR 단순평균
    result[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


# ══════════════════════════════════════════════════════════
# 7. 모멘텀 / Rate of Change
# ══════════════════════════════════════════════════════════

def momentum(x: np.ndarray, period: int = 10) -> np.ndarray:
    """
    모멘텀: Close[i] - Close[i - period]
    양수 = 상승 추세, 음수 = 하락 추세
    """
    result = np.full(len(x), np.nan, dtype=float)
    for i in range(period, len(x)):
        result[i] = x[i] - x[i - period]
    return result


def rate_of_change(x: np.ndarray, period: int = 10) -> np.ndarray:
    """
    ROC: (Close[i] - Close[i-period]) / Close[i-period] * 100
    모멘텀의 % 버전. 종목 간 비교에 유리
    """
    result = np.full(len(x), np.nan, dtype=float)
    for i in range(period, len(x)):
        prev = x[i - period]
        if prev != 0:
            result[i] = (x[i] - prev) / prev * 100.0
    return result


# ══════════════════════════════════════════════════════════
# 8. 통합 지표 계산 함수 (engine에서 한 번에 호출)
# ══════════════════════════════════════════════════════════

def compute_all(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ma_periods: list,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    stoch_k: int = 14,
    stoch_d: int = 3,
    mom_period: int = 10,
) -> dict:
    """
    엔진에서 필요한 모든 지표를 한 번에 계산해서 dict로 반환.
    이 함수를 한 번 호출하면 이후 루프에서 배열 인덱싱만 하면 됨.

    Returns:
        {
            "ma": {period: ndarray, ...},   # 요청한 모든 MA
            "rsi": ndarray,
            "macd_line": ndarray,
            "macd_signal": ndarray,
            "macd_hist": ndarray,
            "bb_mid": ndarray,
            "bb_upper": ndarray,
            "bb_lower": ndarray,
            "atr": ndarray,
            "stoch_k": ndarray,
            "stoch_d": ndarray,
            "momentum": ndarray,
            "roc": ndarray,
        }
    """
    result = {}

    # MA (중복 제거 후 계산)
    result["ma"] = {}
    for p in sorted(set(int(p) for p in ma_periods if p and p > 0)):
        result["ma"][p] = sma(close, p)

    # RSI
    result["rsi"] = rsi(close, rsi_period)

    # MACD
    ml, sl, hist = macd(close, macd_fast, macd_slow, macd_signal)
    result["macd_line"] = ml
    result["macd_signal"] = sl
    result["macd_hist"] = hist

    # 볼린저 밴드
    bb_m, bb_u, bb_l = bollinger_bands(close, bb_period, bb_std)
    result["bb_mid"] = bb_m
    result["bb_upper"] = bb_u
    result["bb_lower"] = bb_l

    # ATR
    result["atr"] = atr(high, low, close, atr_period)

    # Stochastic
    sk, sd = stochastic(high, low, close, stoch_k, stoch_d)
    result["stoch_k"] = sk
    result["stoch_d"] = sd

    # 모멘텀
    result["momentum"] = momentum(close, mom_period)
    result["roc"] = rate_of_change(close, mom_period)

    return result


# ══════════════════════════════════════════════════════════
# 9. 유틸리티 (신호 판단용 헬퍼)
# ══════════════════════════════════════════════════════════

def safe_get(arr: np.ndarray, idx: int, offset: int = 0) -> float:
    """
    배열에서 (idx - offset) 위치의 값을 안전하게 가져옴.
    범위를 벗어나거나 NaN이면 np.nan 반환.
    """
    target = idx - offset
    if target < 0 or target >= len(arr):
        return np.nan
    val = arr[target]
    return float(val)


def is_valid(val: float) -> bool:
    """NaN/inf 체크"""
    return not (np.isnan(val) or np.isinf(val))


def compare(a: float, b: float, operator: str) -> bool:
    """
    두 값을 operator로 비교.
    operator: '>', '<', '>=', '<='
    둘 중 하나라도 NaN이면 False 반환 (안전 우선)
    """
    if not (is_valid(a) and is_valid(b)):
        return False
    ops = {
        ">":  a > b,
        "<":  a < b,
        ">=": a >= b,
        "<=": a <= b,
    }
    return ops.get(operator, False)


def crossover(arr_fast: np.ndarray, arr_slow: np.ndarray, idx: int) -> bool:
    """
    골든크로스 감지: fast가 slow를 위로 돌파 (idx-1에선 아래, idx에선 위)
    """
    if idx < 1:
        return False
    prev_above = safe_get(arr_fast, idx - 1) > safe_get(arr_slow, idx - 1)
    curr_above = safe_get(arr_fast, idx) > safe_get(arr_slow, idx)
    return (not prev_above) and curr_above


def crossunder(arr_fast: np.ndarray, arr_slow: np.ndarray, idx: int) -> bool:
    """
    데드크로스 감지: fast가 slow를 아래로 돌파 (idx-1에선 위, idx에선 아래)
    """
    if idx < 1:
        return False
    prev_above = safe_get(arr_fast, idx - 1) > safe_get(arr_slow, idx - 1)
    curr_above = safe_get(arr_fast, idx) > safe_get(arr_slow, idx)
    return prev_above and (not curr_above)
