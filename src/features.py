from typing import List, Sequence
import pandas as pd

from src.config import BASE_SERIES


# --- Calendar features ------------------------------------------------

def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out['year'] = idx.year
    out['quarter'] = idx.quarter
    out['month'] = idx.month
    out['week_of_year'] = idx.isocalendar().week.astype(int)
    out['week_of_month'] = ((idx.day - 1) // 7 + 1).astype(int)
    out['is_quarter_end'] = idx.is_quarter_end.astype(int)
    return out


# --- Lag features ----------------------------------------------------

def build_lag_features(df: pd.DataFrame,
                       series: Sequence[str] = None,
                       lags: Sequence[int] = None) -> pd.DataFrame:

    if series is None:
        series = BASE_SERIES
    if lags is None:
        lags = (1, 2, 4, 8, 12, 26, 52)

    out = df.copy()
    for col in series:
        for n in lags:
            out[f'{col}_lag_{n}'] = out[col].shift(n)
    return out

# --- Rolling aggregate features --------------------------------------

def build_rolling_features(df: pd.DataFrame,
                           series: Sequence[str] = None,
                           windows: Sequence[int] = None) -> pd.DataFrame:

    if series is None:
        series = BASE_SERIES
    if windows is None:
        windows = (4, 12, 26)

    out = df.copy()
    for col in series:
        shifted = out[col].shift(1)
        for w in windows:
            out[f'{col}_roll{w}w_mean'] = shifted.rolling(w, min_periods=w).mean()
            out[f'{col}_roll{w}w_std'] = shifted.rolling(w, min_periods=w).std()
            out[f'{col}_roll{w}w_sum'] = shifted.rolling(w, min_periods=w).sum()
    return out


# --- Payment-cycle features (domain-specific) -------------------------

def build_payment_cycle_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    # Rolling sums over payment windows (in weeks): 30d=4w, 45d=6w, 60d=8w, 90d=13w, full=12w
    payment_windows = {
        '30d': 4,
        '45d': 6,
        '60d': 8,
        '90d': 13,
        'full': 12,
    }

    for series in ('sales', 'purchases'):
        shifted = out[series].shift(1)
        for label, w in payment_windows.items():
            out[f'{series}_paywindow_{label}'] = shifted.rolling(w, min_periods=w).sum()

    # Expected net position: full-window sales minus full-window purchases
    out['expected_net_position'] = (
        out['sales_paywindow_full'] - out['purchases_paywindow_full']
    )

    # Sales-to-purchase ratio (guard against divide-by-zero)
    denom = out['purchases_paywindow_full'].replace(0, pd.NA)
    out['sales_to_purchase_ratio'] = out['sales_paywindow_full'] / denom

    # Tighter payment-cycle feature: single point-lag at 6 weeks (~45 days)
    out['sales_expected_45d'] = out['sales'].shift(6)
    out['purchases_expected_45d'] = out['purchases'].shift(6)
    out['net_expected_45d'] = (
        out['sales_expected_45d'] - out['purchases_expected_45d']
    )

    return out


# --- Orchestrator -----------------------------------------------------

def build_all_features(target: pd.DataFrame) -> pd.DataFrame:

    df = target.copy()
    df = build_calendar_features(df)
    df = build_lag_features(df)
    df = build_rolling_features(df)
    df = build_payment_cycle_features(df)
    return df