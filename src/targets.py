import re
from typing import Tuple, Optional
import pandas as pd

from src.config import (
    BANKING_DELAY_DAYS,
    DEFAULT_TERM_DAYS,
    SIZE_PERCENTILE,
    PURCHASE_SIZE_THRESHOLD,
    BASE_SERIES,
)


# --- Accrual target ---------------------------------------------------

def build_accrual_target(sales: pd.DataFrame,
                         purchases: pd.DataFrame) -> pd.DataFrame:

    weekly_sales = (
        sales.set_index('Date')['Gross Total']
        .resample('W').sum()
        .rename('sales')
    )
    weekly_purchases = (
        purchases.set_index('Date')['Gross Total']
        .resample('W').sum()
        .rename('purchases')
    )

    cashflow = pd.concat([weekly_sales, weekly_purchases], axis=1).fillna(0)
    cashflow['net_cashflow'] = cashflow['sales'] - cashflow['purchases']
    return cashflow[BASE_SERIES]


# --- Payment term parser --------------------------------------------

def parse_terms(term_str: str) -> Tuple[int, int]:

    term_str = term_str.strip()
    match_range = re.fullmatch(r'(\d+)\s*-\s*(\d+)', term_str)
    match_single = re.fullmatch(r'(\d+)', term_str)

    if match_range:
        return int(match_range.group(1)), int(match_range.group(2))
    elif match_single:
        n = int(match_single.group(1))
        return n, n
    else:
        raise ValueError(f"Unrecognised term format: '{term_str}'")


def project_to_payment_day(term_str: str,
                           invoice_amount: float,
                           size_threshold: float | int ) -> int:

    start, end = parse_terms(term_str)

    if start == end:
        return start + BANKING_DELAY_DAYS

    if invoice_amount >= size_threshold:
        return end + BANKING_DELAY_DAYS
    return start + BANKING_DELAY_DAYS


def day_to_week_offset(day: int) -> int:

    if day == 0:
        return 0
    return (day - 1) // 7 + 1

    # --- Payment-projected target --------------------------------------

def compute_per_customer_thresholds(sales: pd.DataFrame,
                                    percentile: float = SIZE_PERCENTILE
                                    ) -> pd.Series:
    return (
        sales.groupby('GSTIN/UIN')['Gross Total']
        .quantile(percentile)
        .rename('customer_threshold')
    )


def build_payment_projected_target(
    sales: pd.DataFrame,
    purchases: pd.DataFrame,
    buyer_terms: pd.DataFrame,
    supplier_terms: pd.DataFrame,
    per_customer_thresholds: Optional[pd.Series] = None,
) -> pd.DataFrame:

    # If thresholds not provided, compute them now
    if per_customer_thresholds is None:
        per_customer_thresholds = compute_per_customer_thresholds(sales)

    # Merge terms onto invoices via pseudonymised GSTIN
    sales_with_terms = sales.merge(
        buyer_terms[['GSTIN', 'Payment Terms (days)']],
        left_on='GSTIN/UIN', right_on='GSTIN', how='left',
    ).drop(columns=['GSTIN'])

    purchases_with_terms = purchases.merge(
        supplier_terms[['GSTIN', 'Payment Terms (days)']],
        left_on='GSTIN/UIN', right_on='GSTIN', how='left',
    ).drop(columns=['GSTIN'])

    # Apply default term to any unmatched invoices
    default = str(DEFAULT_TERM_DAYS)
    sales_with_terms['Payment Terms (days)'] = (
        sales_with_terms['Payment Terms (days)'].fillna(default)
    )
    purchases_with_terms['Payment Terms (days)'] = (
        purchases_with_terms['Payment Terms (days)'].fillna(default)
    )

    # Attach per-customer threshold to each sales row
    thresholds_df = per_customer_thresholds.reset_index()
    sales_with_terms = sales_with_terms.merge(
        thresholds_df, on='GSTIN/UIN', how='left',
    )

    # Project each sales invoice to its payment date
    def _project_sales(row: pd.Series) -> pd.Timestamp:
        threshold = row['customer_threshold']
        if pd.isna(threshold):
            threshold = float('inf')  # invoice will always be classed 'small'
        day_offset = project_to_payment_day(
            row['Payment Terms (days)'],
            row['Gross Total'],
            threshold,
        )
        week_offset = day_to_week_offset(day_offset)
        return row['Date'] + pd.Timedelta(weeks=week_offset)

    sales_with_terms['projected_payment_date'] = (
        sales_with_terms.apply(_project_sales, axis=1)
    )

    # Project each purchase invoice (no per-supplier thresholds, so use a fixed placeholder threshold that never triggers size-conditioning)

    def _project_purchases(row: pd.Series) -> pd.Timestamp:
        day_offset = project_to_payment_day(
            row['Payment Terms (days)'],
            row['Gross Total'],
            size_threshold=PURCHASE_SIZE_THRESHOLD,
        )
        week_offset = day_to_week_offset(day_offset)
        return row['Date'] + pd.Timedelta(weeks=week_offset)

    purchases_with_terms['projected_payment_date'] = (
        purchases_with_terms.apply(_project_purchases, axis=1)
    )

    # Aggregate to weekly totals on the projected date
    weekly_sales = (
        sales_with_terms.set_index('projected_payment_date')['Gross Total']
        .resample('W').sum()
        .rename('sales')
    )
    weekly_purchases = (
        purchases_with_terms.set_index('projected_payment_date')['Gross Total']
        .resample('W').sum()
        .rename('purchases')
    )

    cashflow = pd.concat([weekly_sales, weekly_purchases], axis=1).fillna(0)
    cashflow['net_cashflow'] = cashflow['sales'] - cashflow['purchases']
    return cashflow[BASE_SERIES]