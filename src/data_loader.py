from typing import Dict
import pandas as pd

from src.config import (
    SALES_CLEAN_PATH,
    PURCHASE_CLEAN_PATH,
    CASHFLOW_ACCRUAL_PATH,
    CASHFLOW_PROJECTED_PATH,
)

def load_invoices() -> Dict[str, pd.DataFrame]:
    sales = pd.read_parquet(SALES_CLEAN_PATH)
    purchases = pd.read_parquet(PURCHASE_CLEAN_PATH)
    return {"sales": sales, "purchases": purchases}


def load_targets() -> Dict[str, pd.DataFrame]:
    accrual = pd.read_parquet(CASHFLOW_ACCRUAL_PATH)
    projected = pd.read_parquet(CASHFLOW_PROJECTED_PATH)
    return {"accrual": accrual, "projected": projected}