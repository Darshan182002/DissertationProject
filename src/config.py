from pathlib import Path

# --- Project root ---
# All paths below are absolute, derived from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data directories ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRIVATE_DATA_DIR = DATA_DIR / "private"

# --- Output directories ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Key data files ---
SALES_CLEAN_PATH = PROCESSED_DATA_DIR / "sales_clean.parquet"
PURCHASE_CLEAN_PATH = PROCESSED_DATA_DIR / "purchase_clean.parquet"
CASHFLOW_ACCRUAL_PATH = PROCESSED_DATA_DIR / "weekly_cashflow.parquet"
CASHFLOW_PROJECTED_PATH = PROCESSED_DATA_DIR / "weekly_cashflow_projected.parquet"

# --- Private data files (gitignored) ---
GSTIN_MAPPING_PATH = PRIVATE_DATA_DIR / "gstin_mapping.json"
PARTY_MAPPING_PATH = PRIVATE_DATA_DIR / "party_mapping.json"
TERMS_FILE_PATH = PRIVATE_DATA_DIR / "customer_terms_template.xlsx"

# --- Payment projection constants ---
BANKING_DELAY_DAYS = 2       # Days added to every non-immediate payment term
DEFAULT_TERM_DAYS = 21       # Fallback term for unmapped suppliers (weighted avg)
SIZE_PERCENTILE = 0.75       # Customer-relative "large invoice" threshold
PURCHASE_SIZE_THRESHOLD = 12_000    # Fixed threshold for purchase invoices; below this, SME pays immediately

# --- Target columns ---
TARGET_COLUMN = "net_cashflow"
BASE_SERIES = ["sales", "purchases", "net_cashflow"]

# --- Reproducibility ---
RANDOM_SEED = 42             # Applied wherever randomness is introduced

# --- Validation ---
WALK_FORWARD_FOLDS = 5       # Number of expanding-window folds for validation
MIN_TRAIN_WEEKS = 52         # Minimum training window (1 year) before first test fold