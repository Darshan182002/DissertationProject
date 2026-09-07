from typing import Callable, Dict, Sequence
import numpy as np
import pandas as pd

from src.validation import walk_forward_splits, compute_mase
from src.config import TARGET_COLUMN

from typing import Callable, Dict, Sequence, Optional


ModelFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def run_walk_forward_evaluation(
    model_fn: ModelFn,
    features: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    n_folds: int = 5,
    min_train_weeks: int = 52,
    augment_fn: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:

    # Drop rows with any NaN in features (warm-up period)
    clean = features.dropna().copy()

    mase_per_fold = []
    n_train_list = []
    n_test_list = []

    for fold_idx, (train_pos, test_pos) in enumerate(
        walk_forward_splits(clean, n_folds=n_folds, min_train_weeks=min_train_weeks)
    ):
        train = clean.iloc[train_pos]
        test = clean.iloc[test_pos]

        y_train = train[target_column].values
        y_test = test[target_column].values

        # Drop all three base series columns to prevent leakage.
        base_columns = ['sales', 'purchases', 'net_cashflow']
        X_train = train.drop(columns=base_columns)
        X_test = test.drop(columns=base_columns)

        # Apply augmentation to training data only (never test)
        if augment_fn is not None:
            X_train, y_train = augment_fn(X_train, y_train)

        y_pred = model_fn(X_train, y_train, X_test)

        mase = compute_mase(y_test, y_pred, y_train)
        mase_per_fold.append(mase)
        n_train_list.append(len(train))
        n_test_list.append(len(test))

    mase_per_fold = np.array(mase_per_fold)
    return {
        'mase_per_fold':    mase_per_fold,
        'mean_mase':        float(mase_per_fold.mean()),
        'std_mase':         float(mase_per_fold.std()),
        'n_train_per_fold': n_train_list,
        'n_test_per_fold':  n_test_list,
    }