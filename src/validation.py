from typing import Generator, Tuple
import numpy as np
import pandas as pd

from src.config import WALK_FORWARD_FOLDS, MIN_TRAIN_WEEKS


def walk_forward_splits(target: pd.DataFrame,
                        n_folds: int = WALK_FORWARD_FOLDS,
                        min_train_weeks: int = MIN_TRAIN_WEEKS
                        ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

    n = len(target)

    if n < min_train_weeks + n_folds:
        raise ValueError(
            f"Series too short for {n_folds} folds with min_train_weeks="
            f"{min_train_weeks}. Need at least {min_train_weeks + n_folds} "
            f"observations, have {n}."
        )

    # Number of test observations per fold: distribute remaining weeks evenly
    test_size = (n - min_train_weeks) // n_folds

    for fold in range(n_folds):
        train_end = min_train_weeks + fold * test_size
        test_start = train_end
        test_end = test_start + test_size

        # For the final fold, extend the test set to the end of the series
        if fold == n_folds - 1:
            test_end = n

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices


def compute_mase(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_train: np.ndarray) -> float:

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_train = np.asarray(y_train).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred lengths differ: {len(y_true)} vs {len(y_pred)}"
        )

    # Forecast error
    forecast_mae = np.mean(np.abs(y_true - y_pred))

    # Naive one-step-ahead MAE on training data
    naive_mae = np.mean(np.abs(np.diff(y_train)))

    if naive_mae == 0:
        raise ValueError(
            "Naive MAE is zero (training series is constant). "
            "MASE is undefined for degenerate series."
        )

    return forecast_mae / naive_mae