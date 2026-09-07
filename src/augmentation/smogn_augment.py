import warnings
from typing import Tuple
import numpy as np
import pandas as pd

import smogn


def augment_with_smogn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_ratio: float = 2.0,
    k_neighbors: int = 5,
) -> Tuple[pd.DataFrame, np.ndarray]:

    n_original = len(X_train)
    n_target = int(n_original * target_ratio)
    n_synthetic_wanted = n_target - n_original

    if n_synthetic_wanted <= 0:
        return X_train, y_train

    combined = X_train.reset_index(drop=True).copy()
    target_col = '_smogn_target'
    combined[target_col] = y_train

    try:
        with warnings.catch_warnings():
            # Silence deprecation warnings from smogn on pandas 2.x
            warnings.simplefilter('ignore')
            
            augmented = smogn.smoter(
                data=combined,
                y=target_col,
                k=k_neighbors,
                samp_method='extreme',
            )
    except Exception as e:
        warnings.warn(
            f"SMOGN augmentation failed: {type(e).__name__}: {e}. "
            f"Returning original training data unchanged."
        )
        return X_train, y_train

    y_augmented = augmented[target_col].values
    X_augmented = augmented.drop(columns=[target_col])

    # Preserve original column order from X_train
    X_augmented = X_augmented[X_train.columns]

    return X_augmented, y_augmented