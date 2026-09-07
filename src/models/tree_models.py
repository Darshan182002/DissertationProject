from typing import Optional
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import RANDOM_SEED


def fit_predict_xgboost(X_train: pd.DataFrame,
                        y_train: np.ndarray,
                        X_test: pd.DataFrame,
                        params: Optional[dict] = None,
                        ) -> np.ndarray:

    default_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'reg_alpha': 0.1,          # L1 regularisation
        'reg_lambda': 1.0,         # L2 regularisation
        'random_state': RANDOM_SEED,
        'verbosity': 0,
    }
    if params is not None:
        default_params.update(params)

    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_lightgbm(X_train: pd.DataFrame,
                         y_train: np.ndarray,
                         X_test: pd.DataFrame,
                         params: Optional[dict] = None,
                         ) -> np.ndarray:

    default_params = {
        'n_estimators': 200,
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.05,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'verbosity': -1,
    }
    if params is not None:
        default_params.update(params)

    model = LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model.predict(X_test)