from typing import Optional
import numpy as np
import pandas as pd

# Silence Prophet's verbose logging on every fit
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet


def fit_predict_prophet(X_train: pd.DataFrame,
                        y_train: np.ndarray,
                        X_test: pd.DataFrame,
                        params: Optional[dict] = None,
                        ) -> np.ndarray:

    default_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': False,     # data is already weekly
        'daily_seasonality': False,      # data is not daily
        'seasonality_mode': 'additive',
        'interval_width': 0.80,
    }
    if params is not None:
        default_params.update(params)

    # Build Prophet's required DataFrame format
    train_df = pd.DataFrame({
        'ds': X_train.index,
        'y':  y_train,
    })

    model = Prophet(**default_params)
    model.fit(train_df)

    # Build the future DataFrame
    future_df = pd.DataFrame({'ds': X_test.index})
    forecast = model.predict(future_df)

    return forecast['yhat'].values