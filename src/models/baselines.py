def fit_predict_naive_seasonal(y_train: np.ndarray,
                               forecast_horizon: int,
                               season_length: int = 52) -> np.ndarray:

    y_train = np.asarray(y_train).ravel()

    if len(y_train) < season_length:
        # Not enough data for seasonal — use last observed value
        return np.full(forecast_horizon, y_train[-1])

    forecasts = np.empty(forecast_horizon)
    for h in range(forecast_horizon):
        # Look season_length back from the corresponding future point
        source_idx = len(y_train) - season_length + h
        if source_idx >= len(y_train):
            # Wrap around if forecast horizon exceeds one season
            source_idx = source_idx - season_length
        forecasts[h] = y_train[source_idx]
    return forecasts


def fit_predict_arima(y_train: np.ndarray,
                      forecast_horizon: int,
                      order: Tuple[int, int, int] = (1, 1, 1),
                      seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                      ) -> np.ndarray:

    from statsmodels.tsa.arima.model import ARIMA

    y_train = np.asarray(y_train).ravel()

    kwargs = {'order': order}
    if seasonal_order is not None:
        kwargs['seasonal_order'] = seasonal_order

    model = ARIMA(y_train, **kwargs)
    fitted = model.fit()
    forecasts = fitted.forecast(steps=forecast_horizon)

    return np.asarray(forecasts).ravel()