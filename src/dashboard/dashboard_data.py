from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_targets
from src.features import build_all_features
from src.models.tree_models import fit_predict_lightgbm
from src.validation import walk_forward_splits, compute_mase
from src.interpretability.shap_analysis import (
    train_final_lightgbm,
    compute_shap_values,
)


@st.cache_data
def load_dashboard_data() -> Dict:
    target = load_targets()['projected']
    features = build_all_features(target).dropna()

    base_columns = ['sales', 'purchases', 'net_cashflow']
    X_full = features.drop(columns=base_columns)
    y_actual = features['net_cashflow'].values

    y_predicted = np.full_like(y_actual, np.nan, dtype=float)
    mase_per_fold = []

    for train_pos, test_pos in walk_forward_splits(features):
        train = features.iloc[train_pos]
        test = features.iloc[test_pos]

        y_train = train['net_cashflow'].values
        y_test = test['net_cashflow'].values

        X_train = train.drop(columns=base_columns)
        X_test = test.drop(columns=base_columns)

        y_pred_fold = fit_predict_lightgbm(X_train, y_train, X_test)
        y_predicted[test_pos] = y_pred_fold
        mase_per_fold.append(compute_mase(y_test, y_pred_fold, y_train))

    final_model = train_final_lightgbm(X_full, y_actual)
    shap_values = compute_shap_values(final_model, X_full)

    shap_matrix = np.asarray(shap_values.values)
    shap_base = float(shap_values.base_values[0])
    feature_names = list(X_full.columns)
    week_dates_list = features.index.tolist()

    return {
        'X_values':      X_full.values,
        'feature_names': feature_names,
        'y_actual':      y_actual,
        'y_predicted':   y_predicted,
        'week_dates':    week_dates_list,
        'shap_matrix':   shap_matrix,
        'shap_base':     shap_base,
        'mean_mase':     float(np.mean(mase_per_fold)),
        'per_fold_mase': mase_per_fold,
    }