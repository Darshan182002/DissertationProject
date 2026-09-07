# AI-Driven Cashflow Forecasting for Data-Scarce SMEs

Code repository for an MSc Data Science dissertation submitted at the University of Bath, September 2026. The project evaluates whether modern machine learning and data augmentation can help small and medium-sized enterprises forecast weekly cashflow when they have less than three years of transaction history.

## Motivation

SMEs make up 99.8% of UK businesses, but only around 38% survive their first five years, and cashflow is the mechanism through which most of the rest fail. Most forecasting research assumes datasets far larger than a typical SME has. This project examines whether modern methods can close that gap on real invoice data from a manufacturing SME in India.

## Key findings

- ARIMA outperformed XGBoost, LightGBM, Prophet and LSTM on both accrual and payment-projected targets, and beat a naive seasonal benchmark by 33.9%.
- None of the three augmentation methods tested — SMOGN, VAE, and a simplified TimeGAN — consistently improved any baseline. The variational autoencoder degraded performance by 63 to 73 per cent.
- SHAP analysis of LightGBM produced a business-interpretable feature hierarchy dominated by sales rolling aggregates, cashflow volatility measures, and engineered payment-cycle features. A Streamlit dashboard translates these attributions into plain-English explanations for non-technical users.

Full results are reported in the dissertation.

## Repository structure

DissertationProject/
├── notebooks/
│ ├── 00_smoke_test.ipynb
│ ├── 01_data_loading_profiling.ipynb
│ └── 02_feature_engineering.ipynb
├── src/
│ ├── augmentation/ # SMOGN, VAE, TimeGAN implementations
│ ├── dashboard/ # Streamlit prototype
│ ├── interpretability/ # SHAP analysis
│ ├── models/ # ARIMA, Prophet, LSTM, tree models
│ ├── config.py # Paths and hyperparameters
│ ├── data_loader.py # Ingestion and pseudonymisation
│ ├── features.py # 69-feature engineering pipeline
│ ├── targets.py # Dual target construction
│ └── validation.py # Walk-forward evaluation
├── reports/
│ └── figures/ # Figures used in the dissertation
├── run_dashboard.py # Entry point for the Streamlit app
└── LICENSE


## Setup

Requires Python 3.13.

```bash
git clone https://github.com/Darshan182002/DissertationProject.git
cd DissertationProject
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels prophet torch shap streamlit smogn
```

Launch the dashboard prototype:

```bash
streamlit run run_dashboard.py
```

## Data

The underlying transaction data is not included in this repository. The work was carried out under a formal non-disclosure agreement with the participating SME, and no identifying information about the company, its buyers, or its suppliers appears anywhere in the code, notebooks, or outputs. All party names and GSTIN identifiers referenced by the pipeline are pseudonymised using deterministic SHA-256 hashing with a project-specific salt, and the salt is never committed to version control.

The code is published to demonstrate the pipeline and analytical approach. Reproducing the results on other data requires supplying inputs matching the schema described in Chapter 3 of the dissertation.

## Reference

Ramesh, D. (2026) *AI-Driven Cashflow Forecasting for Data-Scarce SMEs*. MSc Dissertation, University of Bath.

## Licence

MIT — see [LICENSE](./LICENSE).