import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.dashboard.dashboard_data import load_dashboard_data


def fmt_rupees(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"Rs {value/1_000_000:+,.2f}M"
    elif abs(value) >= 1_000:
        return f"Rs {value/1_000:+,.1f}K"
    else:
        return f"Rs {value:+,.0f}"


st.set_page_config(page_title="SME Cashflow Forecasting", layout="wide")

st.title("SME Cashflow Forecasting Dashboard")
st.markdown(
    "LightGBM model trained on 71 engineered features, evaluated with walk-forward CV. "
    "SHAP-based explanation via Lundberg and Lee (2017)."
)
st.divider()

with st.spinner("Loading model and generating SHAP explanations..."):
    data = load_dashboard_data()

with st.sidebar:
    st.header("Model Details")
    st.metric("Target", "Payment-Projected")
    st.metric("Model", "LightGBM")
    st.metric("Features", len(data['feature_names']))
    st.metric("Mean MASE", f"{data['mean_mase']:.3f}")
    st.metric("Weeks Evaluated", len(data['week_dates']))
    st.divider()
    st.caption("Per-fold MASE:")
    for i, m in enumerate(data['per_fold_mase'], 1):
        st.text(f"Fold {i}: {m:.3f}")

st.subheader("Cashflow Forecast: Actual vs Predicted")

mask = ~np.isnan(data['y_predicted'])
plot_dates = [data['week_dates'][i] for i in range(len(mask)) if mask[i]]
plot_actual = data['y_actual'][mask]
plot_pred = data['y_predicted'][mask]

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=plot_dates,
    y=plot_actual,
    mode='lines',
    name='Actual',
    line=dict(color='#333', width=2),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{customdata}<extra></extra>',
    customdata=[fmt_rupees(v) for v in plot_actual],
))

fig_forecast.add_trace(go.Scatter(
    x=plot_dates,
    y=plot_pred,
    mode='lines',
    name='Predicted',
    line=dict(color='#E91E63', width=2),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: %{customdata}<extra></extra>',
    customdata=[fmt_rupees(v) for v in plot_pred],
))

fig_forecast.add_hline(y=0, line=dict(color='#999', width=1))

fig_forecast.update_layout(
    xaxis_title='Week ending',
    yaxis_title='Weekly cashflow (INR)',
    hovermode='x unified',
    height=450,
    template='plotly_white',
    legend=dict(x=0.85, y=0.98),
    margin=dict(l=60, r=40, t=40, b=60),
)

fig_forecast.update_yaxes(
    tickprefix='Rs ',
    tickformat=',.0f',
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.caption(f"{len(plot_dates)} weeks. Mean MASE = {data['mean_mase']:.3f}.")

st.divider()

st.subheader("SHAP Explanation: Largest-Error Week (Illustrative Example)")
st.markdown(
    "The following breakdown explains the prediction for the week where the "
    "model's forecast diverged most from actual cashflow. This serves as an "
    "illustrative example of the SHAP-based interpretation the dashboard provides."
)

predicted_indices = np.where(mask)[0]
errors = np.abs(plot_actual - plot_pred)
default_pos = int(errors.argmax())
row_idx = int(predicted_indices[default_pos])

selected_date = data['week_dates'][row_idx]
actual_val = float(data['y_actual'][row_idx])
pred_val = float(data['y_predicted'][row_idx])
error = pred_val - actual_val
error_pct = 100 * abs(error) / max(abs(actual_val), 1)

st.markdown(f"### Week ending {selected_date.strftime('%Y-%m-%d')}")

c1, c2, c3 = st.columns(3)
c1.metric("Actual", f"Rs {actual_val/1000:+,.0f}K")
c2.metric("Predicted", f"Rs {pred_val/1000:+,.0f}K")
c3.metric("Error", f"Rs {error/1000:+,.0f}K", delta=f"{error_pct:.1f}% of actual", delta_color="off")

st.markdown("#### Feature Contributions")
st.caption("Hover over any bar for exact contribution. Pink = increases prediction. Blue = decreases prediction.")

shap_values_this_week = data['shap_matrix'][row_idx, :]
feature_values_this_week = data['X_values'][row_idx, :]

contribs = pd.DataFrame({
    'feature': data['feature_names'],
    'value': feature_values_this_week,
    'shap': shap_values_this_week,
})
contribs['abs_shap'] = contribs['shap'].abs()
top = contribs.nlargest(8, 'abs_shap').sort_values('shap')
rest_sum = float(contribs.drop(top.index)['shap'].sum())
n_rest = len(contribs) - len(top)

base = data['shap_base']
prediction = base + float(contribs['shap'].sum())

# Build Plotly bar chart data
labels = [f"{row.feature} = {row.value:,.0f}" for row in top.itertuples()]
values = [float(v) for v in top['shap'].values]

if n_rest > 0:
    labels.append(f"{n_rest} other features")
    values.append(rest_sum)

colors = ['#2196F3' if v < 0 else '#E91E63' for v in values]
hover_text = [fmt_rupees(v) for v in values]

fig_shap = go.Figure()

fig_shap.add_trace(go.Bar(
    x=values,
    y=labels,
    orientation='h',
    marker=dict(color=colors),
    text=hover_text,
    textposition='auto',
    hovertemplate='<b>%{y}</b><br>Contribution: %{text}<extra></extra>',
))

fig_shap.add_vline(x=0, line=dict(color='#666', width=1))

fig_shap.update_layout(
    xaxis_title='Feature contribution to prediction (INR)',
    title=f'Baseline {fmt_rupees(base)} to Prediction {fmt_rupees(prediction)}',
    height=500,
    template='plotly_white',
    showlegend=False,
    margin=dict(l=250, r=100, t=60, b=60),
)

fig_shap.update_xaxes(
    tickprefix='Rs ',
    tickformat=',.0f',
)

st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("#### Plain-English Explanation")
st.markdown(
    "The following explains the model's prediction in plain language for a business audience, "
    "translating the SHAP-derived feature contributions into interpretable narrative statements."
)

top5 = contribs.nlargest(5, 'abs_shap').sort_values('abs_shap', ascending=False)

prediction_str = f"Rs {prediction/1000:+,.0f}K"
baseline_str = f"Rs {base/1000:+,.0f}K"

if prediction > 0:
    direction_prediction = "positive (surplus)"
elif prediction < 0:
    direction_prediction = "negative (deficit)"
else:
    direction_prediction = "roughly break-even"

narrative_parts = [
    f"For the week ending {selected_date.strftime('%d %B %Y')}, the model predicts a "
    f"weekly cashflow of **{prediction_str}**, which is a {direction_prediction} position.",
    "",
    f"Starting from the baseline (average predicted cashflow of **{baseline_str}**), "
    "the following features most contributed to this specific prediction:",
    "",
]

feature_context = {
    'sales_roll26w_mean': ('the average sales over the past 26 weeks (6-month sales trend)',
                            'higher sustained sales', 'lower sustained sales'),
    'sales_roll26w_std':  ('the volatility of sales over the past 26 weeks',
                            'more volatile sales', 'less volatile sales'),
    'sales_roll12w_mean': ('the average sales over the past 12 weeks (3-month sales trend)',
                            'higher recent sales', 'lower recent sales'),
    'sales_roll4w_mean':  ('the average sales over the past 4 weeks (recent sales trend)',
                            'higher recent sales', 'lower recent sales'),
    'net_cashflow_roll12w_std':  ('the volatility of net cashflow over the past 12 weeks',
                                    'more volatile past cashflow', 'less volatile past cashflow'),
    'net_cashflow_roll4w_mean':  ('the average net cashflow over the past 4 weeks',
                                    'stronger recent cashflow', 'weaker recent cashflow'),
    'net_cashflow_roll12w_mean': ('the average net cashflow over the past 12 weeks',
                                    'stronger past cashflow', 'weaker past cashflow'),
    'net_cashflow_lag_26': ('the net cashflow from 26 weeks ago (6-month seasonal reference)',
                             'a strong prior period', 'a weak prior period'),
    'net_cashflow_lag_52': ('the net cashflow from 52 weeks ago (annual seasonal reference)',
                             'a strong same-week-last-year', 'a weak same-week-last-year'),
    'purchases_lag_4':  ('purchases 4 weeks ago', 'higher recent purchases', 'lower recent purchases'),
    'purchases_lag_8':  ('purchases 8 weeks ago', 'higher earlier purchases', 'lower earlier purchases'),
    'purchases_lag_12': ('purchases 12 weeks ago (a quarter ago)',
                          'higher quarterly purchases', 'lower quarterly purchases'),
    'sales_paywindow_30d': ('sales collected in a 30-day payment window',
                              'stronger short-term collections', 'weaker short-term collections'),
    'sales_paywindow_60d': ('sales collected in a 60-day payment window',
                              'stronger mid-term collections', 'weaker mid-term collections'),
    'sales_paywindow_90d': ('sales collected in a 90-day payment window',
                              'stronger longer-term collections', 'weaker longer-term collections'),
    'net_expected_45d':   ('the net expected cashflow over a 45-day window',
                             'stronger expected net position', 'weaker expected net position'),
    'sales_expected_45d': ('the sales expected to be paid within 45 days',
                             'stronger expected sales collection', 'weaker expected sales collection'),
    'purchases_expected_45d': ('the purchases expected to be paid within 45 days',
                                'higher expected outflow', 'lower expected outflow'),
}

def describe_feature(feature_name, feature_value, shap_value):
    if feature_name in feature_context:
        desc, high_effect, low_effect = feature_context[feature_name]
        effect_direction = high_effect if feature_value >= 0 else low_effect
        return f"**{desc}** (value: {feature_value:,.0f}), indicating {effect_direction}"
    else:
        return f"**{feature_name}** (value: {feature_value:,.0f})"

for _, row in top5.iterrows():
    shap_str = f"Rs {row['shap']/1000:+,.1f}K"
    direction_word = "increases" if row['shap'] > 0 else "decreases"
    descriptor = describe_feature(row['feature'], row['value'], row['shap'])
    narrative_parts.append(
        f"- {descriptor}, which **{direction_word}** the prediction by **{shap_str}**"
    )

narrative_parts.extend([
    "",
    f"Combining these effects with the smaller contributions from all other features, "
    f"the model arrives at its final prediction of **{prediction_str}** for this week.",
])

if error > 0:
    diff_direction = "higher than"
else:
    diff_direction = "lower than"
narrative_parts.append(
    f"The actual observed cashflow was **Rs {actual_val/1000:+,.0f}K**, which is "
    f"{abs(error)/1000:,.0f}K **{diff_direction}** the model's prediction. "
    f"This example week has been chosen because it exhibits the largest prediction error "
    f"in the evaluation period, and therefore illustrates the model's behaviour "
    f"under challenging cashflow dynamics."
)

st.markdown("\n".join(narrative_parts))

st.divider()

st.caption(
    "Note: A future refinement to this prototype would enable interactive week selection "
    "for on-demand SHAP explanation of any week in the forecast horizon. "
    "The technical foundation for this feature is in place; enabling it requires "
    "additional work on the Streamlit rerender lifecycle for chart components."
)