from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from lightgbm import LGBMRegressor

from src.config import RANDOM_SEED


def train_final_lightgbm(X_train: pd.DataFrame,
                         y_train: np.ndarray,
                         params: Optional[dict] = None,
                         ) -> LGBMRegressor:

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
    return model


def compute_shap_values(model: LGBMRegressor,
                        X: pd.DataFrame,
                        ) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def get_feature_importance_ranking(shap_values: shap.Explanation,
                                    top_n: int = 20,
                                    ) -> pd.DataFrame:

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    
    ranking = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    ranking.insert(0, 'rank', range(1, len(ranking) + 1))
    return ranking.head(top_n)


def plot_feature_importance(shap_values: shap.Explanation,
                             top_n: int = 15,
                             save_path: Optional[str] = None,
                             ) -> None:

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_summary(shap_values: shap.Explanation,
                  top_n: int = 15,
                  save_path: Optional[str] = None,
                  ) -> None:

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_waterfall(shap_values: shap.Explanation,
                    sample_idx: int = 0,
                    top_n: int = 10,
                    save_path: Optional[str] = None,
                    ) -> None:

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], max_display=top_n, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_feature_importance_clean(shap_values: shap.Explanation,
                                    top_n: int = 15,
                                    save_path: Optional[str] = None,
                                    ) -> None:

    from matplotlib.ticker import FuncFormatter
    
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    
    ranking = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs,
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(ranking['feature'], ranking['importance'], 
                    color='#E91E63', edgecolor='none')
    
    for bar, val in zip(bars, ranking['importance']):
        label = f'{val/1000:,.1f}K' if val >= 1000 else f'{val:,.0f}'
        ax.text(val, bar.get_y() + bar.get_height()/2, f' Rs {label}',
                va='center', ha='left', fontsize=9, color='#333')
    
    ax.xaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.set_xlabel('Mean absolute SHAP value (INR contribution)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importance, LightGBM on Payment-Projected Cashflow',
                 fontsize=12, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(0, ranking['importance'].max() * 1.25)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_summary_clean(shap_values: shap.Explanation,
                        top_n: int = 15,
                        save_path: Optional[str] = None,
                        ) -> None:

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    
    plt.figure(figsize=(14, 8))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    
    ax = plt.gca()
    
    # Force only 5 x-axis ticks maximum, spaced widely
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Apply the rupee formatter
    ax.xaxis.set_major_formatter(FuncFormatter(_format_thousands))
    
    # Increase tick label font size for readability
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.set_xlabel('SHAP value (impact on predicted weekly cashflow in INR)', fontsize=11)
    plt.title(f'Feature Effects on Cashflow Predictions, LightGBM (top {top_n} features)',
              fontsize=12, pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()



def plot_waterfall_clean(shap_values: shap.Explanation,
                          X: pd.DataFrame,
                          sample_idx: int = 0,
                          top_n: int = 8,
                          save_path: Optional[str] = None,
                          ) -> None:

    from matplotlib.ticker import FuncFormatter
    
    # Get sample data
    sample_shap = shap_values[sample_idx]
    base_value = float(sample_shap.base_values)
    predicted_value = float(base_value + sample_shap.values.sum())
    
    # Sort features by absolute contribution, take top_n
    contributions = pd.DataFrame({
        'feature': shap_values.feature_names,
        'value': X.iloc[sample_idx].values,
        'shap': sample_shap.values,
    })
    contributions['abs_shap'] = contributions['shap'].abs()
    top_features = contributions.nlargest(top_n, 'abs_shap').copy()
    
    # Aggregate remaining features
    remaining = contributions.drop(top_features.index)
    remaining_sum = remaining['shap'].sum()
    
    # Sample label
    if hasattr(X.index[sample_idx], 'strftime'):
        sample_label = X.index[sample_idx].strftime('%Y-%m-%d')
    else:
        sample_label = f"sample {sample_idx}"
    
    # Build rows: baseline first, then each feature, then other features, then prediction
    rows = [{'label': f'Baseline (average prediction)',
             'shap': None,
             'position': base_value,
             'type': 'baseline'}]
    
    running_total = base_value
    for _, row in top_features.iterrows():
        running_total += row['shap']
        rows.append({
            'label': f"{row['feature']}\n(value: {row['value']:,.0f})",
            'shap': row['shap'],
            'position': running_total,
            'type': 'contribution',
        })
    
    if len(remaining) > 0:
        running_total += remaining_sum
        rows.append({
            'label': f"{len(remaining)} other features",
            'shap': remaining_sum,
            'position': running_total,
            'type': 'contribution',
        })
    
    rows.append({'label': f'Final prediction',
                 'shap': None,
                 'position': predicted_value,
                 'type': 'prediction'})
    
    # Plot
    fig, ax = plt.subplots(figsize=(13, 8))
    y_positions = list(range(len(rows)))
    y_positions.reverse()
    
    for i, row_data in enumerate(rows):
        y = y_positions[i]
        
        if row_data['type'] in ('baseline', 'prediction'):
            # Reference bar showing absolute position
            color = '#333' if row_data['type'] == 'prediction' else '#666'
            ax.barh(y, row_data['position'], color=color, alpha=0.15, 
                    edgecolor=color, linewidth=1.5, height=0.6)
            value_str = f"Rs {row_data['position']/1000:,.0f}K"
            ax.text(row_data['position'], y, f'  {value_str}',
                    va='center', ha='left', fontsize=10, fontweight='bold', color=color)
        else:
            # Contribution bar
            prev_position = rows[i-1]['position']
            width = abs(row_data['shap'])
            left = min(prev_position, row_data['position'])
            color = '#E91E63' if row_data['shap'] > 0 else '#2196F3'
            
            ax.barh(y, width, left=left, color=color, edgecolor='none', height=0.6)
            
            # Value label
            value_sign = '+' if row_data['shap'] > 0 else ''
            value_str = f"{value_sign}Rs {row_data['shap']/1000:,.1f}K"
            
            # Position label at right edge for positive, left edge for negative
            if row_data['shap'] > 0:
                text_x = row_data['position']
                ha = 'left'
                offset = width * 0.03
            else:
                text_x = row_data['position']
                ha = 'right'
                offset = -width * 0.03
            
            ax.text(text_x + offset, y, value_str,
                    va='center', ha=ha, fontsize=9, fontweight='bold', color='#333')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row_data['label'] for row_data in rows], fontsize=9)
    
    ax.axvline(x=0, color='#ccc', linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.set_xlabel('Predicted weekly cashflow (INR)', fontsize=11)
    ax.set_title(f'Cashflow Prediction Breakdown, Week ending {sample_label}\n'
                 f'From baseline of Rs {base_value/1000:,.0f}K to final prediction of Rs {predicted_value/1000:,.0f}K',
                 fontsize=12, pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Extend x-axis to fit labels
    all_positions = [r['position'] for r in rows]
    x_min = min(all_positions + [0])
    x_max = max(all_positions + [0])
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _format_thousands(x, pos):
    if abs(x) >= 1_000_000:
        return f'Rs {x/1_000_000:.1f}M'
    elif abs(x) >= 1_000:
        return f'Rs {x/1_000:.0f}K'
    else:
        return f'Rs {x:.0f}'

def plot_contribution_chart(shap_values: shap.Explanation,
    X: pd.DataFrame,
    sample_idx: int = 0,
    top_n: int = 10,
    save_path: Optional[str] = None,
    ) -> None:

    from matplotlib.ticker import FuncFormatter
    
    sample_shap = shap_values[sample_idx]
    base_value = float(sample_shap.base_values)
    predicted_value = float(base_value + sample_shap.values.sum())
    
    # Sort features by absolute contribution, take top_n
    contributions = pd.DataFrame({
        'feature': shap_values.feature_names,
        'value': X.iloc[sample_idx].values,
        'shap': sample_shap.values,
    })
    contributions['abs_shap'] = contributions['shap'].abs()
    top_features = contributions.nlargest(top_n, 'abs_shap').sort_values('shap')
    
    # Aggregate remaining features
    remaining = contributions.drop(top_features.index)
    remaining_sum = remaining['shap'].sum()
    
    # Sample label
    if hasattr(X.index[sample_idx], 'strftime'):
        sample_label = X.index[sample_idx].strftime('%Y-%m-%d')
    else:
        sample_label = f"sample {sample_idx}"
    
    # Build the plot
    fig, ax = plt.subplots(figsize=(13, 8))
    
    # Prepare data for plotting
    labels = []
    values = []
    colors = []
    
    for _, row in top_features.iterrows():
        labels.append(f"{row['feature']} (value: {row['value']:,.0f})")
        values.append(row['shap'])
        colors.append('#2196F3' if row['shap'] < 0 else '#E91E63')
    
    if len(remaining) > 0:
        labels.append(f"{len(remaining)} other features (combined)")
        values.append(remaining_sum)
        colors.append('#2196F3' if remaining_sum < 0 else '#E91E63')
    
    y_positions = list(range(len(labels)))
    bars = ax.barh(y_positions, values, color=colors, edgecolor='none', height=0.7)
    
    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        sign = '+' if val >= 0 else ''
        label_text = f"{sign}Rs {val/1000:,.1f}K"
        
        # Position label at bar end
        if val >= 0:
            x_pos = val
            offset = abs(val) * 0.03
            ha = 'left'
        else:
            x_pos = val
            offset = -abs(val) * 0.03
            ha = 'right'
        
        ax.text(x_pos + offset, i, label_text,
                va='center', ha=ha, fontsize=10, fontweight='bold', color='#333')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    
    # Vertical reference line at zero
    ax.axvline(x=0, color='#666', linewidth=1)
    
    ax.xaxis.set_major_formatter(FuncFormatter(_format_thousands))
    ax.set_xlabel('Feature contribution to prediction (INR)', fontsize=11)
    
    # Title with baseline and prediction context
    ax.set_title(
        f'Feature Contributions to Cashflow Prediction, Week ending {sample_label}\n'
        f'Baseline: Rs {base_value/1000:,.0f}K   →   Final Prediction: Rs {predicted_value/1000:,.0f}K   '
        f'(Total contribution: Rs {(predicted_value - base_value)/1000:,.0f}K)',
        fontsize=11, pad=15
    )
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E91E63', label='Increases prediction'),
        Patch(facecolor='#2196F3', label='Decreases prediction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Extend x-axis to fit labels
    max_val = max(abs(v) for v in values)
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()