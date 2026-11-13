import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="muted")

output_dir = r"C:\Users\user\.gemini\antigravity-cli\brain\532b86ad-6119-4f18-bf09-90b497751ef7"
workspace_plots = r"C:\Users\user\Desktop\2026data\plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(workspace_plots, exist_ok=True)

# 1. Load Datasets
df_raw = pd.read_csv("boston.csv")
df_clean = pd.read_csv("boston_clean.csv")

# Feature sets
# For raw: use all raw features
X_raw = df_raw.drop(columns=['MV'])
y_raw = df_raw['MV']

# For clean: replace CRIM and DIS with their log versions
X_clean = df_clean.drop(columns=['MV', 'CRIM', 'DIS'])
y_clean = df_clean['MV']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear Regression": Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
    "Ridge Regression": Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)
}

scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

results = []

for name, model in models.items():
    # Evaluate on Raw
    cv_raw = cross_validate(model, X_raw, y_raw, cv=kf, scoring=scoring)
    # Evaluate on Clean
    cv_clean = cross_validate(model, X_clean, y_clean, cv=kf, scoring=scoring)
    
    results.append({
        'Model': name,
        'Raw_R2': cv_raw['test_r2'].mean(),
        'Clean_R2': cv_clean['test_r2'].mean(),
        'Raw_RMSE': -cv_raw['test_neg_root_mean_squared_error'].mean(),
        'Clean_RMSE': -cv_clean['test_neg_root_mean_squared_error'].mean(),
        'Raw_MAE': -cv_raw['test_neg_mean_absolute_error'].mean(),
        'Clean_MAE': -cv_clean['test_neg_mean_absolute_error'].mean(),
    })

res_df = pd.DataFrame(results)
print("\n=== Model Performance Comparison (Raw vs Clean) ===")
print(res_df.to_string(index=False))
res_df.to_csv("model_performance_comparison.csv", index=False)

# 2. Plotting Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R2 Comparison
x = np.arange(len(res_df))
width = 0.35

axes[0].bar(x - width/2, res_df['Raw_R2'], width, label='Raw Data (N=506)', color='#90caf9')
axes[0].bar(x + width/2, res_df['Clean_R2'], width, label='Cleaned Data (N=490)', color='#1e88e5')
axes[0].set_ylabel('R² Score (Higher is better)')
axes[0].set_title('Cross-Validated R² Score Improvement', fontweight='bold', fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels(res_df['Model'], rotation=15)
axes[0].set_ylim(0.5, 0.95)
axes[0].legend()

for i in range(len(res_df)):
    axes[0].text(i - width/2, res_df['Raw_R2'][i] + 0.01, f"{res_df['Raw_R2'][i]:.2f}", ha='center', fontsize=9)
    axes[0].text(i + width/2, res_df['Clean_R2'][i] + 0.01, f"{res_df['Clean_R2'][i]:.2f}", ha='center', fontsize=9, fontweight='bold')

# RMSE Comparison
axes[1].bar(x - width/2, res_df['Raw_RMSE'], width, label='Raw Data (N=506)', color='#ef9a9a')
axes[1].bar(x + width/2, res_df['Clean_RMSE'], width, label='Cleaned Data (N=490)', color='#e53935')
axes[1].set_ylabel('RMSE in $1,000s (Lower is better)')
axes[1].set_title('Cross-Validated RMSE Reduction', fontweight='bold', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(res_df['Model'], rotation=15)
axes[1].legend()

for i in range(len(res_df)):
    axes[1].text(i - width/2, res_df['Raw_RMSE'][i] + 0.08, f"{res_df['Raw_RMSE'][i]:.2f}", ha='center', fontsize=9)
    axes[1].text(i + width/2, res_df['Clean_RMSE'][i] + 0.08, f"{res_df['Clean_RMSE'][i]:.2f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "model_comparison_bars.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "model_comparison_bars.png"), dpi=200)
plt.close(fig)

# 3. Residual Diagnostics Plot (Random Forest Raw vs Clean)
from sklearn.model_selection import cross_val_predict

rf_raw = RandomForestRegressor(n_estimators=100, random_state=42)
rf_clean = RandomForestRegressor(n_estimators=100, random_state=42)

preds_raw = cross_val_predict(rf_raw, X_raw, y_raw, cv=kf)
residuals_raw = y_raw - preds_raw

preds_clean = cross_val_predict(rf_clean, X_clean, y_clean, cv=kf)
residuals_clean = y_clean - preds_clean

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(preds_raw, residuals_raw, alpha=0.5, color='#d32f2f')
axes[0].axhline(0, color='black', linestyle='--')
axes[0].set_title('Raw Data Residuals (Severe Underprediction at MV=50)', fontweight='bold')
axes[0].set_xlabel('Predicted Value ($1000s)')
axes[0].set_ylabel('Residual (Actual - Predicted)')

axes[1].scatter(preds_clean, residuals_clean, alpha=0.5, color='#2e7d32')
axes[1].axhline(0, color='black', linestyle='--')
axes[1].set_title('Cleaned Data Residuals (Balanced Homoscedasticity)', fontweight='bold')
axes[1].set_xlabel('Predicted Value ($1000s)')
axes[1].set_ylabel('Residual (Actual - Predicted)')

plt.suptitle("Residual Diagnostics: Random Forest", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "residuals_diagnostics.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "residuals_diagnostics.png"), dpi=200)
plt.close(fig)

# 4. Feature Importance Plot
rf_clean.fit(X_clean, y_clean)
importances = pd.Series(rf_clean.feature_importances_, index=X_clean.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', ax=ax, color='#1976d2')
ax.set_title('Random Forest Feature Importance on Cleaned Dataset', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "feature_importance.png"), dpi=200)
plt.close(fig)

print("Model training, cross-validation, and visualization completed!")
