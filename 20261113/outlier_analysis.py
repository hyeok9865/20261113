import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="muted")

output_dir = r"C:\Users\user\.gemini\antigravity-cli\brain\532b86ad-6119-4f18-bf09-90b497751ef7"
workspace_plots = r"C:\Users\user\Desktop\2026data\plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(workspace_plots, exist_ok=True)

df = pd.read_csv(r"C:\Users\user\Desktop\2026data\boston.csv")
print("Data Shape:", df.shape)

stats_list = []
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    extreme_lower = q1 - 3.0 * iqr
    extreme_upper = q3 + 3.0 * iqr
    
    iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    extreme_outliers = df[(df[col] < extreme_lower) | (df[col] > extreme_upper)].shape[0]
    
    mean = df[col].mean()
    std = df[col].std()
    z_scores = (df[col] - mean) / std
    z_outliers = (np.abs(z_scores) > 3).sum()
    
    skew = df[col].skew()
    
    stats_list.append({
        'Feature': col,
        'Min': round(df[col].min(), 2),
        'Q1': round(q1, 2),
        'Median': round(df[col].median(), 2),
        'Q3': round(q3, 2),
        'Max': round(df[col].max(), 2),
        'Skewness': round(skew, 2),
        'IQR_Outliers': iqr_outliers,
        'IQR_Ratio(%)': round(iqr_outliers / len(df) * 100, 1),
        'Extreme_Outliers': extreme_outliers,
        'Z_Outliers(|z|>3)': z_outliers
    })

outlier_summary = pd.DataFrame(stats_list)
print("\n=== Outlier Summary Table ===")
print(outlier_summary.to_string(index=False))

outlier_summary.to_csv(r"C:\Users\user\Desktop\2026data\outlier_summary.csv", index=False)

# 1. Boxplots of all features
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for idx, col in enumerate(df.columns):
    sns.boxplot(y=df[col], ax=axes[idx], color='skyblue', flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'crimson'})
    axes[idx].set_title(f"{col} (Outliers: {stats_list[idx]['IQR_Outliers']})", fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('')

plt.suptitle("Boxplots of All Features with Outliers (IQR 1.5x fence)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "boxplots_all.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "boxplots_all.png"), dpi=200)
plt.close(fig)

# 2. Detailed inspection of key outlier features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# CRIM
sns.histplot(df['CRIM'], kde=True, ax=axes[0, 0], color='crimson')
axes[0, 0].set_title(f"CRIM (Skewness: {df['CRIM'].skew():.2f}) - Severe Right Tail", fontweight='bold')

sns.histplot(np.log1p(df['CRIM']), kde=True, ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title(f"log1p(CRIM) (Skewness: {np.log1p(df['CRIM']).skew():.2f}) - Normalized", fontweight='bold')

# RM
sns.histplot(df['RM'], kde=True, ax=axes[0, 2], color='seagreen')
axes[0, 2].axvline(df['RM'].quantile(0.25) - 1.5*(df['RM'].quantile(0.75)-df['RM'].quantile(0.25)), color='red', linestyle='--', label='IQR Lower')
axes[0, 2].axvline(df['RM'].quantile(0.75) + 1.5*(df['RM'].quantile(0.75)-df['RM'].quantile(0.25)), color='red', linestyle='--', label='IQR Upper')
axes[0, 2].legend()
axes[0, 2].set_title("RM (Rooms) - Natural Bell Curve with Extreme Mansions/SROs", fontweight='bold')

# DIS
sns.histplot(df['DIS'], kde=True, ax=axes[1, 0], color='goldenrod')
axes[1, 0].set_title(f"DIS (Skewness: {df['DIS'].skew():.2f}) - Employment Distance", fontweight='bold')

# B
sns.histplot(df['B'], kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title(f"B (Skewness: {df['B'].skew():.2f}) - Massive Ceiling Clustering (~396)", fontweight='bold')

# MV
sns.histplot(df['MV'], kde=True, ax=axes[1, 2], color='darkorange')
axes[1, 2].axvline(50.0, color='red', linestyle='--', linewidth=2, label='Censored Cap ($50k)')
axes[1, 2].legend()
axes[1, 2].set_title(f"MV Target (Median Value) - Censored Cap at 50.0", fontweight='bold')

plt.suptitle("Detailed Distributions of High-Outlier Features & Target", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "distributions_detail.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "distributions_detail.png"), dpi=200)
plt.close(fig)

# 3. Scatter plots vs MV with highlight on Cap (MV == 50)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

df_cap = df[df['MV'] >= 50.0]
df_normal = df[df['MV'] < 50.0]

# RM vs MV
axes[0, 0].scatter(df_normal['RM'], df_normal['MV'], color='teal', alpha=0.6, label='Normal (MV < 50)')
axes[0, 0].scatter(df_cap['RM'], df_cap['MV'], color='red', s=50, marker='^', label='Censored Cap (MV = 50)')
axes[0, 0].set_xlabel('RM (Average Number of Rooms)')
axes[0, 0].set_ylabel('MV (Price in $1000s)')
axes[0, 0].set_title('RM vs MV: Linear Trend vs Capped Top Prices', fontweight='bold')
axes[0, 0].legend()

# CRIM vs MV
axes[0, 1].scatter(df_normal['CRIM'], df_normal['MV'], color='teal', alpha=0.6, label='Normal')
axes[0, 1].scatter(df_cap['CRIM'], df_cap['MV'], color='red', s=50, marker='^', label='MV = 50')
axes[0, 1].set_xlabel('CRIM (Crime Rate)')
axes[0, 1].set_ylabel('MV (Price in $1000s)')
axes[0, 1].set_title('CRIM vs MV: High Crime Exclusively Low Value', fontweight='bold')
axes[0, 1].legend()

# DIS vs MV
axes[1, 0].scatter(df_normal['DIS'], df_normal['MV'], color='teal', alpha=0.6, label='Normal')
axes[1, 0].scatter(df_cap['DIS'], df_cap['MV'], color='red', s=50, marker='^', label='MV = 50')
axes[1, 0].set_xlabel('DIS (Distance to Employment Centres)')
axes[1, 0].set_ylabel('MV (Price in $1000s)')
axes[1, 0].set_title('DIS vs MV: Distance vs Home Value', fontweight='bold')
axes[1, 0].legend()

# TAX vs MV
axes[1, 1].scatter(df_normal['TAX'], df_normal['MV'], color='teal', alpha=0.6, label='Normal')
axes[1, 1].scatter(df_cap['TAX'], df_cap['MV'], color='red', s=50, marker='^', label='MV = 50')
axes[1, 1].set_xlabel('TAX (Full-value property-tax rate)')
axes[1, 1].set_ylabel('MV (Price in $1000s)')
axes[1, 1].set_title('TAX vs MV: Bimodal Clustering around 666', fontweight='bold')
axes[1, 1].legend()

plt.suptitle("Scatter Analysis vs Target Variable (MV) & Cap Detection", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "scatter_mv_relationships.png"), dpi=200)
fig.savefig(os.path.join(workspace_plots, "scatter_mv_relationships.png"), dpi=200)
plt.close(fig)

print("Visualization generation completed successfully!")
