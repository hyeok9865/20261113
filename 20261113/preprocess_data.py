import pandas as pd
import numpy as np

def clean_boston_dataset(input_csv="boston.csv", output_csv="boston_clean.csv", drop_capped_mv=True, log_transform_skewed=True, winsorize_rm=False):
    """
    Cleans and preprocesses the Boston Housing dataset based on outlier analysis findings:
    1. Removes censored target values (MV == 50.0) if drop_capped_mv is True.
    2. Applies log1p transform to skewed features ('CRIM', 'DIS') if log_transform_skewed is True.
    3. Optionally winsorizes 'RM' at the 1st and 99th percentiles if winsorize_rm is True.
    """
    df = pd.read_csv(input_csv)
    original_len = len(df)
    
    # 1. Filter out censored cap on target variable
    if drop_capped_mv:
        df = df[df['MV'] < 50.0].copy()
        dropped_count = original_len - len(df)
        print(f"[Preprocess] Dropped {dropped_count} censored records with MV == 50.0. Remaining rows: {len(df)}")
    
    # 2. Log transform severely skewed variables
    if log_transform_skewed:
        df['CRIM_log'] = np.log1p(df['CRIM'])
        df['DIS_log'] = np.log(df['DIS'])
        print("[Preprocess] Applied log transformation: CRIM -> CRIM_log, DIS -> DIS_log")
    
    # 3. Optional RM Winsorization
    if winsorize_rm:
        p1, p99 = np.percentile(df['RM'], [1, 99])
        df['RM'] = np.clip(df['RM'], p1, p99)
        print(f"[Preprocess] Winsorized RM between [{p1:.2f}, {p99:.2f}]")
        
    df.to_csv(output_csv, index=False)
    print(f"[Preprocess] Successfully saved cleaned dataset to '{output_csv}' with shape {df.shape}")
    return df

if __name__ == "__main__":
    clean_boston_dataset()
