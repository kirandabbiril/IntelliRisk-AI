# src/fraud/preprocess.py
# ============================================================
# MODULE 2 — Insurance Fraud Detection
# Step 1: Data Loading & Preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
RAW_DIR   = os.path.join(BASE_DIR, 'Data', 'raw')
PROCESSED = os.path.join(BASE_DIR, 'Data', 'processed')
os.makedirs(PROCESSED, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
def load_data():
    path = os.path.join(RAW_DIR, 'fraud_oracle.csv')
    print("📂 Loading fraud_oracle.csv ...")
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# ── 2. Explore ────────────────────────────────────────────────
def explore_data(df):
    # Find the fraud column
    fraud_col = None
    for col in ['FraudFound_P', 'fraud', 'Fraud',
                'FRAUD', 'FraudFound']:
        if col in df.columns:
            fraud_col = col
            break

    if fraud_col is None:
        # Try to find any binary column
        for col in df.columns:
            if df[col].nunique() == 2:
                fraud_col = col
                print(f"⚠️  Using '{col}' as fraud label")
                break

    print(f"\n📊 DATASET OVERVIEW")
    print("="*50)
    print(f"Shape        : {df.shape}")
    print(f"Fraud column : {fraud_col}")
    print(f"Fraud cases  : {df[fraud_col].sum()} "
          f"({df[fraud_col].mean()*100:.1f}%)")
    print(f"Missing vals : {df.isnull().sum().sum()}")
    print(f"\nColumns:\n{list(df.columns)}")
    return fraud_col

# ── 3. Clean & Engineer Features ─────────────────────────────
def clean_data(df, fraud_col):
    print(f"\n🔧 Cleaning data...")

    # Rename fraud column to standard name
    if fraud_col != 'FraudFound_P':
        df = df.rename(columns={fraud_col: 'FraudFound_P'})

    # Drop ID-like columns if they exist
    drop_cols = ['PolicyNumber', 'policy_number',
                 'ClaimNumber', 'claim_number', 'RepNumber']
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"  Dropped ID columns: {drop_cols}")

    # Print column types to understand the data
    print(f"\n  Column types:")
    print(df.dtypes.value_counts())

    # Age risk flag (numeric column)
    if 'Age' in df.columns:
        df['AGE_RISK'] = df['Age'].apply(
            lambda x: 1 if (pd.to_numeric(x, errors='coerce') or 0) < 25
            or (pd.to_numeric(x, errors='coerce') or 0) > 70
            else 0)

    # Multi car flag
    if 'NumberOfCars' in df.columns:
        df['MULTI_CAR'] = (
            pd.to_numeric(df['NumberOfCars'],
                          errors='coerce').fillna(0) > 1
        ).astype(int)

    # High deductible flag
    if 'Deductible' in df.columns:
        df['HIGH_DEDUCTIBLE'] = (
            pd.to_numeric(df['Deductible'],
                          errors='coerce').fillna(0) > 500
        ).astype(int)

    # Suspicious timing flag
    # Claims filed very quickly after policy = suspicious
    if 'Days_Policy_Accident' in df.columns:
        df['QUICK_CLAIM'] = df['Days_Policy_Accident'].apply(
            lambda x: 1 if str(x) in ['none', '1 week', '2 weeks']
            else 0)

    # No witness + no police = suspicious combination
    if 'WitnessPresent' in df.columns and \
       'PoliceReportFiled' in df.columns:
        df['NO_WITNESS_NO_POLICE'] = (
            (df['WitnessPresent'] == 'No') &
            (df['PoliceReportFiled'] == 'No')
        ).astype(int)

    print(f"✅ Feature engineering complete")
    print(f"   Shape after cleaning: {df.shape}")
    return df

# ── 4. Handle Missing Values ──────────────────────────────────
def handle_missing(df):
    print(f"\n🩹 Handling missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        if col != 'FraudFound_P':
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"✅ Missing after clean: {df.isnull().sum().sum()}")
    return df

# ── 5. Encode Categoricals ────────────────────────────────────
def encode_categoricals(df):
    print(f"\n🔤 Encoding categorical columns...")
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"✅ Encoded {len(cat_cols)} columns")
    return df

# ── 6. Save ───────────────────────────────────────────────────
def save_processed(df):
    out = os.path.join(PROCESSED, 'fraud_processed.csv')
    df.to_csv(out, index=False)
    print(f"\n💾 Saved → {out}")
    print(f"   Shape : {df.shape}")
    print(f"   Fraud rate: "
          f"{df['FraudFound_P'].mean()*100:.1f}%")

# ── MAIN ──────────────────────────────────────────────────────
def run_preprocessing():
    print("="*50)
    print("  MODULE 2 — FRAUD PREPROCESSING PIPELINE")
    print("="*50)

    df        = load_data()
    fraud_col = explore_data(df)
    df        = clean_data(df, fraud_col)
    df        = handle_missing(df)
    df        = encode_categoricals(df)
    save_processed(df)

    print("\n✅ FRAUD PREPROCESSING COMPLETE!")
    return df

if __name__ == '__main__':
    df = run_preprocessing()