# src/loan/preprocess.py
# ============================================================
# MODULE 1 — Loan Approval Intelligence
# Step 1: Data Loading & Preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, 'Data', 'raw', 'home-credit')
PROCESSED   = os.path.join(BASE_DIR, 'Data', 'processed')
os.makedirs(PROCESSED, exist_ok=True)

# ── 1. Load Main Application File ───────────────────────────
def load_data():
    print("📂 Loading application_train.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, 'application_train.csv'))
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🎯 Fraud rate (defaults): {df['TARGET'].mean()*100:.2f}%")
    return df

# ── 2. Basic Exploration ─────────────────────────────────────
def explore_data(df):
    print("\n📊 DATASET OVERVIEW")
    print("="*50)
    print(f"Shape        : {df.shape}")
    print(f"Target (0/1) : {df['TARGET'].value_counts().to_dict()}")
    print(f"Missing vals : {df.isnull().sum().sum():,} total")
    print(f"Duplicates   : {df.duplicated().sum()}")
    
    # Top columns with missing values
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n⚠️  Columns with >40% missing: "
          f"{(missing > 40).sum()}")
    return missing

# ── 3. Feature Selection (Top Features) ─────────────────────
def select_features(df):
    """
    Start with the most impactful features.
    These are proven features from research on this dataset.
    """
    features = [
        # Target
        'TARGET',
        
        # Income & Loan
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
        
        # External Credit Scores (most predictive!)
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        
        # Employment & Demographics
        'DAYS_BIRTH',           # Age
        'DAYS_EMPLOYED',        # Employment length
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
        
        # Loan Features
        'REGION_POPULATION_RELATIVE',
        'HOUR_APPR_PROCESS_START',
        'OWN_CAR_AGE',
        
        # Categorical
        'NAME_CONTRACT_TYPE',
        'CODE_GENDER',
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        
        # Risk Flags
        'FLAG_DOCUMENT_3',
        'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        
        # Family
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
    ]
    
    # Keep only columns that exist in the dataset
    features = [f for f in features if f in df.columns]
    df = df[features].copy()
    print(f"\n✅ Selected {len(features)-1} features + TARGET")
    return df

# ── 4. Clean & Engineer Features ─────────────────────────────
def clean_data(df):
    print("\n🔧 Cleaning data...")
    
    # Fix DAYS_EMPLOYED anomaly (365243 = unemployed encoding)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Convert negative days to positive (age/tenure in years)
    df['AGE_YEARS']      = (-df['DAYS_BIRTH']) / 365
    df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED']) / 365
    
    # Key engineered ratios (very impactful for recruiters!)
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']          = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    
    # Average external score
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1',
                                 'EXT_SOURCE_2',
                                 'EXT_SOURCE_3']].mean(axis=1)
    
    # Drop raw day columns (replaced by engineered ones)
    drop_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED',
                 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True)
    
    print(f"✅ Engineered features added")
    return df

# ── 5. Handle Missing Values ──────────────────────────────────
def handle_missing(df):
    print("\n🩹 Handling missing values...")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numeric → median
    for col in num_cols:
        if col != 'TARGET':
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical → mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"✅ Missing values after cleaning: "
          f"{df.isnull().sum().sum()}")
    return df

# ── 6. Encode Categoricals ────────────────────────────────────
def encode_categoricals(df):
    print("\n🔤 Encoding categorical columns...")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    print(f"✅ Encoded {len(cat_cols)} categorical columns")
    return df

# ── 7. Save Processed Data ────────────────────────────────────
def save_processed(df):
    out_path = os.path.join(PROCESSED, 'loan_processed.csv')
    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved → {out_path}")
    print(f"   Final shape: {df.shape}")

# ── MAIN ──────────────────────────────────────────────────────
def run_preprocessing():
    print("="*50)
    print("  MODULE 1 — LOAN PREPROCESSING PIPELINE")
    print("="*50)
    
    df = load_data()
    explore_data(df)
    df = select_features(df)
    df = clean_data(df)
    df = handle_missing(df)
    df = encode_categoricals(df)
    save_processed(df)
    
    print("\n✅ PREPROCESSING COMPLETE!")
    print(f"   Shape  : {df.shape}")
    print(f"   Target : {df['TARGET'].value_counts().to_dict()}")
    return df

if __name__ == '__main__':
    df = run_preprocessing()