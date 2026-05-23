# src/fraud/train.py
# ============================================================
# MODULE 2 — Insurance Fraud Detection
# Step 2: Model Training
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.ensemble         import RandomForestClassifier, IsolationForest
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (classification_report,
                                      roc_auc_score,
                                      confusion_matrix,
                                      ConfusionMatrixDisplay)
from imblearn.over_sampling   import SMOTE
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__))))
PROCESSED   = os.path.join(BASE_DIR, 'Data', 'processed')
MODELS_DIR  = os.path.join(BASE_DIR, 'Models', 'fraud')
REPORTS_DIR = os.path.join(BASE_DIR, 'Data', 'processed', 'reports')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
def load_data():
    path = os.path.join(PROCESSED, 'fraud_processed.csv')
    print("📂 Loading fraud_processed.csv ...")
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🎯 Fraud rate: {df['FraudFound_P'].mean()*100:.1f}%")
    return df

# ── 2. Split Data ─────────────────────────────────────────────
def split_data(df):
    X = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )

    print(f"\n📊 SPLIT SUMMARY")
    print(f"   Train : {X_train.shape[0]:,} rows")
    print(f"   Test  : {X_test.shape[0]:,} rows")
    print(f"   Fraud in train: {y_train.sum()} cases "
          f"({y_train.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test

# ── 3. Apply SMOTE ────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print(f"\n⚖️  Applying SMOTE to balance classes...")
    print(f"   Before: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"   After : {pd.Series(y_res).value_counts().to_dict()}")
    print(f"✅ SMOTE applied — classes balanced")
    return X_res, y_res

# ── 4. Train Models ───────────────────────────────────────────
def train_logistic(X_train, y_train):
    print("\n🔵 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter     = 1000,
        class_weight = 'balanced',
        random_state = 42
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")
    return model

def train_random_forest(X_train, y_train):
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = 12,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest trained")
    return model

def train_xgboost(X_train, y_train):
    print("\n⚡ Training XGBoost...")
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale,
        eval_metric      = 'auc',
        random_state     = 42,
        n_jobs           = -1
    )
    model.fit(X_train, y_train, verbose=False)
    print("✅ XGBoost trained")
    return model

def train_isolation_forest(X_train):
    print("\n🔍 Training Isolation Forest (anomaly detection)...")
    model = IsolationForest(
        n_estimators = 200,
        contamination= 0.06,   # ~6% fraud rate
        random_state = 42,
        n_jobs       = -1
    )
    model.fit(X_train)
    print("✅ Isolation Forest trained")
    return model

# ── 5. Evaluate ───────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name} — RESULTS")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Fraud']))

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = ['Legitimate', 'Fraud']
    ).plot(ax=ax, colorbar=False, cmap='Reds')
    ax.set_title(f'{name} — Confusion Matrix (AUC: {auc:.4f})')
    plt.tight_layout()
    fname = name.lower().replace(' ', '_')
    plt.savefig(
        os.path.join(REPORTS_DIR, f'fraud_{fname}_cm.png'),
        dpi=120)
    plt.close()
    print(f"  📊 Confusion matrix saved")
    return auc

# ── 6. Feature Importance ─────────────────────────────────────
def plot_feature_importance(model, feature_names, name):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind='barh', ax=ax, color='crimson')
        ax.invert_yaxis()
        ax.set_title(f'{name} — Top 15 Fraud Indicators')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        fname = name.lower().replace(' ', '_')
        plt.savefig(
            os.path.join(REPORTS_DIR,
                         f'fraud_{fname}_importance.png'),
            dpi=120)
        plt.close()
        print(f"  📊 Feature importance saved")

# ── 7. Save Best Model ────────────────────────────────────────
def save_models(models_scores, feature_names):
    best_name  = max(models_scores,
                     key=lambda k: models_scores[k]['auc'])
    best_model = models_scores[best_name]['model']
    best_auc   = models_scores[best_name]['auc']

    # Save best
    joblib.dump(best_model,
                os.path.join(MODELS_DIR,
                             'best_fraud_model.pkl'))

    # Save all
    for name, info in models_scores.items():
        fname = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(info['model'],
                    os.path.join(MODELS_DIR, fname))

    # Save feature names
    joblib.dump(feature_names,
                os.path.join(MODELS_DIR,
                             'fraud_feature_names.pkl'))

    print(f"\n🏆 BEST MODEL : {best_name} "
          f"(AUC = {best_auc:.4f})")
    print(f"💾 Models saved → Models/fraud/")
    return best_model, best_name

# ── MAIN ──────────────────────────────────────────────────────
def run_training():
    print("="*50)
    print("  MODULE 2 — FRAUD MODEL TRAINING PIPELINE")
    print("="*50)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    feature_names = X_train.columns.tolist()

    # Apply SMOTE to handle 6% fraud imbalance
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Train models on balanced data
    lr_model  = train_logistic(X_train_bal,  y_train_bal)
    rf_model  = train_random_forest(X_train_bal, y_train_bal)
    xgb_model = train_xgboost(X_train_bal,   y_train_bal)

    # Also train Isolation Forest (unsupervised)
    iso_model = train_isolation_forest(X_train)
    joblib.dump(iso_model,
                os.path.join(MODELS_DIR,
                             'isolation_forest.pkl'))
    print("💾 Isolation Forest saved")

    # Evaluate on original (unbalanced) test set
    print("\n📈 EVALUATING ALL MODELS...")
    models_scores = {}

    for name, model in [
        ('Logistic Regression', lr_model),
        ('Random Forest',       rf_model),
        ('XGBoost',             xgb_model),
    ]:
        auc = evaluate_model(model, X_test, y_test, name)
        plot_feature_importance(model, feature_names, name)
        models_scores[name] = {'model': model, 'auc': auc}

    # Summary
    print(f"\n📊 MODEL COMPARISON")
    print(f"{'Model':<25} {'ROC-AUC':>10}")
    print("-"*37)
    for name, info in sorted(
            models_scores.items(),
            key=lambda x: x[1]['auc'],
            reverse=True):
        print(f"{name:<25} {info['auc']:>10.4f}")

    save_models(models_scores, feature_names)

    print("\n✅ FRAUD TRAINING COMPLETE!")
    print(f"   Reports → Data/processed/reports/")
    print(f"   Models  → Models/fraud/")

if __name__ == '__main__':
    run_training()