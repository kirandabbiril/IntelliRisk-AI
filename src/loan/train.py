# src/loan/train.py
# ============================================================
# MODULE 1 — Loan Approval Intelligence
# Step 2: Model Training (XGBoost + Random Forest)
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (classification_report,
                                       roc_auc_score,
                                       confusion_matrix,
                                       ConfusionMatrixDisplay)
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for scripts)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED   = os.path.join(BASE_DIR, 'Data', 'processed')
MODELS_DIR  = os.path.join(BASE_DIR, 'Models', 'loan')
REPORTS_DIR = os.path.join(BASE_DIR, 'Data', 'processed', 'reports')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── 1. Load Processed Data ───────────────────────────────────
def load_processed():
    path = os.path.join(PROCESSED, 'loan_processed.csv')
    print("📂 Loading processed data...")
    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# ── 2. Split Features / Target ───────────────────────────────
def split_data(df):
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y       # keeps class ratio in both splits
    )
    
    print(f"\n📊 SPLIT SUMMARY")
    print(f"   Train : {X_train.shape[0]:,} rows")
    print(f"   Test  : {X_test.shape[0]:,}  rows")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Default rate (train): "
          f"{y_train.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test

# ── 3. Train Models ──────────────────────────────────────────
def train_logistic(X_train, y_train):
    print("\n🔵 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter     = 1000,
        class_weight = 'balanced',   # handles imbalance
        random_state = 42
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained")
    return model

def train_random_forest(X_train, y_train):
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 10,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs       = -1          # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest trained")
    return model

def train_xgboost(X_train, y_train):
    print("\n⚡ Training XGBoost...")
    
    # Calculate scale for imbalanced data
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale,   # handles imbalance
        use_label_encoder = False,
        eval_metric       = 'auc',
        random_state      = 42,
        n_jobs            = -1
    )
    model.fit(
        X_train, y_train,
        verbose = False
    )
    print("✅ XGBoost trained")
    return model

# ── 4. Evaluate Models ───────────────────────────────────────
def evaluate_model(model, X_test, y_test, name):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_prob)
    
    print(f"\n{'='*50}")
    print(f"  {name} — RESULTS")
    print(f"{'='*50}")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Approved','Rejected'])}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = ['Approved', 'Rejected']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name} — Confusion Matrix (AUC: {auc:.4f})')
    plt.tight_layout()
    
    fname = name.lower().replace(' ', '_')
    plt.savefig(os.path.join(REPORTS_DIR, f'{fname}_confusion.png'),
                dpi=120)
    plt.close()
    print(f"  📊 Confusion matrix saved")
    
    return auc

# ── 5. Feature Importance Plot ───────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(
            model.feature_importances_,
            index = feature_names
        ).sort_values(ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind='barh', ax=ax, color='steelblue')
        ax.invert_yaxis()
        ax.set_title(f'{model_name} — Top 15 Feature Importances')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        
        fname = model_name.lower().replace(' ', '_')
        plt.savefig(
            os.path.join(REPORTS_DIR, f'{fname}_feature_importance.png'),
            dpi=120
        )
        plt.close()
        print(f"  📊 Feature importance plot saved")

# ── 6. Save Best Model ───────────────────────────────────────
def save_best_model(models_scores: dict):
    best_name  = max(models_scores, key=lambda k: models_scores[k]['auc'])
    best_model = models_scores[best_name]['model']
    best_auc   = models_scores[best_name]['auc']
    
    path = os.path.join(MODELS_DIR, 'best_loan_model.pkl')
    joblib.dump(best_model, path)
    
    # Also save all models individually
    for name, info in models_scores.items():
        fname = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(info['model'],
                    os.path.join(MODELS_DIR, fname))
    
    print(f"\n🏆 BEST MODEL : {best_name} (AUC = {best_auc:.4f})")
    print(f"💾 Saved → {path}")
    return best_model, best_name

# ── MAIN ─────────────────────────────────────────────────────
def run_training():
    print("="*50)
    print("  MODULE 1 — LOAN MODEL TRAINING PIPELINE")
    print("="*50)
    
    # Load & split
    df = load_processed()
    X_train, X_test, y_train, y_test = split_data(df)
    feature_names = X_train.columns.tolist()
    
    # Train all three models
    lr_model  = train_logistic(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate all models
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
    
    # Summary table
    print("\n📊 MODEL COMPARISON SUMMARY")
    print(f"{'Model':<25} {'ROC-AUC':>10}")
    print("-" * 37)
    for name, info in sorted(models_scores.items(),
                              key=lambda x: x[1]['auc'],
                              reverse=True):
        print(f"{name:<25} {info['auc']:>10.4f}")
    
    # Save best
    best_model, best_name = save_best_model(models_scores)
    
    # Save feature names for later use in predict.py
    joblib.dump(feature_names,
                os.path.join(MODELS_DIR, 'feature_names.pkl'))
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"   Reports saved → Data/processed/reports/")
    print(f"   Models  saved → Models/loan/")
    
    return best_model, best_name, models_scores

if __name__ == '__main__':
    run_training()