# src/loan/predict.py
# ============================================================
# MODULE 1 — Loan Approval Intelligence
# Step 3: Prediction + SHAP Explainability
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(
             os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'Models', 'loan')
REPORTS    = os.path.join(BASE_DIR, 'Data', 'processed', 'reports')
os.makedirs(REPORTS, exist_ok=True)

# ── 1. Load Model ─────────────────────────────────────────────
def load_model():
    model         = joblib.load(os.path.join(MODELS_DIR,
                                             'best_loan_model.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR,
                                             'feature_names.pkl'))
    print("✅ Model loaded → XGBoost (best)", flush=True)
    return model, feature_names

# ── 2. Predict Single Applicant ───────────────────────────────
def predict_applicant(applicant_dict, model, feature_names):
    input_df = pd.DataFrame([applicant_dict])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    prob_default  = float(model.predict_proba(input_df)[0][1])
    prob_approved = 1 - prob_default

    decision = 'REJECTED ❌' if prob_default > 0.5 else 'APPROVED ✅'

    if prob_default < 0.2:
        risk = 'LOW RISK 🟢'
    elif prob_default < 0.5:
        risk = 'MEDIUM RISK 🟡'
    else:
        risk = 'HIGH RISK 🔴'

    return {
        'decision'       : decision,
        'risk_category'  : risk,
        'approval_prob'  : round(prob_approved * 100, 2),
        'default_prob'   : round(prob_default  * 100, 2),
        'input_features' : input_df,
    }

# ── 3. SHAP Explanation ───────────────────────────────────────
def explain_prediction(model, input_df, feature_names,
                       applicant_id='sample'):
    print("🔍 Generating SHAP explanation...", flush=True)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)

    shap_df = pd.DataFrame({
        'feature' : feature_names,
        'value'   : input_df.values[0],
        'shap'    : shap_vals[0]
    }).sort_values('shap', key=abs, ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71'
               for v in shap_df['shap']]
    bars    = ax.barh(shap_df['feature'],
                      shap_df['shap'], color=colors)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(
        'SHAP Explanation — Why was this decision made?',
        fontsize=13, fontweight='bold')
    ax.set_xlabel(
        'SHAP Value  '
        '(Red = increases default risk  |  '
        'Green = reduces default risk)')
    ax.invert_yaxis()

    for bar, (_, row) in zip(bars, shap_df.iterrows()):
        offset = 0.001 if row['shap'] >= 0 else -0.001
        ax.text(row['shap'] + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{row['value']:.2f}",
                va='center',
                ha='left' if row['shap'] >= 0 else 'right',
                fontsize=8)

    plt.tight_layout()
    out = os.path.join(REPORTS, f'shap_{applicant_id}.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"✅ SHAP plot saved → {out}", flush=True)

    return shap_df

# ── 4. Human-Readable Explanation ────────────────────────────
def generate_explanation(result, shap_df):
    lines = [
        f"\n{'─'*55}",
        f"  DECISION   : {result['decision']}",
        f"  RISK LEVEL : {result['risk_category']}",
        f"  APPROVAL % : {result['approval_prob']}%",
        f"  DEFAULT  % : {result['default_prob']}%",
        f"{'─'*55}",
        f"  KEY REASONS (from SHAP):",
    ]

    risk_factors    = shap_df[shap_df['shap'] > 0].head(3)
    protect_factors = shap_df[shap_df['shap'] < 0].head(3)

    if not risk_factors.empty:
        lines.append("\n  ⚠️  Factors INCREASING default risk:")
        for _, row in risk_factors.iterrows():
            lines.append(
                f"     • {row['feature']:<28} "
                f"value={row['value']:>8.2f}  "
                f"impact=+{row['shap']:.4f}")

    if not protect_factors.empty:
        lines.append("\n  ✅ Factors REDUCING default risk:")
        for _, row in protect_factors.iterrows():
            lines.append(
                f"     • {row['feature']:<28} "
                f"value={row['value']:>8.2f}  "
                f"impact={row['shap']:.4f}")

    lines.append(f"{'─'*55}")
    return '\n'.join(lines)

# ── 5. Sample Applicants ──────────────────────────────────────
def run_sample_predictions():
    print("\n" + "="*55, flush=True)
    print("  MODULE 1 — LOAN PREDICTION + SHAP EXPLAINABILITY",
          flush=True)
    print("="*55, flush=True)

    model, feature_names = load_model()

    applicants = {
        'GOOD_APPLICANT': {
            'AMT_INCOME_TOTAL'        : 150000,
            'AMT_CREDIT'              : 200000,
            'AMT_ANNUITY'             : 15000,
            'AMT_GOODS_PRICE'         : 180000,
            'EXT_SOURCE_1'            : 0.75,
            'EXT_SOURCE_2'            : 0.80,
            'EXT_SOURCE_3'            : 0.72,
            'AGE_YEARS'               : 42,
            'EMPLOYED_YEARS'          : 8,
            'CREDIT_INCOME_RATIO'     : 1.33,
            'ANNUITY_INCOME_RATIO'    : 0.10,
            'CREDIT_TERM'             : 0.075,
            'GOODS_CREDIT_RATIO'      : 0.90,
            'EXT_SOURCE_MEAN'         : 0.757,
            'CNT_CHILDREN'            : 1,
            'CNT_FAM_MEMBERS'         : 3,
            'FLAG_OWN_CAR'            : 1,
            'FLAG_OWN_REALTY'         : 1,
            'DEF_30_CNT_SOCIAL_CIRCLE': 0,
            'DEF_60_CNT_SOCIAL_CIRCLE': 0,
        },
        'RISKY_APPLICANT': {
            'AMT_INCOME_TOTAL'        : 27000,
            'AMT_CREDIT'              : 450000,
            'AMT_ANNUITY'             : 45000,
            'AMT_GOODS_PRICE'         : 400000,
            'EXT_SOURCE_1'            : 0.10,
            'EXT_SOURCE_2'            : 0.15,
            'EXT_SOURCE_3'            : 0.12,
            'AGE_YEARS'               : 22,
            'EMPLOYED_YEARS'          : 0.5,
            'CREDIT_INCOME_RATIO'     : 16.67,
            'ANNUITY_INCOME_RATIO'    : 1.67,
            'CREDIT_TERM'             : 0.10,
            'GOODS_CREDIT_RATIO'      : 0.89,
            'EXT_SOURCE_MEAN'         : 0.123,
            'CNT_CHILDREN'            : 3,
            'CNT_FAM_MEMBERS'         : 5,
            'FLAG_OWN_CAR'            : 0,
            'FLAG_OWN_REALTY'         : 0,
            'DEF_30_CNT_SOCIAL_CIRCLE': 2,
            'DEF_60_CNT_SOCIAL_CIRCLE': 1,
        },
    }

    for name, applicant in applicants.items():
        print(f"\n{'='*55}", flush=True)
        print(f"  APPLICANT: {name}", flush=True)
        print(f"{'='*55}", flush=True)

        result  = predict_applicant(applicant, model,
                                    feature_names)
        shap_df = explain_prediction(model,
                                     result['input_features'],
                                     feature_names,
                                     applicant_id=name)
        explanation = generate_explanation(result, shap_df)
        print(explanation, flush=True)

    print("\n✅ PREDICTION PIPELINE COMPLETE!", flush=True)
    print("   SHAP plots → Data/processed/reports/", flush=True)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    run_sample_predictions()