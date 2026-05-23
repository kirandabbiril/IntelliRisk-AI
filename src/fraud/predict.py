# src/fraud/predict.py
# ============================================================
# MODULE 2 — Insurance Fraud Detection
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

sys.stdout.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(
             os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'Models', 'fraud')
REPORTS    = os.path.join(BASE_DIR, 'Data', 'processed', 'reports')
os.makedirs(REPORTS, exist_ok=True)

# ── 1. Load Model ─────────────────────────────────────────────
def load_model():
    model         = joblib.load(
        os.path.join(MODELS_DIR, 'best_fraud_model.pkl'))
    feature_names = joblib.load(
        os.path.join(MODELS_DIR, 'fraud_feature_names.pkl'))
    print("✅ Fraud model loaded → XGBoost (best)", flush=True)
    return model, feature_names

# ── 2. Predict Single Claim ───────────────────────────────────
def predict_claim(claim_dict: dict, model, feature_names):
    df = pd.DataFrame([claim_dict])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    prob_fraud     = float(model.predict_proba(df)[0][1])
    prob_legit     = 1 - prob_fraud

    if prob_fraud < 0.3:
        verdict  = 'LEGITIMATE ✅'
        risk     = 'LOW RISK 🟢'
        color    = '#2ecc71'
    elif prob_fraud < 0.5:
        verdict  = 'SUSPICIOUS ⚠️'
        risk     = 'MEDIUM RISK 🟡'
        color    = '#f39c12'
    else:
        verdict  = 'FRAUDULENT ❌'
        risk     = 'HIGH RISK 🔴'
        color    = '#e74c3c'

    # Anomaly score from Isolation Forest
    iso_model  = joblib.load(
        os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    anomaly    = iso_model.decision_function(df)[0]
    anomaly_pct = round((1 - (anomaly + 0.5)) * 100, 1)
    anomaly_pct = max(0, min(100, anomaly_pct))

    return {
        'verdict'      : verdict,
        'risk'         : risk,
        'color'        : color,
        'fraud_prob'   : round(prob_fraud * 100, 2),
        'legit_prob'   : round(prob_legit  * 100, 2),
        'anomaly_score': anomaly_pct,
        'input_df'     : df,
    }

# ── 3. SHAP Explanation ───────────────────────────────────────
def explain_claim(model, input_df, feature_names,
                  claim_id='sample'):
    print("🔍 Generating SHAP explanation...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)

    shap_df = pd.DataFrame({
        'Feature' : feature_names,
        'Value'   : input_df.values[0],
        'SHAP'    : shap_vals[0]
    }).sort_values('SHAP', key=abs, ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71'
               for v in shap_df['SHAP']]
    ax.barh(shap_df['Feature'], shap_df['SHAP'],
            color=colors)
    ax.axvline(0, color='white', linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(
        'SHAP — Why was this claim flagged?',
        fontsize=13, fontweight='bold')
    ax.set_xlabel(
        'SHAP Value  '
        '(Red = increases fraud risk  |  '
        'Green = reduces fraud risk)')
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    out = os.path.join(REPORTS, f'fraud_shap_{claim_id}.png')
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"✅ SHAP plot saved → {out}", flush=True)
    return shap_df

# ── 4. Human Explanation ──────────────────────────────────────
def generate_explanation(result, shap_df):
    lines = [
        f"\n{'─'*55}",
        f"  VERDICT      : {result['verdict']}",
        f"  RISK LEVEL   : {result['risk']}",
        f"  FRAUD PROB   : {result['fraud_prob']}%",
        f"  LEGIT PROB   : {result['legit_prob']}%",
        f"  ANOMALY SCORE: {result['anomaly_score']}%",
        f"{'─'*55}",
        f"  KEY FRAUD INDICATORS (SHAP):",
    ]

    fraud_factors = shap_df[shap_df['SHAP'] > 0].head(3)
    clean_factors = shap_df[shap_df['SHAP'] < 0].head(3)

    if not fraud_factors.empty:
        lines.append("\n  🚨 Factors INDICATING fraud:")
        for _, row in fraud_factors.iterrows():
            lines.append(
                f"     • {row['Feature']:<28} "
                f"value={row['Value']:>6.2f}  "
                f"impact=+{row['SHAP']:.4f}")

    if not clean_factors.empty:
        lines.append("\n  ✅ Factors AGAINST fraud:")
        for _, row in clean_factors.iterrows():
            lines.append(
                f"     • {row['Feature']:<28} "
                f"value={row['Value']:>6.2f}  "
                f"impact={row['SHAP']:.4f}")

    lines.append(f"{'─'*55}")
    return '\n'.join(lines)

# ── 5. Sample Claims ──────────────────────────────────────────
def run_sample_predictions():
    print("\n" + "="*55, flush=True)
    print("  MODULE 2 — FRAUD PREDICTION + SHAP", flush=True)
    print("="*55, flush=True)

    model, feature_names = load_model()

    claims = {
        'LEGITIMATE_CLAIM': {
            'Month'               : 2,
            'WeekOfMonth'         : 5,
            'DayOfWeek'           : 6,
            'Make'                : 6,
            'AccidentArea'        : 1,
            'DayOfWeekClaimed'    : 6,
            'MonthClaimed'        : 5,
            'WeekOfMonthClaimed'  : 1,
            'Sex'                 : 0,
            'MaritalStatus'       : 2,
            'Age'                 : 21,
            'Fault'               : 0,
            'PolicyType'          : 5,
            'VehicleCategory'     : 1,
            'VehiclePrice'        : 5,
            'Deductible'          : 300,
            'DriverRating'        : 1,
            'Days_Policy_Accident': 3,
            'Days_Policy_Claim'   : 2,
            'PastNumberOfClaims'  : 3,
            'AgeOfVehicle'        : 1,
            'AgeOfPolicyHolder'   : 3,
            'PoliceReportFiled'   : 0,
            'WitnessPresent'      : 0,
            'AgentType'           : 0,
            'NumberOfSuppliments' : 3,
            'AddressChange_Claim' : 0,
            'NumberOfCars'        : 2,
            'Year'                : 1994,
            'BasePolicy'          : 2,
            'AGE_RISK'            : 1,
            'MULTI_CAR'           : 0,
            'HIGH_DEDUCTIBLE'     : 0,
            'QUICK_CLAIM'         : 0,
            'NO_WITNESS_NO_POLICE': 1,
        },
        'FRAUDULENT_CLAIM': {
            'Month'               : 5,
            'WeekOfMonth'         : 1,
            'DayOfWeek'           : 2,
            'Make'                : 6,
            'AccidentArea'        : 1,
            'DayOfWeekClaimed'    : 6,
            'MonthClaimed'        : 12,
            'WeekOfMonthClaimed'  : 4,
            'Sex'                 : 1,
            'MaritalStatus'       : 2,
            'Age'                 : 0,
            'Fault'               : 0,
            'PolicyType'          : 0,
            'VehicleCategory'     : 0,
            'VehiclePrice'        : 5,
            'Deductible'          : 400,
            'DriverRating'        : 1,
            'Days_Policy_Accident': 3,
            'Days_Policy_Claim'   : 2,
            'PastNumberOfClaims'  : 3,
            'AgeOfVehicle'        : 7,
            'AgeOfPolicyHolder'   : 0,
            'PoliceReportFiled'   : 0,
            'WitnessPresent'      : 0,
            'AgentType'           : 0,
            'NumberOfSuppliments' : 3,
            'AddressChange_Claim' : 3,
            'NumberOfCars'        : 0,
            'Year'                : 1994,
            'BasePolicy'          : 0,
            'AGE_RISK'            : 1,
            'MULTI_CAR'           : 0,
            'HIGH_DEDUCTIBLE'     : 0,
            'QUICK_CLAIM'         : 0,
            'NO_WITNESS_NO_POLICE': 1,
        },
    }

    for name, claim in claims.items():
        print(f"\n{'='*55}", flush=True)
        print(f"  CLAIM: {name}", flush=True)
        print(f"{'='*55}", flush=True)

        result  = predict_claim(claim, model, feature_names)
        shap_df = explain_claim(model,
                                result['input_df'],
                                feature_names,
                                claim_id=name)
        explanation = generate_explanation(result, shap_df)
        print(explanation, flush=True)

    print("\n✅ FRAUD PREDICTION COMPLETE!", flush=True)
    print("   SHAP plots → Data/processed/reports/", flush=True)

if __name__ == '__main__':
    run_sample_predictions()