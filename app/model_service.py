# app/model_service.py
# ============================================================
# Shared model-loading and scoring logic for IntelliRisk AI.
#
# Extracted from main.py so that both the Streamlit UI (main.py)
# and the agentic chatbot (agent_tools.py) call ONE code path for
# loading models and scoring applicants/claims — no duplicated
# business logic between the UI and the agent.
# ============================================================
import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAN_MODELS_DIR  = os.path.join(BASE_DIR, 'Models', 'loan')
FRAUD_MODELS_DIR = os.path.join(BASE_DIR, 'Models', 'fraud')
DATA_DIR         = os.path.join(BASE_DIR, 'Data', 'processed')


# ── Model / data loaders (cached) ──────────────────────────────
@st.cache_resource
def load_loan_model():
    m = joblib.load(os.path.join(LOAN_MODELS_DIR, 'best_loan_model.pkl'))
    f = joblib.load(os.path.join(LOAN_MODELS_DIR, 'feature_names.pkl'))
    return m, f


@st.cache_resource
def load_fraud_models():
    m   = joblib.load(os.path.join(FRAUD_MODELS_DIR, 'best_fraud_model.pkl'))
    f   = joblib.load(os.path.join(FRAUD_MODELS_DIR, 'fraud_feature_names.pkl'))
    iso = joblib.load(os.path.join(FRAUD_MODELS_DIR, 'isolation_forest.pkl'))
    return m, f, iso


@st.cache_data
def load_loan_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'loan_processed.csv'))


@st.cache_data
def load_fraud_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'fraud_processed.csv'))


# ── Scoring functions (single source of truth) ─────────────────
def predict_loan(app: dict, model, feats):
    df = pd.DataFrame([app])
    for c in feats:
        if c not in df:
            df[c] = 0
    df = df[feats]
    p  = float(model.predict_proba(df)[0][1])
    # Bank policy thresholds
    if p < 0.10:
        dec, risk, col = 'APPROVED', 'LOW RISK', '#10b981'
    elif p < 0.15:
        dec, risk, col = 'APPROVED', 'MEDIUM RISK', '#f59e0b'
    else:
        dec, risk, col = 'REJECTED', 'HIGH RISK', '#ef4444'
    return {'decision': dec, 'risk': risk, 'color': col,
            'ap': round((1 - p) * 100, 1), 'dp': round(p * 100, 1), 'df': df}


def predict_fraud(claim: dict, model, feats, iso):
    df      = pd.DataFrame([claim])
    for c in feats:
        if c not in df:
            df[c] = 0
    df      = df[feats]
    p       = float(model.predict_proba(df)[0][1])
    ano     = iso.decision_function(df)[0]
    ano_pct = max(0, min(100, round((1 - (ano + 0.5)) * 100, 1)))
    if p < 0.3:
        ver, risk, col = 'LEGITIMATE', 'LOW RISK', '#10b981'
    elif p < 0.5:
        ver, risk, col = 'SUSPICIOUS', 'MEDIUM RISK', '#f59e0b'
    else:
        ver, risk, col = 'FRAUDULENT', 'HIGH RISK', '#ef4444'
    return {'verdict': ver, 'risk': risk, 'color': col,
            'fp': round(p * 100, 2), 'lp': round((1 - p) * 100, 2),
            'ano': ano_pct, 'df': df}


def get_shap(model, df, feats):
    try:
        import shap_patch
        import shap
        ex   = shap.TreeExplainer(model)
        vals = ex.shap_values(df)
        return pd.DataFrame({
            'Feature': feats,
            'Value'  : df.values[0],
            'SHAP'   : vals[0]
        }).sort_values('SHAP', key=abs, ascending=False).head(12), True
    except Exception:
        return pd.DataFrame({
            'Feature': list(feats[:12]),
            'Value'  : list(df.values[0][:12]),
            'SHAP'   : [0.0] * 12
        }), False
