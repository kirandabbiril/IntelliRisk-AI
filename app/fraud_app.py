# app/fraud_app.py
# ============================================================
# MODULE 2 — Insurance Fraud Detection
# Streamlit Dashboard — Premium Redesign
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, sys, warnings
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import plotly.express as px
import shap
warnings.filterwarnings('ignore')

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models', 'fraud')
DATA_DIR   = os.path.join(BASE_DIR, 'Data', 'processed')
sys.path.insert(0, BASE_DIR)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "IntelliRisk AI — Fraud Detection",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #07090f; }

.page-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(
        135deg, #ef4444 0%, #f97316 50%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 6px 0;
}
.page-sub {
    color: #8892b0;
    font-size: 15px;
    font-weight: 300;
    margin-bottom: 16px;
}
.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.badge {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-white {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    color: #cbd5e1;
}
.badge-green {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
}
.badge-orange {
    background: rgba(249,115,22,0.12);
    border: 1px solid rgba(249,115,22,0.3);
    color: #fdba74;
}
.step-card {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #2a1515;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s;
}
.step-number {
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px;
    margin: 0 auto 12px auto;
}
.step-title {
    color: #e2e8f0; font-weight: 700;
    font-size: 14px; margin-bottom: 7px;
}
.step-desc { color: #64748b; font-size: 12px; line-height: 1.55; }
.section-header {
    font-size: 11px; font-weight: 700;
    color: #ef4444; text-transform: uppercase;
    letter-spacing: 1.8px;
    border-bottom: 1px solid #2a1515;
    padding-bottom: 7px; margin: 18px 0 13px 0;
}
.fraud-banner {
    background: linear-gradient(135deg, #2d0000, #450a0a);
    border: 2px solid #ef4444; border-radius: 16px;
    padding: 26px; text-align: center;
    font-size: 30px; font-weight: 800;
    color: #ef4444; margin: 12px 0; letter-spacing: 2px;
}
.legit-banner {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #10b981; border-radius: 16px;
    padding: 26px; text-align: center;
    font-size: 30px; font-weight: 800;
    color: #10b981; margin: 12px 0; letter-spacing: 2px;
}
.suspicious-banner {
    background: linear-gradient(135deg, #1c1200, #2d1f00);
    border: 2px solid #f59e0b; border-radius: 16px;
    padding: 26px; text-align: center;
    font-size: 30px; font-weight: 800;
    color: #f59e0b; margin: 12px 0; letter-spacing: 2px;
}
.sb-metric {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #2a1515; border-radius: 10px;
    padding: 14px 12px; margin-bottom: 9px; text-align: center;
}
.sb-label {
    color: #64748b; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
}
.sb-value {
    color: #fca5a5; font-size: 21px;
    font-weight: 700; margin-top: 3px;
}
.tech-card {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #2a1515; border-radius: 11px;
    padding: 15px; margin-bottom: 10px;
}
.tech-name {
    font-size: 14px; font-weight: 700;
    color: #fca5a5; margin-bottom: 3px;
}
.tech-desc { font-size: 12px; color: #8892b0; line-height: 1.5; }
.footer {
    background: linear-gradient(135deg, #0b0e1a, #0f1322);
    border: 1px solid #2a1515; border-radius: 14px;
    padding: 22px; text-align: center; margin-top: 40px;
}
.footer-title {
    color: #fca5a5; font-weight: 800;
    font-size: 17px; margin-bottom: 6px;
}
.footer-sub { color: #64748b; font-size: 12px; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    m   = joblib.load(
        os.path.join(MODELS_DIR, 'best_fraud_model.pkl'))
    f   = joblib.load(
        os.path.join(MODELS_DIR, 'fraud_feature_names.pkl'))
    iso = joblib.load(
        os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    return m, f, iso

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(DATA_DIR, 'fraud_processed.csv'))

# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:18px 0 10px 0;'>
            <div style='font-size:40px;'>🛡️</div>
            <div style='font-size:19px;font-weight:800;
                        color:#fca5a5;margin-top:6px;'>
                IntelliRisk AI
            </div>
            <div style='font-size:11px;color:#64748b;
                        margin-top:3px;'>
                Fraud Detection Module
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Model Performance Metrics ─────────────────────────
        st.markdown("**📊 Model Performance**")
        for label, value in [
            ("Best Model",    "XGBoost"),
            ("ROC-AUC Score", "0.8115"),
            ("Training Data", "15,420"),
            ("Fraud Rate",    "6.0%"),
            ("SMOTE",         "Balanced"),
        ]:
            st.markdown(f"""
            <div class='sb-metric'>
                <div class='sb-label'>{label}</div>
                <div class='sb-value'>{value}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Tech Stack ────────────────────────────────────────
        st.markdown("**💻 Tech Stack**")
        techs = [
            ("🐍", "Python",
             "Core language for everything — data, "
             "models and the web app."),
            ("⚡", "XGBoost",
             "AI model trained on 15K real insurance "
             "claims to detect fraud patterns."),
            ("⚖️", "SMOTE",
             "Balances the data — fraud cases are rare "
             "(6%), so we create more fraud examples "
             "for training."),
            ("🔍", "SHAP",
             "Shows exactly why a claim was flagged — "
             "no black box decisions."),
            ("🌲", "Isolation Forest",
             "A second AI that detects unusual claims "
             "by finding ones that look nothing like "
             "normal behaviour."),
            ("📊", "Plotly",
             "Powers all the interactive charts "
             "and gauge visuals."),
            ("🌐", "Streamlit",
             "Turns the Python model into this "
             "live web dashboard."),
        ]
        for icon, name, desc in techs:
            st.markdown(f"""
            <div class='tech-card'>
                <div class='tech-name'>{icon} {name}</div>
                <div class='tech-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── Prediction Function ───────────────────────────────────────
def predict(claim, model, feats, iso):
    df  = pd.DataFrame([claim])
    for c in feats:
        if c not in df: df[c] = 0
    df      = df[feats]
    p       = float(model.predict_proba(df)[0][1])
    ano     = iso.decision_function(df)[0]
    ano_pct = max(0, min(100,
                  round((1 - (ano + 0.5)) * 100, 1)))

    if p < 0.3:
        ver, risk, col = 'LEGITIMATE', 'LOW RISK',    '#10b981'
    elif p < 0.5:
        ver, risk, col = 'SUSPICIOUS', 'MEDIUM RISK', '#f59e0b'
    else:
        ver, risk, col = 'FRAUDULENT', 'HIGH RISK',   '#ef4444'

    return {
        'verdict' : ver,
        'risk'    : risk,
        'color'   : col,
        'fp'      : round(p * 100, 2),
        'lp'      : round((1 - p) * 100, 2),
        'ano'     : ano_pct,
        'df'      : df,
    }

# ── SHAP Explanation ──────────────────────────────────────────
def get_shap(model, df, feats):
    ex   = shap.TreeExplainer(model)
    vals = ex.shap_values(df)
    return pd.DataFrame({
        'Feature' : feats,
        'Value'   : df.values[0],
        'SHAP'    : vals[0],
    }).sort_values('SHAP', key=abs, ascending=False).head(12)

# ── Gauge Chart ───────────────────────────────────────────────
def gauge(pct, color, title):
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct,
        title  = {'text': title,
                  'font': {'size': 13, 'color': '#8892b0'}},
        number = {'suffix': '%',
                  'font'  : {'size': 30, 'color': color}},
        gauge  = {
            'axis'       : {'range': [0, 100],
                            'tickcolor': '#2a1515'},
            'bar'        : {'color': color},
            'bgcolor'    : '#0f1322',
            'bordercolor': '#2a1515',
            'steps'      : [
                {'range': [0,  30], 'color': '#052e16'},
                {'range': [30, 60], 'color': '#1c1200'},
                {'range': [60,100], 'color': '#2d0000'},
            ],
        }
    ))
    fig.update_layout(
        height        = 225,
        margin        = dict(l=20, r=20, t=50, b=10),
        paper_bgcolor = 'rgba(0,0,0,0)',
        font_color    = '#8892b0')
    return fig

# ── SHAP Bar Chart ────────────────────────────────────────────
def shap_fig(shap_df):
    colors = ['#ef4444' if v > 0 else '#10b981'
              for v in shap_df['SHAP']]
    fig = go.Figure(go.Bar(
        x            = shap_df['SHAP'],
        y            = shap_df['Feature'],
        orientation  = 'h',
        marker_color = colors,
        text         = [f"{v:+.3f}" for v in shap_df['SHAP']],
        textposition = 'outside'))
    fig.add_vline(x=0, line_color='#334155', line_width=1.5)
    fig.update_layout(
        title       = {
            'text': '🔍 Why was this claim flagged?',
            'font': {'size': 16, 'color': '#e2e8f0'}},
        height      = 430,
        xaxis_title = 'Impact  '
                      '(Red = increases fraud risk  |  '
                      'Green = reduces fraud risk)',
        yaxis       = dict(autorange='reversed',
                           tickfont=dict(size=12)),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font_color    = '#cbd5e1',
        margin        = dict(l=10, r=90, t=60, b=50),
        xaxis         = dict(gridcolor='#1e2a45',
                             zerolinecolor='#334155'))
    return fig

# ════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════
def main():
    model, feats, iso = load_models()
    render_sidebar()

    # ── Big Left Title ────────────────────────────────────────
    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div class='page-title'>🛡️ IntelliRisk AI</div>
        <div class='page-sub'>
            AI-Powered Insurance Fraud Detection Platform
        </div>
        <div class='badge-row'>
            <span class='badge'>⚡ XGBoost</span>
            <span class='badge'>🔍 SHAP Explainability</span>
            <span class='badge badge-green'>✅ AUC 0.8115</span>
            <span class='badge badge-white'>📁 15K Claims</span>
            <span class='badge badge-orange'>
                🌲 Isolation Forest
            </span>
            <span class='badge badge-green'>
                🟢 Real-Time Detection
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── How It Works ──────────────────────────────────────────
    st.markdown("#### ⚡ How It Works")
    cols  = st.columns(4)
    steps = [
        ("1", "Enter Claim",
         "Fill in the insurance claim details "
         "and vehicle information"),
        ("2", "Dual AI Analysis",
         "XGBoost + Isolation Forest both analyse "
         "the claim independently"),
        ("3", "Fraud Scoring",
         "Calculates fraud probability and anomaly "
         "score in real time"),
        ("4", "Full Explanation",
         "SHAP shows exactly which factors triggered "
         "the fraud flag"),
    ]
    for col, (n, t, d) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-number'>{n}</div>
                <div class='step-title'>{t}</div>
                <div class='step-desc'>{d}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Fraud Analyzer",
        "📊 Fraud Analytics",
        "ℹ️ About",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — FRAUD ANALYZER
    # ════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 📋 Insurance Claim Details")
        st.caption(
            "Enter the claim information below and click "
            "Analyze for an instant fraud assessment.")

        c1, c2, c3 = st.columns(3)

        # ── Column 1: Vehicle & Accident ─────────────────────
        with c1:
            st.markdown(
                "<div class='section-header'>"
                "🚗 Vehicle & Accident</div>",
                unsafe_allow_html=True)
            make = st.selectbox(
                "Vehicle Make",
                [0, 1, 2, 3, 4, 5, 6, 7], index=6,
                format_func=lambda x:
                ['Pontiac', 'Honda', 'Toyota', 'Mazda',
                 'Dodge', 'Chevrolet', 'Ford', 'BMW'][x])
            vcat = st.selectbox(
                "Vehicle Category", [0, 1, 2],
                format_func=lambda x:
                ['Sport', 'Sedan', 'Utility'][x])
            vprice = st.selectbox(
                "Vehicle Price Range",
                [0, 1, 2, 3, 4, 5], index=5,
                format_func=lambda x:
                ['Under $20K', '$20K-$29K', '$30K-$39K',
                 '$40K-$59K', '$60K-$69K', '$70K+'][x])
            area = st.selectbox(
                "Accident Area", [0, 1],
                format_func=lambda x:
                ['Rural', 'Urban'][x])
            age_veh = st.slider(
                "Age of Vehicle", 0, 7, 3,
                help="How old the vehicle is (encoded 0-7)")

        # ── Column 2: Policy & Claim ──────────────────────────
        with c2:
            st.markdown(
                "<div class='section-header'>"
                "📋 Policy & Claim Info</div>",
                unsafe_allow_html=True)
            ptype = st.selectbox(
                "Policy Type", [0, 1, 2, 3, 4, 5],
                format_func=lambda x:
                ['Sport-Collision', 'Sport-All Perils',
                 'Sport-Liability', 'Sedan-Collision',
                 'Sedan-All Perils', 'Sedan-Liability'][x])
            bpol = st.selectbox(
                "Base Policy", [0, 1, 2],
                format_func=lambda x:
                ['Collision', 'All Perils', 'Liability'][x])
            ded = st.selectbox(
                "Deductible ($)", [300, 400, 500, 700])
            drat = st.slider(
                "Driver Rating", 1, 4, 2,
                help="1 = Excellent  |  4 = Poor "
                     "driving history")
            pclaims = st.selectbox(
                "Past Number of Claims", [0, 1, 2, 3],
                format_func=lambda x:
                ['None', '1 Claim', '2 Claims',
                 'More than 2'][x])

        # ── Column 3: Risk Indicators ─────────────────────────
        with c3:
            st.markdown(
                "<div class='section-header'>"
                "🚨 Risk Indicators</div>",
                unsafe_allow_html=True)
            police = st.selectbox(
                "Police Report Filed?", [0, 1],
                format_func=lambda x:
                ['No ⚠️', 'Yes ✅'][x],
                help="No police report = higher fraud signal")
            witness = st.selectbox(
                "Witness Present?", [0, 1],
                format_func=lambda x:
                ['No ⚠️', 'Yes ✅'][x],
                help="No witness = higher fraud signal")
            fault = st.selectbox(
                "Fault", [0, 1],
                format_func=lambda x:
                ['Policy Holder', 'Third Party'][x])
            addr = st.selectbox(
                "Address Changed Before Claim?",
                [0, 1, 2, 3, 4],
                format_func=lambda x:
                ['No Change', 'Under 6 Months',
                 '1 Year', '2-3 Years', '4-8 Years'][x],
                help="Recent address change = suspicious")
            nsuppl = st.selectbox(
                "Number of Supplements",
                [0, 1, 2, 3, 4],
                format_func=lambda x:
                ['None', '1', '2', '3', '4+'][x])
            age_h = st.slider(
                "Age of Policy Holder", 0, 8, 3)

        st.divider()

        # ── Analyze Button ────────────────────────────────────
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            go_btn = st.button(
                "🔍 Analyze Claim for Fraud",
                type               = "primary",
                use_container_width= True)

        # ── Results ───────────────────────────────────────────
        if go_btn:
            nwnp  = 1 if police == 0 and witness == 0 else 0
            claim = {
                'Month'               : 3,
                'WeekOfMonth'         : 2,
                'DayOfWeek'           : 3,
                'Make'                : make,
                'AccidentArea'        : area,
                'DayOfWeekClaimed'    : 3,
                'MonthClaimed'        : 3,
                'WeekOfMonthClaimed'  : 2,
                'Sex'                 : 1,
                'MaritalStatus'       : 1,
                'Age'                 : 25,
                'Fault'               : fault,
                'PolicyType'          : ptype,
                'VehicleCategory'     : vcat,
                'VehiclePrice'        : vprice,
                'Deductible'          : ded,
                'DriverRating'        : drat,
                'Days_Policy_Accident': 3,
                'Days_Policy_Claim'   : 2,
                'PastNumberOfClaims'  : pclaims,
                'AgeOfVehicle'        : age_veh,
                'AgeOfPolicyHolder'   : age_h,
                'PoliceReportFiled'   : police,
                'WitnessPresent'      : witness,
                'AgentType'           : 0,
                'NumberOfSuppliments' : nsuppl,
                'AddressChange_Claim' : addr,
                'NumberOfCars'        : 1,
                'Year'                : 1994,
                'BasePolicy'          : bpol,
                'AGE_RISK'            : 0,
                'MULTI_CAR'           : 0,
                'HIGH_DEDUCTIBLE'     : 1 if ded > 500 else 0,
                'QUICK_CLAIM'         : 0,
                'NO_WITNESS_NO_POLICE': nwnp,
            }
            res = predict(claim, model, feats, iso)

            st.divider()
            st.markdown("## 🔍 Fraud Analysis Results")

            # ── Verdict Banner ────────────────────────────────
            if res['verdict'] == 'LEGITIMATE':
                st.markdown(
                    "<div class='legit-banner'>"
                    "✅ &nbsp; CLAIM APPEARS LEGITIMATE"
                    "</div>",
                    unsafe_allow_html=True)
            elif res['verdict'] == 'SUSPICIOUS':
                st.markdown(
                    "<div class='suspicious-banner'>"
                    "⚠️ &nbsp; SUSPICIOUS — "
                    "MANUAL REVIEW NEEDED"
                    "</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='fraud-banner'>"
                    "❌ &nbsp; HIGH PROBABILITY "
                    "FRAUD DETECTED"
                    "</div>",
                    unsafe_allow_html=True)

            # ── Metric Cards ──────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Verdict",       res['verdict'])
            m2.metric("Risk Level",    res['risk'])
            m3.metric("Fraud Prob",    f"{res['fp']}%")
            m4.metric("Anomaly Score", f"{res['ano']}%")

            st.divider()

            # ── Gauge Charts ──────────────────────────────────
            g1, g2, g3 = st.columns(3)
            with g1:
                st.plotly_chart(
                    gauge(res['fp'], '#ef4444',
                          'Fraud Probability'),
                    use_container_width=True)
            with g2:
                st.plotly_chart(
                    gauge(res['lp'], '#10b981',
                          'Legitimacy Score'),
                    use_container_width=True)
            with g3:
                st.plotly_chart(
                    gauge(res['ano'], '#f59e0b',
                          'Anomaly Score'),
                    use_container_width=True)

            st.divider()

            # ── SHAP Chart ────────────────────────────────────
            with st.spinner(
                    "🔍 Generating AI explanation..."):
                sd = get_shap(model, res['df'], feats)
                st.plotly_chart(
                    shap_fig(sd),
                    use_container_width=True)

            # ── Plain English Explanation ─────────────────────
            st.markdown("### 🤖 AI Fraud Explanation")
            st.caption(
                "The AI breaks down exactly why this "
                "claim was or wasn't flagged.")

            ff = sd[sd['SHAP'] > 0].head(3)
            lf = sd[sd['SHAP'] < 0].head(3)

            e1c, e2c = st.columns(2)
            with e1c:
                st.error("🚨 **Fraud Indicators**")
                for _, row in ff.iterrows():
                    st.markdown(
                        f"• **{row['Feature']}** = "
                        f"`{row['Value']:.2f}` — "
                        f"pushing fraud risk **up** by "
                        f"{row['SHAP']:.3f} points")
            with e2c:
                st.success("✅ **Legitimacy Indicators**")
                for _, row in lf.iterrows():
                    st.markdown(
                        f"• **{row['Feature']}** = "
                        f"`{row['Value']:.2f}` — "
                        f"pushing fraud risk **down** by "
                        f"{abs(row['SHAP']):.3f} points")

    # ════════════════════════════════════════════════════════
    # TAB 2 — FRAUD ANALYTICS
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Fraud Analytics Dashboard")
        st.caption(
            "Insights from 15,420 real Oracle insurance "
            "claims used to train the model.")
        try:
            df = load_data()

            # ── KPI Row ───────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Claims",
                      f"{len(df):,}")
            k2.metric("Fraud Cases",
                      f"{df['FraudFound_P'].sum():,}",
                      f"{df['FraudFound_P'].mean()*100:.1f}%")
            k3.metric("Legitimate",
                      f"{(df['FraudFound_P']==0).sum():,}")
            k4.metric("Detection AUC", "0.8115")

            st.divider()
            a1, a2 = st.columns(2)

            # ── Chart 1: Fraud by Fault ───────────────────────
            with a1:
                fg = df.groupby('Fault')[
                    'FraudFound_P'].mean().reset_index()
                fg['Fault_Label'] = fg['Fault'].map(
                    {0: 'Policy Holder', 1: 'Third Party'})
                f1 = px.bar(
                    fg, x='Fault_Label', y='FraudFound_P',
                    title='Fraud Rate by Fault Type',
                    color='FraudFound_P',
                    color_continuous_scale=[
                        [0,   '#052e16'],
                        [0.5, '#f59e0b'],
                        [1,   '#ef4444']],
                    labels={'FraudFound_P': 'Fraud Rate'})
                f1.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor ='rgba(0,0,0,0)',
                    font_color   ='#cbd5e1')
                st.plotly_chart(f1, use_container_width=True)

            # ── Chart 2: Fraud by Police Report ──────────────
            with a2:
                pg = df.groupby('PoliceReportFiled')[
                    'FraudFound_P'].mean().reset_index()
                pg['Police'] = pg['PoliceReportFiled'].map(
                    {0: 'No Report Filed ⚠️',
                     1: 'Report Filed ✅'})
                f2 = px.bar(
                    pg, x='Police', y='FraudFound_P',
                    title='Fraud Rate: Was Police '
                          'Report Filed?',
                    color='FraudFound_P',
                    color_continuous_scale=[
                        [0,   '#052e16'],
                        [0.5, '#f59e0b'],
                        [1,   '#ef4444']],
                    labels={'FraudFound_P': 'Fraud Rate'})
                f2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor ='rgba(0,0,0,0)',
                    font_color   ='#cbd5e1')
                st.plotly_chart(f2, use_container_width=True)

            # ── Chart 3: Fraud by Vehicle Price ──────────────
            f3 = px.histogram(
                df, x='VehiclePrice',
                color='FraudFound_P',
                title='Fraud Distribution by '
                      'Vehicle Price Range',
                color_discrete_map={
                    0: '#10b981', 1: '#ef4444'},
                barmode='group',
                labels={
                    'FraudFound_P': 'Fraud(1) / Legit(0)'})
            f3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor ='rgba(0,0,0,0)',
                font_color   ='#cbd5e1')
            st.plotly_chart(f3, use_container_width=True)

        except Exception as e:
            st.warning(f"Analytics error: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 3 — ABOUT
    # ════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### ℹ️ About This Platform")
        ab1, ab2 = st.columns([3, 2])

        with ab1:
            st.markdown("""
            #### What is the Fraud Detection Module?
            This module uses two AI models working together
            to detect fraudulent insurance claims in real
            time — the same approach used by major insurers
            like Allstate, Progressive, and Lloyd's of London.

            #### How It Detects Fraud
            - ✅ **XGBoost** — learns fraud patterns from
              15,420 real Oracle insurance claims
            - ✅ **Isolation Forest** — detects claims that
              look completely different from normal behaviour
            - ✅ **SMOTE** — handles the imbalance (only 6%
              of claims are fraud) by creating synthetic
              fraud examples during training
            - ✅ **SHAP** — explains every flag in plain
              English so investigators know where to look

            #### Dataset
            Originally an **Oracle Machine Learning** case
            study — 15,420 vehicle insurance claims with
            33 features covering policy, vehicle, accident,
            and claimant details.
            """)

        with ab2:
            st.markdown("#### Model Comparison")
            pm = pd.DataFrame({
                'Model': ['XGBoost ⭐',
                          'Random Forest',
                          'Logistic Regression'],
                'AUC'  : [0.8115, 0.7965, 0.7915]})
            fp = px.bar(
                pm, x='AUC', y='Model',
                orientation='h',
                title='ROC-AUC Score by Model',
                color='AUC',
                color_continuous_scale=[
                    [0,   '#2d0000'],
                    [0.5, '#f59e0b'],
                    [1,   '#ef4444']],
                range_x=[0.6, 0.85])
            fp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor ='rgba(0,0,0,0)',
                font_color   ='#cbd5e1',
                height=240,
                margin=dict(l=10, r=20, t=40, b=10))
            st.plotly_chart(fp, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
        <div class='footer-title'>🛡️ IntelliRisk AI</div>
        <div class='footer-sub'>
            AI-Powered Insurance Fraud Detection Platform<br>
            Python &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp;
            SHAP &nbsp;·&nbsp; SMOTE &nbsp;·&nbsp;
            Streamlit &nbsp;·&nbsp; Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Entry Point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()