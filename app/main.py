# app/main.py
# ============================================================
# INTELLIRISK AI — Unified Platform Entry Point
# Integrates Loan Approval + Insurance Fraud Detection
# FIX 1: Monthly annuity converted to annual before model input
# FIX 2: Rejection threshold lowered to 0.15
# ============================================================
from analytics_dashboard import render_analytics_dashboard
from rag_chatbot import render_rag_chatbot
from model_service import (
    load_loan_model, load_fraud_models, load_loan_data, load_fraud_data,
    predict_loan, predict_fraud, get_shap,
)
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, warnings
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# ── Path Setup ───────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "IntelliRisk AI — Enterprise Platform",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Session State ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #07090f; }
.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1528 50%, #0a0e1a 100%);
    border: 1px solid #1e2a45; border-radius: 24px;
    padding: 60px 40px; text-align: center;
    margin-bottom: 40px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60%; left: -60%;
    width: 220%; height: 220%;
    background: radial-gradient(circle, rgba(99,102,241,0.04) 0%, transparent 65%);
    pointer-events: none;
}
.hero-logo { font-size: 64px; margin-bottom: 16px; display: block; }
.hero-title {
    font-size: 58px; font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 40%, #22d3ee 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin: 0 0 14px 0;
}
.hero-sub {
    color: #8892b0; font-size: 18px; font-weight: 300;
    margin-bottom: 30px; line-height: 1.6; max-width: 600px;
    margin-left: auto; margin-right: auto;
}
.hero-badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.hbadge {
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc; padding: 5px 16px; border-radius: 20px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
}
.hbadge-green { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #34d399; }
.hbadge-gold  { background: rgba(240,192,64,0.12);  border: 1px solid rgba(240,192,64,0.3);  color: #f0c040; }
.hbadge-red   { background: rgba(239,68,68,0.12);   border: 1px solid rgba(239,68,68,0.3);   color: #fca5a5; }
.module-card {
    background: linear-gradient(135deg, #0f1322, #141829); border: 1px solid #1e2a45;
    border-radius: 20px; padding: 36px 30px; text-align: center;
    transition: all 0.3s ease; cursor: pointer; height: 100%; position: relative; overflow: hidden;
}
.module-card:hover { border-color: #6366f1; transform: translateY(-4px); box-shadow: 0 20px 40px rgba(99,102,241,0.15); }
.module-card-loan  { border-top: 3px solid #f0c040; }
.module-card-fraud { border-top: 3px solid #ef4444; }
.module-icon { font-size: 52px; margin-bottom: 16px; display: block; }
.module-title { font-size: 24px; font-weight: 800; margin-bottom: 10px; }
.module-title-loan  { color: #f0c040; }
.module-title-fraud { color: #ef4444; }
.module-desc { color: #8892b0; font-size: 14px; line-height: 1.6; margin-bottom: 20px; }
.module-stats { display: flex; justify-content: center; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.mstat { background: rgba(255,255,255,0.04); border: 1px solid #1e2a45; border-radius: 8px; padding: 8px 14px; text-align: center; }
.mstat-val   { font-size: 16px; font-weight: 700; color: #e2e8f0; }
.mstat-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }
.stat-card { background: linear-gradient(135deg, #0f1322, #141829); border: 1px solid #1e2a45; border-radius: 14px; padding: 24px; text-align: center; }
.stat-number {
    font-size: 36px; font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.stat-label { color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.feature-card { background: linear-gradient(135deg, #0f1322, #141829); border: 1px solid #1e2a45; border-radius: 14px; padding: 22px; height: 100%; }
.feature-icon  { font-size: 28px; margin-bottom: 10px; }
.feature-title { color: #e2e8f0; font-weight: 700; font-size: 15px; margin-bottom: 8px; }
.feature-desc  { color: #64748b; font-size: 13px; line-height: 1.55; }
.page-title-loan {
    font-size: 48px; font-weight: 800;
    background: linear-gradient(135deg, #f0c040 0%, #f5d76e 50%, #ffffff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin: 0 0 6px 0;
}
.page-title-fraud {
    font-size: 48px; font-weight: 800;
    background: linear-gradient(135deg, #ef4444 0%, #f97316 50%, #ffffff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin: 0 0 6px 0;
}
.page-sub { color: #8892b0; font-size: 15px; font-weight: 300; margin-bottom: 16px; }
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.badge-gold   { background: rgba(240,192,64,0.12); border: 1px solid rgba(240,192,64,0.3); color: #f0c040; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-red    { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #fca5a5; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-white  { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15); color: #cbd5e1; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-green  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #34d399; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.badge-orange { background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.3); color: #fdba74; padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.step-card { background: linear-gradient(135deg, #0f1322, #141829); border-radius: 14px; padding: 22px 18px; text-align: center; height: 100%; }
.step-number-gold { background: linear-gradient(135deg, #f0c040, #f5d76e); color: #07090f; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 15px; margin: 0 auto 12px auto; }
.step-number-red  { background: linear-gradient(135deg, #ef4444, #f97316); color: white;   width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 15px; margin: 0 auto 12px auto; }
.step-title { color: #e2e8f0; font-weight: 700; font-size: 14px; margin-bottom: 7px; }
.step-desc  { color: #64748b; font-size: 12px; line-height: 1.55; }
.section-header-gold { font-size: 11px; font-weight: 700; color: #f0c040; text-transform: uppercase; letter-spacing: 1.8px; border-bottom: 1px solid #1e2a45; padding-bottom: 7px; margin: 18px 0 13px 0; }
.section-header-red  { font-size: 11px; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 1.8px; border-bottom: 1px solid #2a1515; padding-bottom: 7px; margin: 18px 0 13px 0; }
.approved-banner    { background: linear-gradient(135deg, #052e16, #064e3b); border: 2px solid #10b981; border-radius: 16px; padding: 26px; text-align: center; font-size: 30px; font-weight: 800; color: #10b981; margin: 12px 0; letter-spacing: 2px; }
.rejected-banner    { background: linear-gradient(135deg, #2d0000, #450a0a); border: 2px solid #ef4444; border-radius: 16px; padding: 26px; text-align: center; font-size: 30px; font-weight: 800; color: #ef4444; margin: 12px 0; letter-spacing: 2px; }
.fraud-banner       { background: linear-gradient(135deg, #2d0000, #450a0a); border: 2px solid #ef4444; border-radius: 16px; padding: 26px; text-align: center; font-size: 30px; font-weight: 800; color: #ef4444; margin: 12px 0; letter-spacing: 2px; }
.legit-banner       { background: linear-gradient(135deg, #052e16, #064e3b); border: 2px solid #10b981; border-radius: 16px; padding: 26px; text-align: center; font-size: 30px; font-weight: 800; color: #10b981; margin: 12px 0; letter-spacing: 2px; }
.suspicious-banner  { background: linear-gradient(135deg, #1c1200, #2d1f00); border: 2px solid #f59e0b; border-radius: 16px; padding: 26px; text-align: center; font-size: 30px; font-weight: 800; color: #f59e0b; margin: 12px 0; letter-spacing: 2px; }
.sb-metric-gold { background: linear-gradient(135deg, #0f1322, #141829); border: 1px solid #1e2a45; border-radius: 10px; padding: 14px 12px; margin-bottom: 9px; text-align: center; }
.sb-metric-red  { background: linear-gradient(135deg, #0f1322, #141829); border: 1px solid #2a1515; border-radius: 10px; padding: 14px 12px; margin-bottom: 9px; text-align: center; }
.sb-label { color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; }
.footer { background: linear-gradient(135deg, #0b0e1a, #0f1322); border: 1px solid #1e2a45; border-radius: 14px; padding: 22px; text-align: center; margin-top: 40px; }
.footer-title { font-weight: 800; font-size: 17px; margin-bottom: 6px; }
.footer-sub { color: #64748b; font-size: 12px; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# ── MODEL LOADERS / SCORING ──────────────────────────────────
#   (moved to model_service.py — shared with the agentic chatbot)
# ════════════════════════════════════════════════════════════

def gauge(pct, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        title={'text':title,'font':{'size':13,'color':'#8892b0'}},
        number={'suffix':'%','font':{'size':30,'color':color}},
        gauge={
            'axis'       :{'range':[0,100],'tickcolor':'#1e2a45'},
            'bar'        :{'color':color},
            'bgcolor'    :'#0f1322',
            'bordercolor':'#1e2a45',
            'steps'      :[
                {'range':[0,30], 'color':'#052e16'},
                {'range':[30,60],'color':'#422006'},
                {'range':[60,100],'color':'#2d0000'},
            ]}))
    fig.update_layout(height=225,margin=dict(l=20,r=20,t=50,b=10),
                      paper_bgcolor='rgba(0,0,0,0)',font_color='#8892b0')
    return fig

def shap_chart(shap_df, title):
    colors = ['#ef4444' if v>0 else '#10b981' for v in shap_df['SHAP']]
    fig = go.Figure(go.Bar(
        x=shap_df['SHAP'],y=shap_df['Feature'],
        orientation='h',marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df['SHAP']],
        textposition='outside'))
    fig.add_vline(x=0,line_color='#334155',line_width=1.5)
    fig.update_layout(
        title={'text':title,'font':{'size':16,'color':'#e2e8f0'}},
        height=430,
        xaxis_title='Impact  (Red = increases risk  |  Green = reduces risk)',
        yaxis=dict(autorange='reversed',tickfont=dict(size=12)),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',margin=dict(l=10,r=90,t=60,b=50),
        xaxis=dict(gridcolor='#1e2a45',zerolinecolor='#334155'))
    return fig

def render_shap_section(model, df, feats, title_chart, title_section, flag_color, flag_label, support_label):
    with st.spinner("🔍 Generating AI explanation..."):
        sd, shap_ok = get_shap(model, df, feats)

    if not shap_ok:
        st.info("ℹ️ SHAP explainability is not available in this deployment environment. "
                "The prediction above is fully accurate — only the explanation chart is unavailable.")
        return

    st.plotly_chart(shap_chart(sd, title_chart), use_container_width=True)
    st.markdown(f"### 🤖 {title_section}")
    risk_factors    = sd[sd['SHAP']>0].head(3)
    support_factors = sd[sd['SHAP']<0].head(3)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.error(f"{'⚠️' if 'Approval' in flag_label else '🚨'} **{flag_label}**")
        for _, row in risk_factors.iterrows():
            st.markdown(f"• **{row['Feature']}** = `{row['Value']:.2f}` — "
                        f"pushing risk **up** by {row['SHAP']:.3f} points")
    with ec2:
        st.success(f"✅ **{support_label}**")
        for _, row in support_factors.iterrows():
            st.markdown(f"• **{row['Feature']}** = `{row['Value']:.2f}` — "
                        f"pushing risk **down** by {abs(row['SHAP']):.3f} points")

# ════════════════════════════════════════════════════════════
# ── SIDEBAR ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:18px 0 10px 0;'>
            <div style='font-size:40px;'>🧠</div>
            <div style='font-size:19px;font-weight:800;
                        background:linear-gradient(135deg,#a5b4fc,#22d3ee);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;margin-top:6px;'>
                IntelliRisk AI
            </div>
            <div style='font-size:11px;color:#64748b;margin-top:3px;'>
                Enterprise Risk Platform
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown("**🧭 Navigation**")
        if st.button("🏠 Home",            use_container_width=True, type="secondary"):
            st.session_state.page = 'home';    st.rerun()
        if st.button("🏦 Loan Approval Module",  use_container_width=True, type="secondary"):
            st.session_state.page = 'loan';    st.rerun()
        if st.button("🛡️ Fraud Detection Module",use_container_width=True, type="secondary"):
            st.session_state.page = 'fraud';   st.rerun()
        if st.button("📊 Analytics Dashboard",   use_container_width=True, type="secondary"):
            st.session_state.page = 'analytics';st.rerun()
        if st.button("🤖 AI Agent (Tools + RAG)", use_container_width=True, type="secondary"):
            st.session_state.page = 'chatbot'; st.rerun()
        st.divider()
        st.markdown("**📊 Platform Overview**")
        for label, value in [
            ("Modules","2 Active"),("Loan AUC","0.7656"),
            ("Fraud AUC","0.8115"),("Total Records","322,931"),
            ("Models Trained","6 Models"),("XAI","SHAP ✅"),
        ]:
            st.markdown(f"""
            <div class='sb-metric-gold'>
                <div class='sb-label'>{label}</div>
                <div style='color:#a5b4fc;font-size:18px;font-weight:700;margin-top:3px;'>{value}</div>
            </div>""", unsafe_allow_html=True)
        st.divider()
        page_names = {'home':'🏠 Home','loan':'🏦 Loan Module','fraud':'🛡️ Fraud Module',
                      'analytics':'📊 Analytics','chatbot':'🤖 AI Agent'}
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                    border-radius:8px;padding:10px;text-align:center;'>
            <div style='color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>
                Currently Viewing</div>
            <div style='color:#a5b4fc;font-weight:700;font-size:13px;margin-top:4px;'>
                {page_names.get(st.session_state.page,'🏠 Home')}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# ── HOME PAGE ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════
def render_home():
    st.markdown("""
    <div class='hero'>
        <span class='hero-logo'>🧠</span>
        <div class='hero-title'>IntelliRisk AI</div>
        <div class='hero-sub'>
            Enterprise-Grade AI Platform for Financial Risk Intelligence<br>
            Loan Approval Prediction · Insurance Fraud Detection · Explainable AI
        </div>
        <div class='hero-badges'>
            <span class='hbadge'>⚡ XGBoost</span>
            <span class='hbadge'>🔍 SHAP Explainability</span>
            <span class='hbadge hbadge-green'>🌲 Isolation Forest</span>
            <span class='hbadge hbadge-gold'>🏦 307K Loan Records</span>
            <span class='hbadge hbadge-red'>🛡️ 15K Fraud Claims</span>
            <span class='hbadge hbadge-green'>🟢 Real-Time AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    s1,s2,s3,s4,s5 = st.columns(5)
    for col,(num,label) in zip([s1,s2,s3,s4,s5],[
        ("322,931","Total Records"),("6","ML Models"),
        ("0.8115","Best AUC"),("2","AI Modules"),("100%","Explainable")]):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-number'>{num}</div>"
                        f"<div class='stat-label'>{label}</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🚀 Choose a Module to Get Started")
    st.caption("Select a module below to launch the AI platform.")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div class='module-card module-card-loan'>
            <span class='module-icon'>🏦</span>
            <div class='module-title module-title-loan'>Loan Approval Intelligence</div>
            <div class='module-desc'>AI-powered loan approval prediction using real-world data from 307,511 Home Credit borrowers. Predicts default risk, assigns risk categories, and explains decisions in plain English using SHAP.</div>
            <div class='module-stats'>
                <div class='mstat'><div class='mstat-val'>0.7656</div><div class='mstat-label'>ROC-AUC</div></div>
                <div class='mstat'><div class='mstat-val'>307K</div><div class='mstat-label'>Records</div></div>
                <div class='mstat'><div class='mstat-val'>34</div><div class='mstat-label'>Features</div></div>
                <div class='mstat'><div class='mstat-val'>3</div><div class='mstat-label'>Models</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏦 Launch Loan Module →", key="loan_btn", type="primary", use_container_width=True):
            st.session_state.page = 'loan'; st.rerun()
    with mc2:
        st.markdown("""
        <div class='module-card module-card-fraud'>
            <span class='module-icon'>🛡️</span>
            <div class='module-title module-title-fraud'>Insurance Fraud Detection</div>
            <div class='module-desc'>Dual AI system using XGBoost + Isolation Forest to detect fraudulent insurance claims. Originally an Oracle ML case study — trained on 15,420 real vehicle insurance claims with SMOTE balancing and SHAP explanations.</div>
            <div class='module-stats'>
                <div class='mstat'><div class='mstat-val'>0.8115</div><div class='mstat-label'>ROC-AUC</div></div>
                <div class='mstat'><div class='mstat-val'>15K</div><div class='mstat-label'>Claims</div></div>
                <div class='mstat'><div class='mstat-val'>36</div><div class='mstat-label'>Features</div></div>
                <div class='mstat'><div class='mstat-val'>4</div><div class='mstat-label'>Models</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🛡️ Launch Fraud Module →", key="fraud_btn", type="primary", use_container_width=True):
            st.session_state.page = 'fraud'; st.rerun()

    st.divider()
    st.markdown("### ✨ Platform Features")
    f1,f2,f3,f4 = st.columns(4)
    for col,(icon,title,desc) in zip([f1,f2,f3,f4],[
        ("🔍","Explainable AI","SHAP values show exactly why every decision was made — no black box."),
        ("⚡","Real-Time Predictions","Instant AI decisions on any new applicant or claim in milliseconds."),
        ("📊","Interactive Analytics","Full portfolio dashboards with charts, trends, and KPIs."),
        ("🤖","Agentic AI Analyst","Ask about policy, live portfolio stats, or what-if scenarios — an LLM agent picks the right tool (policy RAG, live SQL, or the trained models) automatically."),
    ]):
        with col:
            st.markdown(f"<div class='feature-card'><div class='feature-icon'>{icon}</div>"
                        f"<div class='feature-title'>{title}</div>"
                        f"<div class='feature-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚡ How IntelliRisk AI Works")
    w1,w2,w3,w4 = st.columns(4)
    for col,(n,t,d) in zip([w1,w2,w3,w4],[
        ("1","Real Data","Trained on 307K+ real-world financial records from Home Credit & Oracle"),
        ("2","Smart Models","XGBoost, Random Forest, Logistic Regression and Isolation Forest"),
        ("3","AI Decisions","Instant predictions with probability scores and risk categories"),
        ("4","Clear Reasons","SHAP explains every decision — just like a real bank officer would"),
    ]):
        with col:
            st.markdown(f"""
            <div class='feature-card' style='text-align:center;'>
                <div style='background:linear-gradient(135deg,#6366f1,#22d3ee);color:white;
                    width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
                    justify-content:center;font-weight:800;font-size:15px;margin:0 auto 12px auto;'>{n}</div>
                <div class='feature-title'>{t}</div>
                <div class='feature-desc'>{d}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer' style='margin-top:40px;'>
        <div class='footer-title' style='background:linear-gradient(135deg,#a5b4fc,#22d3ee);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
            🧠 IntelliRisk AI</div>
        <div class='footer-sub'>Enterprise AI Platform — Loan Intelligence &nbsp;·&nbsp; Fraud Detection &nbsp;·&nbsp; Agentic Analyst<br>
            Python &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; SHAP &nbsp;·&nbsp; SMOTE &nbsp;·&nbsp;
            Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; DuckDB &nbsp;·&nbsp; Llama 3.3 (Groq)</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# ── LOAN MODULE PAGE ─────────────────────────────────────────
# ════════════════════════════════════════════════════════════
def render_loan():
    model, feats = load_loan_model()
    if st.button("← Back to Home", key="loan_back"):
        st.session_state.page = 'home'; st.rerun()

    st.markdown("""
    <div style='padding:10px 0 20px 0;'>
        <div class='page-title-loan'>🏦 Loan Approval Intelligence</div>
        <div class='page-sub'>AI-Powered Loan Approval Decision Platform</div>
        <div class='badge-row'>
            <span class='badge-gold'>⚡ XGBoost</span>
            <span class='badge-gold'>🔍 SHAP</span>
            <span class='badge-green'>✅ AUC 0.7656</span>
            <span class='badge-white'>📁 307K Records</span>
            <span class='badge-green'>🟢 Real-Time</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### ⚡ How It Works")
    cols = st.columns(4)
    for col,(n,t,d) in zip(cols,[
        ("1","Enter Details","Fill in the applicant's financial and personal information"),
        ("2","AI Analysis","XGBoost analyzes against 307K real borrower patterns"),
        ("3","Risk Scoring","Calculates default probability and assigns risk category"),
        ("4","Plain English","SHAP shows exactly which factors drove the decision"),
    ]):
        with col:
            st.markdown(f"<div class='step-card' style='border:1px solid #1e2a45;'>"
                        f"<div class='step-number-gold'>{n}</div>"
                        f"<div class='step-title'>{t}</div>"
                        f"<div class='step-desc'>{d}</div></div>", unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🎯 Loan Predictor","📊 Analytics","ℹ️ About"])

    with tab1:
        st.markdown("### 📋 Applicant Information")
        st.caption("Fill in the details below and click Analyze for an instant AI decision.")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("<div class='section-header-gold'>💰 Financial Details</div>", unsafe_allow_html=True)
            income  = st.number_input("Annual Income ($)",10000,1000000,75000,5000)
            credit  = st.number_input("Loan Amount ($)",10000,2000000,200000,10000)
            annuity = st.number_input("Monthly Payment ($)",500,50000,5000,500,
                                      help="Your expected monthly repayment amount")
            goods   = st.number_input("Asset Value ($)",5000,2000000,180000,5000)
        with c2:
            st.markdown("<div class='section-header-gold'>📈 Credit Profile</div>", unsafe_allow_html=True)
            e1 = st.slider("Credit Score — Bureau A",0.0,1.0,0.5,0.01)
            e2 = st.slider("Credit Score — Bureau B",0.0,1.0,0.6,0.01)
            e3 = st.slider("Credit Score — Bureau C",0.0,1.0,0.55,0.01)
            st.markdown("<div class='section-header-gold'>👤 Personal Info</div>", unsafe_allow_html=True)
            age = st.slider("Age",18,70,35)
            emp = st.slider("Years at Current Job",0.0,40.0,5.0,0.5)
        with c3:
            st.markdown("<div class='section-header-gold'>🏠 Assets & Risk</div>", unsafe_allow_html=True)
            car    = st.selectbox("Owns a Car?",["Yes","No"])
            realty = st.selectbox("Owns Property?",["Yes","No"])
            kids   = st.number_input("No. of Children",0,10,0)
            fam    = st.number_input("Family Members",1,15,2)
            d30    = st.number_input("Social Defaults — 30 Day",0,10,0)
            d60    = st.number_input("Social Defaults — 60 Day",0,10,0)

        st.divider()
        _,mid,_ = st.columns([1,2,1])
        with mid:
            go_btn = st.button("🚀 Analyze Loan Application", type="primary", use_container_width=True, key="loan_analyze")

        if go_btn:
            # ── FIX 1: Convert monthly payment → annual for model ──
            annual_annuity = annuity * 12

            app = {
                'AMT_INCOME_TOTAL'        : income,
                'AMT_CREDIT'              : credit,
                'AMT_ANNUITY'             : annual_annuity,        # FIX 1 applied
                'AMT_GOODS_PRICE'         : goods,
                'EXT_SOURCE_1'            : e1,
                'EXT_SOURCE_2'            : e2,
                'EXT_SOURCE_3'            : e3,
                'AGE_YEARS'               : age,
                'EMPLOYED_YEARS'          : emp,
                'CREDIT_INCOME_RATIO'     : credit/(income+1),
                'ANNUITY_INCOME_RATIO'    : annual_annuity/(income+1),  # FIX 1 applied
                'CREDIT_TERM'             : annual_annuity/(credit+1),  # FIX 1 applied
                'GOODS_CREDIT_RATIO'      : goods/(credit+1),
                'EXT_SOURCE_MEAN'         : np.mean([e1,e2,e3]),
                'CNT_CHILDREN'            : kids,
                'CNT_FAM_MEMBERS'         : fam,
                'FLAG_OWN_CAR'            : 1 if car=="Yes" else 0,
                'FLAG_OWN_REALTY'         : 1 if realty=="Yes" else 0,
                'DEF_30_CNT_SOCIAL_CIRCLE': d30,
                'DEF_60_CNT_SOCIAL_CIRCLE': d60,
            }
            res = predict_loan(app, model, feats)
            st.divider()
            st.markdown("## 📊 Decision Results")
            if res['decision'] == 'APPROVED':
                st.markdown("<div class='approved-banner'>✅ &nbsp; LOAN APPROVED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='rejected-banner'>❌ &nbsp; LOAN REJECTED</div>", unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Decision",res['decision'])
            m2.metric("Risk Level",res['risk'])
            m3.metric("Approval Chance",f"{res['ap']}%")
            m4.metric("Default Risk",f"{res['dp']}%")

            st.divider()
            g1,g2,g3 = st.columns(3)
            with g1: st.plotly_chart(gauge(res['ap'],'#10b981','Approval Chance'),use_container_width=True)
            with g2: st.plotly_chart(gauge(res['dp'],'#ef4444','Default Risk'),use_container_width=True)
            with g3: st.plotly_chart(gauge(res['dp'],res['color'],'Risk Score'),use_container_width=True)

            st.divider()
            render_shap_section(
                model, res['df'], feats,
                '🔍 Why did the AI make this decision?',
                'AI Decision Explanation',
                '#ef4444', 'Factors Working Against Approval',
                'Factors Supporting Approval'
            )

    with tab2:
        st.markdown("### 📊 Portfolio Analytics")
        try:
            df = load_loan_data()
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Records",f"{len(df):,}")
            k2.metric("Approved",f"{(df['TARGET']==0).sum():,}",f"{(df['TARGET']==0).mean()*100:.1f}%")
            k3.metric("Defaulted",f"{(df['TARGET']==1).sum():,}",f"-{(df['TARGET']==1).mean()*100:.1f}%")
            k4.metric("Avg Loan",f"${df['AMT_CREDIT'].mean():,.0f}")
            st.divider()
            a1,a2 = st.columns(2)
            with a1:
                df['Age Group'] = pd.cut(df['AGE_YEARS'],bins=[18,25,35,45,55,100],labels=['18-25','26-35','36-45','46-55','55+'])
                ag = df.groupby('Age Group',observed=True)['TARGET'].mean().reset_index()
                f1 = px.bar(ag,x='Age Group',y='TARGET',title='Default Rate by Age Group',color='TARGET',color_continuous_scale=[[0,'#052e16'],[0.5,'#f59e0b'],[1,'#ef4444']],labels={'TARGET':'Default Rate'})
                f1.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
                st.plotly_chart(f1,use_container_width=True)
            with a2:
                f2 = px.histogram(df,x='AMT_CREDIT',color='TARGET',nbins=50,title='Loan Amount Distribution',color_discrete_map={0:'#10b981',1:'#ef4444'},labels={'TARGET':'0=Repaid/1=Default'})
                f2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
                st.plotly_chart(f2,use_container_width=True)
            samp = df.sample(min(3000,len(df)),random_state=42)
            f3 = px.scatter(samp,x='AMT_INCOME_TOTAL',y='AMT_CREDIT',color=samp['TARGET'].map({0:'Repaid ✅',1:'Defaulted ❌'}),title='Income vs Loan — Who Repays vs Defaults?',color_discrete_map={'Repaid ✅':'#10b981','Defaulted ❌':'#ef4444'},opacity=0.45)
            f3.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
            st.plotly_chart(f3,use_container_width=True)
        except Exception as e:
            st.warning(f"Analytics error: {e}")

    with tab3:
        st.markdown("### ℹ️ About Loan Module")
        ab1,ab2 = st.columns([3,2])
        with ab1:
            st.markdown("""
            #### What is the Loan Approval Module?
            Predicts loan approval decisions using real-world data from 307,511 Home Credit borrowers.
            #### Key Features
            - ✅ Real-time decisions in milliseconds
            - ✅ 34 financial signals analysed
            - ✅ Plain English explanation via SHAP
            - ✅ Risk categories: Low / Medium / High
            - ✅ Portfolio analytics dashboard
            #### Dataset
            **307,511 real loan applications** from Home Credit International.
            """)
        with ab2:
            pm = pd.DataFrame({'Model':['XGBoost ⭐','Random Forest','Logistic Regression'],'AUC':[0.7656,0.7506,0.6406]})
            fp = px.bar(pm,x='AUC',y='Model',orientation='h',title='Model Comparison',color='AUC',color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#f59e0b'],[1,'#f0c040']],range_x=[0.5,0.85])
            fp.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1',height=240,margin=dict(l=10,r=20,t=40,b=10))
            st.plotly_chart(fp,use_container_width=True)

# ════════════════════════════════════════════════════════════
# ── FRAUD MODULE PAGE ────────────────────────────────────────
# ════════════════════════════════════════════════════════════
def render_fraud():
    model, feats, iso = load_fraud_models()
    if st.button("← Back to Home", key="fraud_back"):
        st.session_state.page = 'home'; st.rerun()

    st.markdown("""
    <div style='padding:10px 0 20px 0;'>
        <div class='page-title-fraud'>🛡️ Insurance Fraud Detection</div>
        <div class='page-sub'>AI-Powered Insurance Fraud Detection Platform</div>
        <div class='badge-row'>
            <span class='badge-red'>⚡ XGBoost</span>
            <span class='badge-red'>🔍 SHAP</span>
            <span class='badge-green'>✅ AUC 0.8115</span>
            <span class='badge-white'>📁 15K Claims</span>
            <span class='badge-orange'>🌲 Isolation Forest</span>
            <span class='badge-green'>🟢 Real-Time</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### ⚡ How It Works")
    cols = st.columns(4)
    for col,(n,t,d) in zip(cols,[
        ("1","Enter Claim","Fill in the insurance claim details and vehicle information"),
        ("2","Dual AI","XGBoost + Isolation Forest both analyse the claim independently"),
        ("3","Fraud Scoring","Calculates fraud probability and anomaly score in real time"),
        ("4","Explanation","SHAP shows exactly which factors triggered the fraud flag"),
    ]):
        with col:
            st.markdown(f"<div class='step-card' style='border:1px solid #2a1515;'>"
                        f"<div class='step-number-red'>{n}</div>"
                        f"<div class='step-title'>{t}</div>"
                        f"<div class='step-desc'>{d}</div></div>", unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🔍 Fraud Analyzer","📊 Analytics","ℹ️ About"])

    with tab1:
        st.markdown("### 📋 Insurance Claim Details")
        st.caption("Enter claim information and click Analyze for an instant fraud assessment.")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("<div class='section-header-red'>🚗 Vehicle & Accident</div>", unsafe_allow_html=True)
            make    = st.selectbox("Vehicle Make",[0,1,2,3,4,5,6,7],index=6,format_func=lambda x:['Pontiac','Honda','Toyota','Mazda','Dodge','Chevrolet','Ford','BMW'][x])
            vcat    = st.selectbox("Vehicle Category",[0,1,2],format_func=lambda x:['Sport','Sedan','Utility'][x])
            vprice  = st.selectbox("Vehicle Price Range",[0,1,2,3,4,5],index=5,format_func=lambda x:['Under $20K','$20K-$29K','$30K-$39K','$40K-$59K','$60K-$69K','$70K+'][x])
            area    = st.selectbox("Accident Area",[0,1],format_func=lambda x:['Rural','Urban'][x])
            age_veh = st.slider("Age of Vehicle",0,7,3)
        with c2:
            st.markdown("<div class='section-header-red'>📋 Policy & Claim</div>", unsafe_allow_html=True)
            ptype   = st.selectbox("Policy Type",[0,1,2,3,4,5],format_func=lambda x:['Sport-Collision','Sport-All Perils','Sport-Liability','Sedan-Collision','Sedan-All Perils','Sedan-Liability'][x])
            bpol    = st.selectbox("Base Policy",[0,1,2],format_func=lambda x:['Collision','All Perils','Liability'][x])
            ded     = st.selectbox("Deductible ($)",[300,400,500,700])
            drat    = st.slider("Driver Rating",1,4,2)
            pclaims = st.selectbox("Past Claims",[0,1,2,3],format_func=lambda x:['None','1','2','More than 2'][x])
        with c3:
            st.markdown("<div class='section-header-red'>🚨 Risk Indicators</div>", unsafe_allow_html=True)
            police  = st.selectbox("Police Report Filed?",[0,1],format_func=lambda x:['No ⚠️','Yes ✅'][x])
            witness = st.selectbox("Witness Present?",[0,1],format_func=lambda x:['No ⚠️','Yes ✅'][x])
            fault   = st.selectbox("Fault",[0,1],format_func=lambda x:['Policy Holder','Third Party'][x])
            addr    = st.selectbox("Address Changed?",[0,1,2,3,4],format_func=lambda x:['No Change','<6 Months','1 Year','2-3 Years','4-8 Years'][x])
            nsuppl  = st.selectbox("Supplements",[0,1,2,3,4],format_func=lambda x:['None','1','2','3','4+'][x])
            age_h   = st.slider("Age of Policy Holder",0,8,3)

        st.divider()
        _,mid,_ = st.columns([1,2,1])
        with mid:
            go_btn = st.button("🔍 Analyze Claim for Fraud", type="primary", use_container_width=True, key="fraud_analyze")

        if go_btn:
            nwnp  = 1 if police==0 and witness==0 else 0
            claim = {
                'Month':3,'WeekOfMonth':2,'DayOfWeek':3,'Make':make,'AccidentArea':area,
                'DayOfWeekClaimed':3,'MonthClaimed':3,'WeekOfMonthClaimed':2,'Sex':1,
                'MaritalStatus':1,'Age':25,'Fault':fault,'PolicyType':ptype,
                'VehicleCategory':vcat,'VehiclePrice':vprice,'Deductible':ded,
                'DriverRating':drat,'Days_Policy_Accident':3,'Days_Policy_Claim':2,
                'PastNumberOfClaims':pclaims,'AgeOfVehicle':age_veh,
                'AgeOfPolicyHolder':age_h,'PoliceReportFiled':police,
                'WitnessPresent':witness,'AgentType':0,'NumberOfSuppliments':nsuppl,
                'AddressChange_Claim':addr,'NumberOfCars':1,'Year':1994,'BasePolicy':bpol,
                'AGE_RISK':0,'MULTI_CAR':0,'HIGH_DEDUCTIBLE':1 if ded>500 else 0,
                'QUICK_CLAIM':0,'NO_WITNESS_NO_POLICE':nwnp,
            }
            res = predict_fraud(claim, model, feats, iso)
            st.divider()
            st.markdown("## 🔍 Fraud Analysis Results")
            if res['verdict'] == 'LEGITIMATE':
                st.markdown("<div class='legit-banner'>✅ &nbsp; CLAIM APPEARS LEGITIMATE</div>", unsafe_allow_html=True)
            elif res['verdict'] == 'SUSPICIOUS':
                st.markdown("<div class='suspicious-banner'>⚠️ &nbsp; SUSPICIOUS — MANUAL REVIEW NEEDED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='fraud-banner'>❌ &nbsp; HIGH PROBABILITY FRAUD DETECTED</div>", unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Verdict",res['verdict'])
            m2.metric("Risk Level",res['risk'])
            m3.metric("Fraud Prob",f"{res['fp']}%")
            m4.metric("Anomaly Score",f"{res['ano']}%")

            st.divider()
            g1,g2,g3 = st.columns(3)
            with g1: st.plotly_chart(gauge(res['fp'],'#ef4444','Fraud Probability'),use_container_width=True)
            with g2: st.plotly_chart(gauge(res['lp'],'#10b981','Legitimacy Score'),use_container_width=True)
            with g3: st.plotly_chart(gauge(res['ano'],'#f59e0b','Anomaly Score'),use_container_width=True)

            st.divider()
            render_shap_section(
                model, res['df'], feats,
                '🔍 Why was this claim flagged?',
                'AI Fraud Explanation',
                '#ef4444', 'Fraud Indicators',
                'Legitimacy Indicators'
            )

    with tab2:
        st.markdown("### 📊 Fraud Analytics")
        try:
            df = load_fraud_data()
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Claims",f"{len(df):,}")
            k2.metric("Fraud Cases",f"{df['FraudFound_P'].sum():,}",f"{df['FraudFound_P'].mean()*100:.1f}%")
            k3.metric("Legitimate",f"{(df['FraudFound_P']==0).sum():,}")
            k4.metric("AUC Score","0.8115")
            st.divider()
            a1,a2 = st.columns(2)
            with a1:
                fg = df.groupby('Fault')['FraudFound_P'].mean().reset_index()
                fg['Fault_Label'] = fg['Fault'].map({0:'Policy Holder',1:'Third Party'})
                f1 = px.bar(fg,x='Fault_Label',y='FraudFound_P',title='Fraud Rate by Fault Type',color='FraudFound_P',color_continuous_scale=[[0,'#052e16'],[0.5,'#f59e0b'],[1,'#ef4444']],labels={'FraudFound_P':'Fraud Rate'})
                f1.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
                st.plotly_chart(f1,use_container_width=True)
            with a2:
                pg = df.groupby('PoliceReportFiled')['FraudFound_P'].mean().reset_index()
                pg['Police'] = pg['PoliceReportFiled'].map({0:'No Report ⚠️',1:'Filed ✅'})
                f2 = px.bar(pg,x='Police',y='FraudFound_P',title='Fraud Rate: Police Report?',color='FraudFound_P',color_continuous_scale=[[0,'#052e16'],[0.5,'#f59e0b'],[1,'#ef4444']],labels={'FraudFound_P':'Fraud Rate'})
                f2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
                st.plotly_chart(f2,use_container_width=True)
            f3 = px.histogram(df,x='VehiclePrice',color='FraudFound_P',title='Fraud by Vehicle Price Range',color_discrete_map={0:'#10b981',1:'#ef4444'},barmode='group',labels={'FraudFound_P':'Fraud/Legit'})
            f3.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
            st.plotly_chart(f3,use_container_width=True)
        except Exception as e:
            st.warning(f"Analytics error: {e}")

    with tab3:
        st.markdown("### ℹ️ About Fraud Module")
        ab1,ab2 = st.columns([3,2])
        with ab1:
            st.markdown("""
            #### What is the Fraud Detection Module?
            Uses two AI models together to detect fraudulent insurance claims in real time.
            #### How It Works
            - ✅ **XGBoost** — learns from 15,420 claims
            - ✅ **Isolation Forest** — anomaly detection
            - ✅ **SMOTE** — handles 6% fraud imbalance
            - ✅ **SHAP** — plain English explanations
            #### Dataset
            **Oracle Machine Learning** case study — 15,420 vehicle insurance claims with 33 features.
            """)
        with ab2:
            pm = pd.DataFrame({'Model':['XGBoost ⭐','Random Forest','Logistic Regression'],'AUC':[0.8115,0.7965,0.7915]})
            fp = px.bar(pm,x='AUC',y='Model',orientation='h',title='Model Comparison',color='AUC',color_continuous_scale=[[0,'#2d0000'],[0.5,'#f59e0b'],[1,'#ef4444']],range_x=[0.6,0.85])
            fp.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1',height=240,margin=dict(l=10,r=20,t=40,b=10))
            st.plotly_chart(fp,use_container_width=True)

# ════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════
def main():
    render_sidebar()
    if st.session_state.page == 'home':
        render_home()
    elif st.session_state.page == 'loan':
        render_loan()
    elif st.session_state.page == 'fraud':
        render_fraud()
    elif st.session_state.page == 'analytics':
        render_analytics_dashboard()
    elif st.session_state.page == 'chatbot':
        render_rag_chatbot()

if __name__ == '__main__':
    main()