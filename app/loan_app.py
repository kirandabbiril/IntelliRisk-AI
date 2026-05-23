# app/loan_app.py
# ============================================================
# MODULE 1 — Loan Approval Intelligence
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

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'Models', 'loan')
DATA_DIR   = os.path.join(BASE_DIR, 'Data', 'processed')
sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title = "IntelliRisk AI — Loan Intelligence",
    page_icon  = "🏦",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #07090f; }

.page-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #f0c040 0%, #f5d76e 50%, #ffffff 100%);
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
    background: rgba(240,192,64,0.12);
    border: 1px solid rgba(240,192,64,0.3);
    color: #f0c040;
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

.step-card {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #1e2a45;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s;
}
.step-number {
    background: linear-gradient(135deg, #f0c040, #f5d76e);
    color: #07090f;
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px;
    margin: 0 auto 12px auto;
}
.step-title { color: #e2e8f0; font-weight: 700; font-size: 14px; margin-bottom: 7px; }
.step-desc  { color: #64748b; font-size: 12px; line-height: 1.55; }

.section-header {
    font-size: 11px; font-weight: 700;
    color: #f0c040; text-transform: uppercase;
    letter-spacing: 1.8px;
    border-bottom: 1px solid #1e2a45;
    padding-bottom: 7px; margin: 18px 0 13px 0;
}

.approved-banner {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #10b981; border-radius: 16px;
    padding: 26px; text-align: center;
    font-size: 30px; font-weight: 800;
    color: #10b981; margin: 12px 0; letter-spacing: 2px;
}
.rejected-banner {
    background: linear-gradient(135deg, #2d0000, #450a0a);
    border: 2px solid #ef4444; border-radius: 16px;
    padding: 26px; text-align: center;
    font-size: 30px; font-weight: 800;
    color: #ef4444; margin: 12px 0; letter-spacing: 2px;
}

/* Sidebar */
.sb-metric {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #1e2a45; border-radius: 10px;
    padding: 14px 12px; margin-bottom: 9px; text-align: center;
}
.sb-label { color: #64748b; font-size: 10px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px; }
.sb-value { color: #f0c040; font-size: 21px; font-weight: 700; margin-top: 3px; }

.tech-card {
    background: linear-gradient(135deg, #0f1322, #141829);
    border: 1px solid #1e2a45; border-radius: 11px;
    padding: 15px; margin-bottom: 10px;
}
.tech-name { font-size: 14px; font-weight: 700; color: #f0c040; margin-bottom: 3px; }
.tech-role { font-size: 10px; font-weight: 700; color: #64748b;
             text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
.tech-desc { font-size: 12px; color: #8892b0; line-height: 1.5; }

.footer {
    background: linear-gradient(135deg, #0b0e1a, #0f1322);
    border: 1px solid #1e2a45; border-radius: 14px;
    padding: 22px; text-align: center; margin-top: 40px;
}
.footer-title { color: #f0c040; font-weight: 800; font-size: 17px; margin-bottom: 6px; }
.footer-sub { color: #64748b; font-size: 12px; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load(os.path.join(MODELS_DIR,'best_loan_model.pkl'))
    f = joblib.load(os.path.join(MODELS_DIR,'feature_names.pkl'))
    return m, f

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR,'loan_processed.csv'))

# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:18px 0 10px 0;'>
            <div style='font-size:40px;'>🏦</div>
            <div style='font-size:19px;font-weight:800;color:#f0c040;
                        margin-top:6px;'>IntelliRisk AI</div>
            <div style='font-size:11px;color:#64748b;margin-top:3px;'>
                Loan Intelligence Module
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**📊 Model Performance**")
        for label, value in [
            ("Best Model",    "XGBoost"),
            ("ROC-AUC Score", "0.7656"),
            ("Training Data", "307,511"),
            ("Features Used", "34"),
            ("Test Accuracy", "73%"),
        ]:
            st.markdown(f"""
            <div class='sb-metric'>
                <div class='sb-label'>{label}</div>
                <div class='sb-value'>{value}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**💻 Tech Stack**")
        techs = [
            ("🐍", "Python",
             "Core language for data processing, model training and the web app."),
            ("⚡", "XGBoost",
             "AI model that learned patterns from 307K borrowers to predict default risk."),
            ("🔍", "SHAP",
             "Makes AI transparent — shows which factors caused each decision."),
            ("📊", "Plotly",
             "Creates interactive charts and gauge visualizations."),
            ("🌐", "Streamlit",
             "Turns Python code into this web dashboard instantly."),
            ("🗂️", "Pandas",
             "Loads and cleans the raw loan dataset for the AI to process."),
        ]
        for icon, name, desc in techs:
            st.markdown(f"""
            <div class='tech-card'>
                <div class='tech-name'>{icon} {name}</div>
                <div class='tech-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
def predict_loan(app, model, feats):
    df = pd.DataFrame([app])
    for c in feats:
        if c not in df: df[c] = 0
    df = df[feats]
    p  = float(model.predict_proba(df)[0][1])
    if p < 0.15:   dec,risk,col='APPROVED','LOW RISK','#10b981'
    elif p < 0.25: dec,risk,col='APPROVED','MEDIUM RISK','#f59e0b'
    else:         dec,risk,col='REJECTED','HIGH RISK','#ef4444'
    return {'decision':dec,'risk':risk,'color':col,
            'ap':round((1-p)*100,1),
            'dp':round(p*100,1),'df':df}

def get_shap(model, df, feats):
    ex   = shap.TreeExplainer(model)
    vals = ex.shap_values(df)
    return pd.DataFrame({'Feature':feats,'Value':df.values[0],
                         'SHAP':vals[0]}
    ).sort_values('SHAP',key=abs,ascending=False).head(12)

def gauge(pct, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        title={'text':title,'font':{'size':13,'color':'#8892b0'}},
        number={'suffix':'%','font':{'size':30,'color':color}},
        gauge={'axis':{'range':[0,100],'tickcolor':'#1e2a45'},
               'bar':{'color':color},'bgcolor':'#0f1322',
               'bordercolor':'#1e2a45',
               'steps':[{'range':[0,30],'color':'#052e16'},
                        {'range':[30,60],'color':'#422006'},
                        {'range':[60,100],'color':'#2d0000'}]}))
    fig.update_layout(height=225,margin=dict(l=20,r=20,t=50,b=10),
                      paper_bgcolor='rgba(0,0,0,0)',font_color='#8892b0')
    return fig

def shap_fig(shap_df):
    colors = ['#ef4444' if v>0 else '#10b981' for v in shap_df['SHAP']]
    fig = go.Figure(go.Bar(
        x=shap_df['SHAP'], y=shap_df['Feature'],
        orientation='h', marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df['SHAP']],
        textposition='outside'))
    fig.add_vline(x=0, line_color='#334155', line_width=1.5)
    fig.update_layout(
        title={'text':'🔍 Why did the AI make this decision?',
               'font':{'size':16,'color':'#e2e8f0'}},
        height=430,
        xaxis_title='Impact  (Red = increases risk  |  Green = reduces risk)',
        yaxis=dict(autorange='reversed',tickfont=dict(size=12)),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',margin=dict(l=10,r=90,t=60,b=50),
        xaxis=dict(gridcolor='#1e2a45',zerolinecolor='#334155'))
    return fig

# ════════════════════════════════════════════════════════════
def main():
    model, feats = load_model()
    render_sidebar()

    # ── Big Left Title ────────────────────────────────────────
    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div class='page-title'>🏦 IntelliRisk AI</div>
        <div class='page-sub'>
            AI-Powered Loan Approval Intelligence Platform
        </div>
        <div class='badge-row'>
            <span class='badge'>⚡ XGBoost</span>
            <span class='badge'>🔍 SHAP Explainability</span>
            <span class='badge badge-green'>✅ AUC 0.7656</span>
            <span class='badge badge-white'>📁 307K Records</span>
            <span class='badge badge-green'>🟢 Real-Time Predictions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── How It Works ──────────────────────────────────────────
    st.markdown("#### ⚡ How It Works")
    cols = st.columns(4)
    steps = [
        ("1","Enter Details",
         "Fill in the applicant's financial and personal information"),
        ("2","AI Analysis",
         "XGBoost analyzes the profile against 307K real borrower patterns"),
        ("3","Risk Scoring",
         "Calculates default probability and assigns a risk category"),
        ("4","Plain English",
         "SHAP shows exactly which factors drove the decision — no black box"),
    ]
    for col,(n,t,d) in zip(cols,steps):
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
        "🎯 Loan Predictor",
        "📊 Analytics Dashboard",
        "ℹ️ About",
    ])

    # ════ TAB 1 ══════════════════════════════════════════════
    with tab1:
        st.markdown("### 📋 Applicant Information")
        st.caption("Fill in the details below and click "
                   "Analyze for an instant AI-powered decision.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='section-header'>"
                        "💰 Financial Details</div>",
                        unsafe_allow_html=True)
            income    = st.number_input("Annual Income ($)",
                            10000,1000000,75000,5000,
                            help="Total yearly income before tax")
            credit    = st.number_input("Loan Amount ($)",
                            10000,2000000,200000,10000,
                            help="Total loan amount requested")
            annuity   = st.number_input("Monthly Payment ($)",
                            1000,100000,15000,500,
                            help="Expected monthly repayment")
            goods     = st.number_input("Asset Value ($)",
                            5000,2000000,180000,5000,
                            help="Value of asset being purchased")

        with c2:
            st.markdown("<div class='section-header'>"
                        "📈 Credit Profile</div>",
                        unsafe_allow_html=True)
            e1 = st.slider("Credit Score — Bureau A",0.0,1.0,0.5,0.01,
                           help="Score from 1st credit bureau. Higher = better")
            e2 = st.slider("Credit Score — Bureau B",0.0,1.0,0.6,0.01,
                           help="Score from 2nd credit bureau")
            e3 = st.slider("Credit Score — Bureau C",0.0,1.0,0.55,0.01,
                           help="Score from 3rd credit bureau")
            st.markdown("<div class='section-header'>"
                        "👤 Personal Info</div>",
                        unsafe_allow_html=True)
            age  = st.slider("Age",18,70,35,
                             help="Applicant's age in years")
            emp  = st.slider("Years at Current Job",0.0,40.0,5.0,0.5,
                             help="Employment stability indicator")

        with c3:
            st.markdown("<div class='section-header'>"
                        "🏠 Assets & Risk Flags</div>",
                        unsafe_allow_html=True)
            car    = st.selectbox("Owns a Car?",["Yes","No"])
            realty = st.selectbox("Owns Property?",["Yes","No"])
            kids   = st.number_input("No. of Children",0,10,0)
            fam    = st.number_input("Family Members",1,15,2)
            d30    = st.number_input(
                "Social Defaults — 30 Day",0,10,0,
                help="How many people in their circle "
                     "defaulted in 30 days")
            d60    = st.number_input(
                "Social Defaults — 60 Day",0,10,0,
                help="How many people in their circle "
                     "defaulted in 60 days")

        st.divider()
        _, mid, _ = st.columns([1,2,1])
        with mid:
            go_btn = st.button("🚀 Analyze Loan Application",
                               type="primary",
                               use_container_width=True)

        if go_btn:
            app = {
                'AMT_INCOME_TOTAL'        : income,
                'AMT_CREDIT'              : credit,
                'AMT_ANNUITY'             : annuity,
                'AMT_GOODS_PRICE'         : goods,
                'EXT_SOURCE_1'            : e1,
                'EXT_SOURCE_2'            : e2,
                'EXT_SOURCE_3'            : e3,
                'AGE_YEARS'               : age,
                'EMPLOYED_YEARS'          : emp,
                'CREDIT_INCOME_RATIO'     : credit/(income+1),
                'ANNUITY_INCOME_RATIO'    : annuity/(income+1),
                'CREDIT_TERM'             : annuity/(credit+1),
                'GOODS_CREDIT_RATIO'      : goods/(credit+1),
                'EXT_SOURCE_MEAN'         : np.mean([e1,e2,e3]),
                'CNT_CHILDREN'            : kids,
                'CNT_FAM_MEMBERS'         : fam,
                'FLAG_OWN_CAR'            : 1 if car=="Yes" else 0,
                'FLAG_OWN_REALTY'         : 1 if realty=="Yes" else 0,
                'DEF_30_CNT_SOCIAL_CIRCLE': d30,
                'DEF_60_CNT_SOCIAL_CIRCLE': d60,
            }
            res = predict(app, model, feats)

            st.divider()
            st.markdown("## 📊 Decision Results")

            if res['decision'] == 'APPROVED':
                st.markdown(
                    "<div class='approved-banner'>"
                    "✅ &nbsp; LOAN APPROVED</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='rejected-banner'>"
                    "❌ &nbsp; LOAN REJECTED</div>",
                    unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Decision",      res['decision'])
            m2.metric("Risk Level",    res['risk'])
            m3.metric("Approval Chance",f"{res['ap']}%")
            m4.metric("Default Risk",  f"{res['dp']}%")

            st.divider()
            g1,g2,g3 = st.columns(3)
            with g1:
                st.plotly_chart(gauge(res['ap'],'#10b981',
                    'Approval Chance'),use_container_width=True)
            with g2:
                st.plotly_chart(gauge(res['dp'],'#ef4444',
                    'Default Risk'),use_container_width=True)
            with g3:
                st.plotly_chart(gauge(res['dp'],res['color'],
                    'Overall Risk Score'),use_container_width=True)

            st.divider()
            with st.spinner("🔍 Generating AI explanation..."):
                sd = get_shap(model, res['df'], feats)
                st.plotly_chart(shap_fig(sd),
                                use_container_width=True)

            st.markdown("### 🤖 AI Decision Explanation")
            st.caption("The AI breaks down its reasoning "
                       "in plain English below.")
            rf = sd[sd['SHAP']>0].head(3)
            pf = sd[sd['SHAP']<0].head(3)
            e1c, e2c = st.columns(2)
            with e1c:
                st.error("⚠️ **Factors Working Against Approval**")
                for _,row in rf.iterrows():
                    st.markdown(
                        f"• **{row['Feature']}** = `{row['Value']:.2f}`"
                        f" — pushing risk **up** by "
                        f"{row['SHAP']:.3f} points")
            with e2c:
                st.success("✅ **Factors Supporting Approval**")
                for _,row in pf.iterrows():
                    st.markdown(
                        f"• **{row['Feature']}** = `{row['Value']:.2f}`"
                        f" — pushing risk **down** by "
                        f"{abs(row['SHAP']):.3f} points")

    # ════ TAB 2 ══════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Portfolio Analytics")
        st.caption("Insights from the 307,511 borrower "
                   "records used to train the model.")
        try:
            df = load_data()
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Records",   f"{len(df):,}")
            k2.metric("Approved",
                      f"{(df['TARGET']==0).sum():,}",
                      f"{(df['TARGET']==0).mean()*100:.1f}%")
            k3.metric("Defaulted",
                      f"{(df['TARGET']==1).sum():,}",
                      f"-{(df['TARGET']==1).mean()*100:.1f}%")
            k4.metric("Avg Loan",
                      f"${df['AMT_CREDIT'].mean():,.0f}")

            st.divider()
            a1,a2 = st.columns(2)
            with a1:
                df['Age Group'] = pd.cut(df['AGE_YEARS'],
                    bins=[18,25,35,45,55,100],
                    labels=['18-25','26-35','36-45','46-55','55+'])
                ag = df.groupby('Age Group',observed=True)[
                    'TARGET'].mean().reset_index()
                f1 = px.bar(ag,x='Age Group',y='TARGET',
                    title='Default Rate by Age Group',
                    color='TARGET',
                    color_continuous_scale=[
                        [0,'#052e16'],[0.5,'#f59e0b'],[1,'#ef4444']],
                    labels={'TARGET':'Default Rate'})
                f1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1')
                st.plotly_chart(f1,use_container_width=True)
            with a2:
                f2 = px.histogram(df,x='AMT_CREDIT',
                    color='TARGET',nbins=50,
                    title='Loan Amount — Who Repays vs Defaults',
                    color_discrete_map={0:'#10b981',1:'#ef4444'},
                    labels={'TARGET':'0=Repaid / 1=Default'})
                f2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1')
                st.plotly_chart(f2,use_container_width=True)

            samp = df.sample(min(3000,len(df)),random_state=42)
            f3 = px.scatter(samp,
                x='AMT_INCOME_TOTAL',y='AMT_CREDIT',
                color=samp['TARGET'].map(
                    {0:'Repaid ✅',1:'Defaulted ❌'}),
                title='Income vs Loan Amount — Repaid vs Defaulted',
                color_discrete_map={
                    'Repaid ✅':'#10b981','Defaulted ❌':'#ef4444'},
                opacity=0.45)
            f3.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',font_color='#cbd5e1')
            st.plotly_chart(f3,use_container_width=True)
        except Exception as e:
            st.warning(f"Analytics error: {e}")

    # ════ TAB 3 ══════════════════════════════════════════════
    with tab3:
        st.markdown("### ℹ️ About IntelliRisk AI")
        ab1,ab2 = st.columns([3,2])
        with ab1:
            st.markdown("""
            #### What is IntelliRisk AI?
            IntelliRisk AI is an enterprise-grade machine learning
            platform that predicts loan approval decisions using
            real-world financial data — the same approach used by
            JPMorgan, Capital One, and modern fintech companies.

            #### The Problem It Solves
            Traditional loan approval is slow, inconsistent, and
            hard to explain to customers. IntelliRisk AI:
            - ✅ Makes decisions in **milliseconds**
            - ✅ Analyses **34 financial signals** simultaneously
            - ✅ Explains every decision in **plain English**
            - ✅ Flags **high-risk applicants** with clear reasons
            - ✅ Runs **real-time predictions** on any new applicant

            #### Dataset
            Trained on **307,511 real loan applications** from
            Home Credit International — used in an official
            Kaggle competition with a **$70,000 prize pool**.
            """)
        with ab2:
            st.markdown("#### Model Comparison")
            pm = pd.DataFrame({
                'Model':['XGBoost ⭐','Random Forest',
                         'Logistic Regression'],
                'AUC'  :[0.7656,0.7506,0.6406]})
            fp = px.bar(pm,x='AUC',y='Model',orientation='h',
                title='ROC-AUC Score by Model',
                color='AUC',
                color_continuous_scale=[
                    [0,'#1e3a5f'],[0.5,'#f59e0b'],[1,'#f0c040']],
                range_x=[0.5,0.85])
            fp.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#cbd5e1',height=240,
                margin=dict(l=10,r=20,t=40,b=10))
            st.plotly_chart(fp,use_container_width=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
        <div class='footer-title'>🏦 IntelliRisk AI</div>
        <div class='footer-sub'>
            AI-Powered Loan Approval Intelligence Platform<br>
            Python &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp;
            SHAP &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()