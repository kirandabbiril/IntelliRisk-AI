# 🧠 IntelliRisk AI — Enterprise Risk Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-yellow?style=for-the-badge)
![DuckDB](https://img.shields.io/badge/DuckDB-Agentic_SQL-teal?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-Powered Loan Approval Intelligence + Insurance Fraud Detection + Agentic AI Analyst**

*Built with real-world financial data — the same approach used by JPMorgan, Capital One, and modern fintechs*

[🚀 Live Demo](#) · [📊 Features](#-features) · [🛠️ Tech Stack](#️-tech-stack) · [📁 Dataset](#-datasets)

</div>

---

## 📌 Project Overview

**IntelliRisk AI** is an enterprise-grade AI platform that combines two production ML modules, an agentic AI analyst, and a live analytics dashboard into a single unified system:

| Module | Description | Model | AUC |
|--------|-------------|-------|-----|
| 🏦 **Loan Approval Intelligence** | Predicts loan default risk and approves/rejects applications | XGBoost | **0.7656** |
| 🛡️ **Insurance Fraud Detection** | Detects fraudulent insurance claims using dual AI | XGBoost + Isolation Forest | **0.8115** |
| 🤖 **Agentic AI Analyst** | Routes questions to policy RAG, live SQL analytics, or the trained models | Llama 3.3 (Groq) + tool calling | — |
| 📊 **Analytics Dashboard** | Platform-wide KPIs, model comparisons, and trend charts | Plotly | — |

This project demonstrates real-world applications of machine learning **and agentic AI** in the **banking and insurance industries** — domains actively hiring for Data Analyst, Business Analyst, Risk Analyst, Fraud Analyst, and Data Scientist roles.

---

## ✨ Features

### 🏦 Module 1 — Loan Approval Intelligence
- ✅ Real-time loan approval prediction
- ✅ Default probability scoring (Low / Medium / High risk)
- ✅ **Explainable AI** — SHAP shows *why* each decision was made
- ✅ 34 engineered features including credit scores, income ratios, employment history
- ✅ Trained on **307,511 real borrower records** from Home Credit International
- ✅ Three models compared: XGBoost, Random Forest, Logistic Regression
- ✅ Interactive portfolio analytics dashboard

### 🛡️ Module 2 — Insurance Fraud Detection
- ✅ Real-time fraud probability scoring
- ✅ **Dual AI system** — XGBoost + Isolation Forest working together
- ✅ Anomaly detection using Isolation Forest
- ✅ **SMOTE** — handles 6% fraud class imbalance
- ✅ SHAP explainability — plain English fraud explanations
- ✅ Trained on **15,420 real Oracle insurance claims**
- ✅ Fraud pattern analytics dashboard

### 🤖 Module 3 — Agentic AI Analyst
Instead of a single fixed RAG pipeline, an LLM (Llama 3.3 via Groq) is given **four tools** and decides which one(s) a question actually needs:
- ✅ **`search_policy_docs`** — retrieval over the loan/fraud policy knowledge base (credit score bands, DTI thresholds, rejection reasons)
- ✅ **`query_portfolio_data`** — text-to-SQL: the LLM writes a SQL query, executed read-only against the real 307,511-row loan portfolio and 15,420-row fraud dataset via **DuckDB**
- ✅ **`score_loan_applicant`** / **`score_fraud_claim`** — runs the actual trained XGBoost models live on a hypothetical applicant/claim described in the conversation
- ✅ **SQL safety guardrails** — only SELECT/WITH statements execute; DDL/DML keywords, multi-statement injection, and unbounded row counts are all blocked before a query ever runs
- ✅ **Transparent tool trace** — every answer shows exactly which tool(s) fired, the SQL that ran (with results), or the model output used
- ✅ **Voice in, voice out** — Groq Whisper transcribes spoken questions; gTTS reads answers aloud

### 📊 Module 4 — Analytics Dashboard
- ✅ 3-tab interactive dashboard (Overview, Loan, Fraud)
- ✅ KPI cards with real project metrics
- ✅ AUC comparison across all 6 trained models
- ✅ Class balance charts, SHAP feature importance, fault type analysis
- ✅ Platform data split and tech stack overview

### 🧠 Platform Features
- ✅ Unified single-page application with sidebar navigation
- ✅ Professional dark-mode enterprise UI
- ✅ Interactive gauge charts and SHAP bar charts
- ✅ Fully explainable — no black box decisions

---

## 🖥️ Screenshots

### 🏠 Landing Page
> Unified platform entry point showing all modules

### 🏦 Loan Approval — Decision Results
> Real-time approval decision with gauge charts and SHAP explanation

### 🛡️ Fraud Detection — Analysis Results
> Fraud probability scoring with anomaly detection and AI explanation

### 🤖 Agentic AI Analyst — Voice + Text
> Ask policy, data, or what-if questions; the agent shows exactly which tool(s) it used, including the live SQL it ran

### 📊 Analytics Dashboard
> Platform-wide performance metrics across all 6 models

---

## 🛠️ Tech Stack

| Technology | Role | Why |
|-----------|------|-----|
| **Python 3.11** | Core Language | Data processing, model training, web app |
| **XGBoost** | Primary ML Model | Best AUC performance on both modules |
| **Scikit-Learn** | ML Toolkit | Preprocessing, Random Forest, Logistic Regression, evaluation |
| **SHAP** | Explainable AI | Shows exactly which features drove each prediction |
| **SMOTE** | Class Balancing | Handles 6% fraud rate imbalance by synthesizing minority examples |
| **Isolation Forest** | Anomaly Detection | Detects unusual claims that don't fit normal behaviour patterns |
| **Llama 3.3 70B (Groq)** | Agent + RAG LLM | Decides which tool(s) to call, writes SQL, and generates grounded answers |
| **DuckDB** | Agentic SQL Engine | Executes LLM-generated, read-only SQL against the real portfolio data |
| **Groq Whisper Large v3** | Speech to Text | Transcribes voice questions in real time |
| **gTTS** | Text to Speech | Reads agent answers aloud (Google TTS) |
| **Streamlit** | Web Dashboard | Turns Python models into interactive web apps |
| **Plotly** | Visualizations | Interactive gauge charts, bar charts, scatter plots |
| **Pandas & NumPy** | Data Engineering | Loading, cleaning, and feature engineering on raw datasets |
| **Joblib** | Model Persistence | Saving and loading trained models without retraining |
| **Matplotlib & Seaborn** | Static Plots | Confusion matrices and feature importance plots |

---

## 📁 Datasets

### Module 1 — Home Credit Default Risk
- **Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- **Records:** 307,511 loan applications
- **Features:** 122 original → 34 engineered
- **Target:** Default (1) vs Repaid (0)
- **Context:** Used in an official Kaggle competition with a **$70,000 prize pool**

### Module 2 — Oracle Vehicle Insurance Fraud
- **Source:** [Kaggle — Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- **Records:** 15,420 insurance claims
- **Features:** 33 original → 36 engineered
- **Target:** Fraud (1) vs Legitimate (0)
- **Context:** Originally an **Oracle Machine Learning** real-world case study

---

## 📊 Model Performance

### 🏦 Loan Approval Module

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| **XGBoost ⭐** | **0.7656** | **73%** |
| Random Forest | 0.7506 | 73% |
| Logistic Regression | 0.6406 | 59% |

### 🛡️ Fraud Detection Module

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| **XGBoost ⭐** | **0.8115** | **Best overall** |
| Random Forest | 0.7965 | Strong recall |
| Logistic Regression | 0.7915 | Good baseline |
| Isolation Forest | — | Anomaly scoring |

---

## 🤖 Agentic AI Analyst — How It Works

```
User asks a question (voice or text)
           ↓
Llama 3.3 (via Groq) reads the question and decides which tool(s) it needs
           ↓
   ┌────────────────────┬─────────────────────────┬───────────────────────────┐
   │  search_policy_docs │  query_portfolio_data   │  score_loan_applicant /   │
   │  (RAG over policy   │  (LLM writes SQL, run   │  score_fraud_claim        │
   │   knowledge base)   │  read-only via DuckDB   │  (runs the real trained   │
   │                     │  against 322K records)  │  models on a scenario)    │
   └────────────────────┴─────────────────────────┴───────────────────────────┘
           ↓
Tool result(s) fed back to the model — it may call more tools before answering
           ↓
Final answer generated; UI shows a "Tools used" panel with the SQL + result rows,
policy sources, or model output that backed the answer
           ↓
gTTS reads the answer aloud (optional)
```

**Example questions the agent can answer:**
- *"Why would a loan be rejected?"* → policy search
- *"What percent of loans default when the applicant owns no property?"* → live SQL over the real data
- *"Would a 22-year-old earning $18k asking for a $400k loan be approved?"* → live model scoring
- *"What's the fraud rate for third-party fault claims vs policy-holder fault?"* → live SQL over the real data

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/kirandabbiril/IntelliRisk-AI.git
cd IntelliRisk-AI

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="gsk_your-key-here"

# Run the app
streamlit run app/main.py
```

---

## 📁 Project Structure

```
IntelliRisk AI/
├── app/
│   ├── main.py                  # Unified platform entry point + router
│   ├── model_service.py         # Shared model loading + scoring (used by UI and agent)
│   ├── data_tools.py            # DuckDB safe-SQL layer for the agent's text-to-SQL tool
│   ├── agent_tools.py           # Tool schemas + dispatcher (policy RAG, SQL, scoring)
│   ├── rag_chatbot.py           # Agentic tool-calling loop + voice input/output
│   ├── rag_engine.py            # TF-IDF retrieval over the policy knowledge base
│   ├── knowledge_base.py        # Policy documents (loan + fraud)
│   ├── analytics_dashboard.py   # Analytics dashboard (3 tabs, Plotly)
│   └── shap_patch.py            # SHAP compatibility patch
├── Models/
│   ├── loan/                    # Trained loan models + feature names
│   └── fraud/                   # Trained fraud models + isolation forest
├── Data/
│   └── processed/               # Preprocessed datasets (loan + fraud)
├── src/
│   ├── loan/                    # Loan preprocessing + training pipeline
│   └── fraud/                   # Fraud preprocessing + training pipeline
├── requirements.txt
└── README.md
```

---

## 🎯 Who Is This For?

This project is designed to demonstrate skills relevant to:
- **Data Analyst / Business Analyst** — natural-language analytics over real data, live SQL generation with auditable results, translating business questions into concrete numbers
- **Risk Analyst** — credit risk modeling, fraud detection, SHAP explainability
- **Data Scientist** — ML pipeline, model comparison, feature engineering
- **AI/ML Engineer** — agentic tool-calling architecture, LLM orchestration, RAG
- **Fintech roles** — real-world banking and insurance datasets

---

<div align="center">

**🧠 IntelliRisk AI** · Built by Kiran Dabbiril

Python · XGBoost · SHAP · SMOTE · DuckDB · Llama 3.3 (Groq) · Agentic Tool-Calling · Streamlit · Plotly

</div>
