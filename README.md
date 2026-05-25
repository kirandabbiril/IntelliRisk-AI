# 🧠 IntelliRisk AI — Enterprise Risk Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-yellow?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-teal?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-Powered Loan Approval Intelligence + Insurance Fraud Detection + RAG Chatbot**

*Built with real-world financial data — the same approach used by JPMorgan, Capital One, and modern fintechs*

[🚀 Live Demo](#) · [📊 Features](#features) · [🛠️ Tech Stack](#tech-stack) · [📁 Dataset](#datasets)

</div>

---

## 📌 Project Overview

**IntelliRisk AI** is an enterprise-grade AI platform that combines two powerful financial intelligence modules, a RAG-powered AI chatbot, and an analytics dashboard into a single unified system:

| Module | Description | Model | AUC |
|--------|-------------|-------|-----|
| 🏦 **Loan Approval Intelligence** | Predicts loan default risk and approves/rejects applications | XGBoost | **0.7656** |
| 🛡️ **Insurance Fraud Detection** | Detects fraudulent insurance claims using dual AI | XGBoost + Isolation Forest | **0.8115** |
| 🤖 **RAG AI Chatbot** | Answers policy questions using retrieval-augmented generation | Llama 3.3 + ChromaDB | — |
| 📊 **Analytics Dashboard** | Platform-wide KPIs, model comparisons, and trend charts | Plotly | — |

This project demonstrates real-world applications of machine learning in the **banking and insurance industries** — domains actively hiring for Data Analyst, Risk Analyst, Fraud Analyst, Data Scientist, and Data Engineer roles.

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

### 🤖 Module 3 — RAG AI Chatbot
- ✅ **Retrieval-Augmented Generation** — answers grounded in real policy documents
- ✅ **ChromaDB** vector database with 10 policy documents (5 loan + 5 fraud)
- ✅ **Groq Whisper** — voice input, speak your question directly
- ✅ **gTTS** — voice output, answers read aloud automatically
- ✅ **Llama 3.3 70B** via Groq API — fast, free, and accurate
- ✅ Source citations shown for every answer
- ✅ Toggle voice on/off with one click

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

### 🤖 RAG Chatbot — Voice + Text
> Ask questions by voice or text, get grounded answers with source citations

### 📊 Analytics Dashboard
> Platform-wide performance metrics across all 6 models

---

## 🛠️ Tech Stack

| Technology | Role | Why |
|-----------|------|-----|
| **Python 3.10** | Core Language | Data processing, model training, web app |
| **XGBoost** | Primary ML Model | Best AUC performance on both modules |
| **Scikit-Learn** | ML Toolkit | Preprocessing, Random Forest, Logistic Regression, evaluation |
| **SHAP** | Explainable AI | Shows exactly which features drove each prediction |
| **SMOTE** | Class Balancing | Handles 6% fraud rate imbalance by synthesizing minority examples |
| **Isolation Forest** | Anomaly Detection | Detects unusual claims that don't fit normal behaviour patterns |
| **ChromaDB** | Vector Database | Stores and retrieves policy documents for RAG |
| **Sentence Transformers** | Embeddings | Converts text to vectors for semantic search (all-MiniLM-L6-v2) |
| **Llama 3.3 70B** | RAG Language Model | Reads retrieved docs and generates grounded answers via Groq API |
| **Groq Whisper Large v3** | Speech to Text | Transcribes voice questions in real time |
| **gTTS** | Text to Speech | Reads chatbot answers aloud (Google TTS) |
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

## 🤖 RAG Chatbot — How It Works

```
User speaks or types a question
           ↓
Sentence Transformers converts question to vector
           ↓
ChromaDB searches 10 policy documents for most relevant chunks
           ↓
Top 3 documents passed to Llama 3.3 70B (via Groq)
           ↓
Llama reads documents and generates a grounded answer
           ↓
gTTS reads the answer aloud + answer shown in chat with sources
```

**Example questions the chatbot can answer:**
- *"Why would a loan be rejected?"*
- *"What credit score is needed for approval?"*
- *"Why is policy holder fault more suspicious for fraud?"*
- *"How does the anomaly detection score work?"*

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/your-username/intellirisk-ai.git
cd intellirisk-ai

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
│   ├── analytics_dashboard.py   # Analytics dashboard (3 tabs, Plotly)
│   ├── rag_chatbot.py           # RAG chatbot with voice input/output
│   ├── rag_engine.py            # ChromaDB vector store + retrieval
│   └── knowledge_base.py        # 10 policy documents (loan + fraud)
├── Models/
│   ├── loan/                    # Trained loan models + feature names
│   └── fraud/                   # Trained fraud models + isolation forest
├── Data/
│   └── processed/               # Preprocessed datasets
└── README.md
```

---

## 🎯 Who Is This For?

This project is designed to demonstrate skills relevant to:
- **Data Analyst** — data exploration, dashboards, KPI reporting
- **Risk Analyst** — credit risk modeling, fraud detection, SHAP explainability
- **Data Scientist** — ML pipeline, model comparison, feature engineering
- **ML Engineer** — RAG pipeline, vector databases, LLM integration
- **Fintech roles** — real-world banking and insurance datasets

---

<div align="center">

**🧠 IntelliRisk AI** · Built by Kiran Dabbiril

Python · XGBoost · SHAP · SMOTE · ChromaDB · Llama 3.3 · Groq · Streamlit · Plotly

</div>
