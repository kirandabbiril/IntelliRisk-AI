# 🧠 IntelliRisk AI — Enterprise Risk Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-Powered Loan Approval Intelligence + Insurance Fraud Detection**

*Built with real-world financial data — the same approach used by JPMorgan, Capital One, and modern fintechs*

[🚀 Live Demo](#) · [📊 Features](#features) · [🛠️ Tech Stack](#tech-stack) · [📁 Dataset](#datasets)

</div>

---

## 📌 Project Overview

**IntelliRisk AI** is an enterprise-grade AI platform that combines two powerful financial intelligence modules into a single unified system:

| Module | Description | Model | AUC |
|--------|-------------|-------|-----|
| 🏦 **Loan Approval Intelligence** | Predicts loan default risk and approves/rejects applications | XGBoost | **0.7656** |
| 🛡️ **Insurance Fraud Detection** | Detects fraudulent insurance claims using dual AI | XGBoost + Isolation Forest | **0.8115** |

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

### 🧠 Platform Features
- ✅ Unified single-page application with navigation
- ✅ Professional dark-mode enterprise UI
- ✅ Interactive gauge charts and SHAP bar charts
- ✅ Confusion matrices and feature importance plots
- ✅ Fully explainable — no black box decisions

---

## 🖥️ Screenshots

### 🏠 Landing Page
> Unified platform entry point showing both modules

### 🏦 Loan Approval — Decision Results
> Real-time approval decision with gauge charts and SHAP explanation

### 🛡️ Fraud Detection — Analysis Results
> Fraud probability scoring with anomaly detection and AI explanation

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

## 🗂️ Project Structure