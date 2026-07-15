# app/agent_tools.py
# ============================================================
# Tool definitions for the IntelliRisk AI agentic chatbot.
#
# Three tools, three different jobs — this is the "agentic" part:
# an LLM decides WHICH of these to call (zero, one, or several)
# for a given question, instead of always doing one fixed RAG step.
#
#   1. search_policy_docs    -> existing TF-IDF RAG over policy text
#   2. query_portfolio_data  -> text-to-SQL over the real 322K-row data
#   3. score_loan_applicant  / score_fraud_claim
#                             -> runs the real trained models live
#
# Tool schemas follow the OpenAI/Groq function-calling format.
# ============================================================
import numpy as np

from rag_engine import retrieve_relevant_docs
from data_tools import run_sql, SCHEMA_DESCRIPTION
from model_service import (
    load_loan_model, load_fraud_models, predict_loan, predict_fraud,
)

# ── Category maps (ground-truth encodings from the raw datasets) ──
FRAUD_MAKE_MAP = {
    'Accura': 0, 'BMW': 1, 'Chevrolet': 2, 'Dodge': 3, 'Ferrari': 4, 'Ford': 5,
    'Honda': 6, 'Jaguar': 7, 'Lexus': 8, 'Mazda': 9, 'Mecedes': 10, 'Mercury': 11,
    'Nisson': 12, 'Pontiac': 13, 'Porche': 14, 'Saab': 15, 'Saturn': 16,
    'Toyota': 17, 'VW': 18,
}
FRAUD_AREA_MAP        = {'Rural': 0, 'Urban': 1}
FRAUD_POLICY_TYPE_MAP = {
    'Sedan - All Perils': 0, 'Sedan - Collision': 1, 'Sedan - Liability': 2,
    'Sport - All Perils': 3, 'Sport - Collision': 4, 'Sport - Liability': 5,
    'Utility - All Perils': 6, 'Utility - Collision': 7, 'Utility - Liability': 8,
}
FRAUD_VEHICLE_CATEGORY_MAP = {'Sedan': 0, 'Sport': 1, 'Utility': 2}
FRAUD_VEHICLE_PRICE_MAP = {
    '20000 to 29000': 0, '30000 to 39000': 1, '40000 to 59000': 2,
    '60000 to 69000': 3, 'less than 20000': 4, 'more than 69000': 5,
}
FRAUD_YES_NO_MAP     = {'No': 0, 'Yes': 1}
FRAUD_FAULT_MAP      = {'Policy Holder': 0, 'Third Party': 1}
FRAUD_BASE_POLICY_MAP = {'All Perils': 0, 'Collision': 1, 'Liability': 2}


# ════════════════════════════════════════════════════════════
# ── TOOL SCHEMAS (given to the LLM) ──────────────────────────
# ════════════════════════════════════════════════════════════
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_docs",
            "description": (
                "Search IntelliRisk AI's loan-approval and fraud-detection POLICY "
                "documents (credit score bands, DTI thresholds, employment rules, "
                "fraud red flags, rejection reasons). Use this for 'what is the "
                "policy / what counts as / why would X be rejected' questions. "
                "Do NOT use this for questions about the actual portfolio data "
                "(counts, averages, rates) — use query_portfolio_data for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy question to search for, in plain English.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_portfolio_data",
            "description": (
                "Run a read-only SQL SELECT query against the REAL portfolio data "
                "(loan_portfolio: 307,511 loan applicants; fraud_claims: 15,420 "
                "insurance claims) to answer analytical questions — counts, "
                "averages, rates, breakdowns by segment, comparisons, top-N. "
                "Only SELECT/WITH statements are executed; DDL/DML is blocked.\n\n"
                f"{SCHEMA_DESCRIPTION}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single DuckDB-compatible SELECT query against loan_portfolio and/or fraud_claims.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_loan_applicant",
            "description": (
                "Run IntelliRisk AI's real trained XGBoost model (AUC 0.7656) on a "
                "hypothetical loan applicant described in the conversation, and "
                "return the approval decision, risk tier, and probabilities. Use "
                "when the user describes an applicant and asks if they'd be "
                "approved, or wants to test a what-if scenario."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_income":     {"type": "number", "description": "Applicant's annual income in dollars. Default 75000."},
                    "loan_amount":       {"type": "number", "description": "Requested loan/credit amount in dollars. Default 200000."},
                    "monthly_payment":   {"type": "number", "description": "Expected monthly repayment (annuity) in dollars. Default 5000."},
                    "asset_value":       {"type": "number", "description": "Value of the goods/asset being financed. Default 180000."},
                    "credit_score_1":    {"type": "number", "description": "Credit bureau score 1, range 0-1. Default 0.5."},
                    "credit_score_2":    {"type": "number", "description": "Credit bureau score 2, range 0-1. Default 0.6."},
                    "credit_score_3":    {"type": "number", "description": "Credit bureau score 3, range 0-1. Default 0.55."},
                    "age":               {"type": "number", "description": "Applicant age in years. Default 35."},
                    "years_employed":    {"type": "number", "description": "Years at current job. Default 5."},
                    "owns_car":          {"type": "boolean", "description": "Whether applicant owns a car. Default false."},
                    "owns_property":     {"type": "boolean", "description": "Whether applicant owns real estate. Default false."},
                    "num_children":      {"type": "integer", "description": "Number of children. Default 0."},
                    "family_members":    {"type": "integer", "description": "Total family members. Default 2."},
                    "defaults_30day":    {"type": "integer", "description": "Social-circle defaults within 30 days. Default 0."},
                    "defaults_60day":    {"type": "integer", "description": "Social-circle defaults within 60 days. Default 0."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_fraud_claim",
            "description": (
                "Run IntelliRisk AI's real trained fraud models (XGBoost + Isolation "
                "Forest, AUC 0.8115) on a hypothetical insurance claim described in "
                "the conversation, and return the fraud verdict, risk tier, and "
                "anomaly score. Use when the user describes a claim scenario and "
                "asks whether it looks fraudulent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_make":        {"type": "string", "enum": list(FRAUD_MAKE_MAP.keys()), "description": "Default Ford."},
                    "vehicle_category":    {"type": "string", "enum": list(FRAUD_VEHICLE_CATEGORY_MAP.keys()), "description": "Default Sedan."},
                    "vehicle_price_range": {"type": "string", "enum": list(FRAUD_VEHICLE_PRICE_MAP.keys()), "description": "Default 'more than 69000'."},
                    "accident_area":       {"type": "string", "enum": list(FRAUD_AREA_MAP.keys()), "description": "Default Urban."},
                    "age_of_vehicle_years":{"type": "integer", "description": "Age of the vehicle in years (0-7+). Default 3."},
                    "policy_type":         {"type": "string", "enum": list(FRAUD_POLICY_TYPE_MAP.keys()), "description": "Default 'Sedan - Collision'."},
                    "base_policy":         {"type": "string", "enum": list(FRAUD_BASE_POLICY_MAP.keys()), "description": "Default Collision."},
                    "deductible":          {"type": "number", "description": "Deductible in dollars (300/400/500/700). Default 400."},
                    "driver_rating":       {"type": "integer", "description": "Driver rating 1-4. Default 2."},
                    "past_claims":         {"type": "integer", "description": "0=None,1=1 claim,2=2 claims,3=more than 2. Default 0."},
                    "police_report_filed": {"type": "boolean", "description": "Default false."},
                    "witness_present":     {"type": "boolean", "description": "Default false."},
                    "fault":               {"type": "string", "enum": list(FRAUD_FAULT_MAP.keys()), "description": "Default 'Policy Holder'."},
                    "address_changed":     {"type": "boolean", "description": "Whether the address changed recently. Default false."},
                    "num_supplements":     {"type": "integer", "description": "Number of claim supplements filed, 0-4. Default 0."},
                    "age_of_policy_holder":{"type": "integer", "description": "Approximate age of the policy holder. Default 30."},
                },
                "required": [],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════
# ── TOOL DISPATCH ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════
def call_search_policy_docs(args: dict) -> dict:
    query = args.get("query", "")
    docs  = retrieve_relevant_docs(query, n_results=3)
    return {"tool": "search_policy_docs", "query": query, "results": docs}


def call_query_portfolio_data(args: dict) -> dict:
    sql    = args.get("sql", "")
    result = run_sql(sql)
    return {"tool": "query_portfolio_data", **result}


def call_score_loan_applicant(args: dict) -> dict:
    income  = args.get("annual_income", 75000)
    credit  = args.get("loan_amount", 200000)
    monthly = args.get("monthly_payment", 5000)
    goods   = args.get("asset_value", 180000)
    e1      = args.get("credit_score_1", 0.5)
    e2      = args.get("credit_score_2", 0.6)
    e3      = args.get("credit_score_3", 0.55)
    age     = args.get("age", 35)
    emp     = args.get("years_employed", 5)
    car     = args.get("owns_car", False)
    realty  = args.get("owns_property", False)
    kids    = args.get("num_children", 0)
    fam     = args.get("family_members", 2)
    d30     = args.get("defaults_30day", 0)
    d60     = args.get("defaults_60day", 0)

    annual_annuity = monthly * 12  # same monthly->annual fix used in the UI
    app = {
        'AMT_INCOME_TOTAL': income, 'AMT_CREDIT': credit,
        'AMT_ANNUITY': annual_annuity, 'AMT_GOODS_PRICE': goods,
        'EXT_SOURCE_1': e1, 'EXT_SOURCE_2': e2, 'EXT_SOURCE_3': e3,
        'AGE_YEARS': age, 'EMPLOYED_YEARS': emp,
        'CREDIT_INCOME_RATIO': credit / (income + 1),
        'ANNUITY_INCOME_RATIO': annual_annuity / (income + 1),
        'CREDIT_TERM': annual_annuity / (credit + 1),
        'GOODS_CREDIT_RATIO': goods / (credit + 1),
        'EXT_SOURCE_MEAN': np.mean([e1, e2, e3]),
        'CNT_CHILDREN': kids, 'CNT_FAM_MEMBERS': fam,
        'FLAG_OWN_CAR': 1 if car else 0, 'FLAG_OWN_REALTY': 1 if realty else 0,
        'DEF_30_CNT_SOCIAL_CIRCLE': d30, 'DEF_60_CNT_SOCIAL_CIRCLE': d60,
    }
    model, feats = load_loan_model()
    res = predict_loan(app, model, feats)
    return {
        "tool": "score_loan_applicant",
        "decision": res["decision"], "risk_tier": res["risk"],
        "approval_probability_pct": res["ap"], "default_probability_pct": res["dp"],
        "inputs_used": {k: v for k, v in app.items()},
    }


def call_score_fraud_claim(args: dict) -> dict:
    make    = FRAUD_MAKE_MAP.get(args.get("vehicle_make", "Ford"), 5)
    vcat    = FRAUD_VEHICLE_CATEGORY_MAP.get(args.get("vehicle_category", "Sedan"), 0)
    vprice  = FRAUD_VEHICLE_PRICE_MAP.get(args.get("vehicle_price_range", "more than 69000"), 5)
    area    = FRAUD_AREA_MAP.get(args.get("accident_area", "Urban"), 1)
    age_veh = args.get("age_of_vehicle_years", 3)
    ptype   = FRAUD_POLICY_TYPE_MAP.get(args.get("policy_type", "Sedan - Collision"), 1)
    bpol    = FRAUD_BASE_POLICY_MAP.get(args.get("base_policy", "Collision"), 1)
    ded     = args.get("deductible", 400)
    drat    = args.get("driver_rating", 2)
    pclaims = args.get("past_claims", 0)
    police  = 1 if args.get("police_report_filed", False) else 0
    witness = 1 if args.get("witness_present", False) else 0
    fault   = FRAUD_FAULT_MAP.get(args.get("fault", "Policy Holder"), 0)
    addr    = 1 if args.get("address_changed", False) else 0
    nsuppl  = args.get("num_supplements", 0)
    age_h   = args.get("age_of_policy_holder", 30)

    nwnp = 1 if police == 0 and witness == 0 else 0
    claim = {
        'Month': 3, 'WeekOfMonth': 2, 'DayOfWeek': 3, 'Make': make, 'AccidentArea': area,
        'DayOfWeekClaimed': 3, 'MonthClaimed': 3, 'WeekOfMonthClaimed': 2, 'Sex': 1,
        'MaritalStatus': 1, 'Age': age_h, 'Fault': fault, 'PolicyType': ptype,
        'VehicleCategory': vcat, 'VehiclePrice': vprice, 'Deductible': ded,
        'DriverRating': drat, 'Days_Policy_Accident': 3, 'Days_Policy_Claim': 2,
        'PastNumberOfClaims': pclaims, 'AgeOfVehicle': age_veh,
        'AgeOfPolicyHolder': age_h, 'PoliceReportFiled': police,
        'WitnessPresent': witness, 'AgentType': 0, 'NumberOfSuppliments': nsuppl,
        'AddressChange_Claim': addr, 'NumberOfCars': 1, 'Year': 1994, 'BasePolicy': bpol,
        'AGE_RISK': 0, 'MULTI_CAR': 0, 'HIGH_DEDUCTIBLE': 1 if ded > 500 else 0,
        'QUICK_CLAIM': 0, 'NO_WITNESS_NO_POLICE': nwnp,
    }
    model, feats, iso = load_fraud_models()
    res = predict_fraud(claim, model, feats, iso)
    return {
        "tool": "score_fraud_claim",
        "verdict": res["verdict"], "risk_tier": res["risk"],
        "fraud_probability_pct": res["fp"], "legitimacy_probability_pct": res["lp"],
        "anomaly_score_pct": res["ano"],
    }


DISPATCH = {
    "search_policy_docs": call_search_policy_docs,
    "query_portfolio_data": call_query_portfolio_data,
    "score_loan_applicant": call_score_loan_applicant,
    "score_fraud_claim": call_score_fraud_claim,
}


def dispatch_tool_call(name: str, args: dict) -> dict:
    fn = DISPATCH.get(name)
    if fn is None:
        return {"tool": name, "ok": False, "error": f"Unknown tool '{name}'."}
    try:
        return fn(args)
    except Exception as e:
        return {"tool": name, "ok": False, "error": str(e)}
