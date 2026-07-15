# app/data_tools.py
# ============================================================
# Read-only, SQL-safe analytics layer over the loan + fraud
# portfolios, used by the agentic chatbot's "portfolio_query"
# tool (text-to-SQL). The LLM writes SQL against these two
# tables; this module is the guardrail that makes it safe to
# actually execute LLM-written SQL:
#   - only SELECT / WITH statements are allowed
#   - one statement per call (no ";"-chained statements)
#   - a small set of banned keywords blocks DDL/DML and file I/O
#   - every query is capped at MAX_ROWS returned rows
# ============================================================
import os
import re
import duckdb
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data', 'processed')

MAX_ROWS = 200

# Statements/keywords never allowed in agent-generated SQL.
_BANNED_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|"
    r"pragma|export|import|install|load|call|vacuum|checkpoint)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


@st.cache_resource
def get_connection():
    """One in-memory DuckDB connection with the two portfolios registered
    as tables. Cached so the CSVs are loaded once per server process."""
    con = duckdb.connect(database=':memory:')
    loan_df  = pd.read_csv(os.path.join(DATA_DIR, 'loan_processed.csv'))
    fraud_df = pd.read_csv(os.path.join(DATA_DIR, 'fraud_processed.csv'))
    con.register('loan_portfolio_df', loan_df)
    con.register('fraud_claims_df', fraud_df)
    # Materialize into real tables (register() ties lifetime to the
    # python object, CREATE TABLE AS makes them independent of that).
    con.execute("CREATE TABLE loan_portfolio AS SELECT * FROM loan_portfolio_df")
    con.execute("CREATE TABLE fraud_claims AS SELECT * FROM fraud_claims_df")
    return con


SCHEMA_DESCRIPTION = """\
Table: loan_portfolio  (307,511 rows — Home Credit loan applicants)
Columns: TARGET (0=repaid,1=defaulted), AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
AMT_GOODS_PRICE, EXT_SOURCE_1/2/3 (credit bureau scores 0-1), REGION_POPULATION_RELATIVE,
HOUR_APPR_PROCESS_START, OWN_CAR_AGE, FLAG_OWN_CAR (0/1), FLAG_OWN_REALTY (0/1),
OCCUPATION_TYPE, ORGANIZATION_TYPE (label-encoded, too many categories to map — group by
the raw int code if asked), FLAG_DOCUMENT_3 (0/1), REG_CITY_NOT_WORK_CITY (0/1),
LIVE_CITY_NOT_WORK_CITY (0/1), DEF_30_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE,
CNT_CHILDREN, CNT_FAM_MEMBERS, AGE_YEARS, EMPLOYED_YEARS, CREDIT_INCOME_RATIO,
ANNUITY_INCOME_RATIO, CREDIT_TERM, GOODS_CREDIT_RATIO, EXT_SOURCE_MEAN.

All categorical columns below are label-encoded integers — use the mapping to translate
business language into the correct integer filter value:
  NAME_CONTRACT_TYPE: 0=Cash loans, 1=Revolving loans
  CODE_GENDER: 0=F, 1=M, 2=XNA
  NAME_INCOME_TYPE: 0=Businessman, 1=Commercial associate, 2=Maternity leave, 3=Pensioner,
    4=State servant, 5=Student, 6=Unemployed, 7=Working
  NAME_EDUCATION_TYPE: 0=Academic degree, 1=Higher education, 2=Incomplete higher,
    3=Lower secondary, 4=Secondary / secondary special
  NAME_FAMILY_STATUS: 0=Civil marriage, 1=Married, 2=Separated, 3=Single / not married,
    4=Unknown, 5=Widow
  NAME_HOUSING_TYPE: 0=Co-op apartment, 1=House / apartment, 2=Municipal apartment,
    3=Office apartment, 4=Rented apartment, 5=With parents

Table: fraud_claims  (15,420 rows — Oracle vehicle insurance claims)
Columns: FraudFound_P (0=legit,1=fraud), Month, WeekOfMonth, DayOfWeek, DayOfWeekClaimed,
MonthClaimed, WeekOfMonthClaimed, Age, Deductible, DriverRating, NumberOfSuppliments,
NumberOfCars, Year, AGE_RISK (1=age<25 or >70), MULTI_CAR (0/1), HIGH_DEDUCTIBLE (0/1),
QUICK_CLAIM (0/1), NO_WITNESS_NO_POLICE (0/1).

All categorical columns below are label-encoded integers — use the mapping to translate
business language into the correct integer filter value:
  Make: 0=Accura,1=BMW,2=Chevrolet,3=Dodge,4=Ferrari,5=Ford,6=Honda,7=Jaguar,8=Lexus,
    9=Mazda,10=Mecedes,11=Mercury,12=Nisson,13=Pontiac,14=Porche,15=Saab,16=Saturn,17=Toyota,18=VW
  AccidentArea: 0=Rural, 1=Urban
  Sex: 0=Female, 1=Male
  MaritalStatus: 0=Divorced, 1=Married, 2=Single, 3=Widow
  Fault: 0=Policy Holder, 1=Third Party
  PolicyType: 0=Sedan-AllPerils,1=Sedan-Collision,2=Sedan-Liability,3=Sport-AllPerils,
    4=Sport-Collision,5=Sport-Liability,6=Utility-AllPerils,7=Utility-Collision,8=Utility-Liability
  VehicleCategory: 0=Sedan, 1=Sport, 2=Utility
  VehiclePrice: 0='20000 to 29000',1='30000 to 39000',2='40000 to 59000',3='60000 to 69000',
    4='less than 20000',5='more than 69000'
  PoliceReportFiled: 0=No, 1=Yes
  WitnessPresent: 0=No, 1=Yes
  AgentType: 0=External, 1=Internal
  BasePolicy: 0=All Perils, 1=Collision, 2=Liability
  Days_Policy_Accident: 0='1 to 7',1='15 to 30',2='8 to 15',3='more than 30',4='none'
  Days_Policy_Claim: 0='15 to 30',1='8 to 15',2='more than 30',3='none'
  AddressChange_Claim: 0='1 year',1='2 to 3 years',2='4 to 8 years',3='no change',4='under 6 months'
  PastNumberOfClaims: 0='1',1='2 to 4',2='more than 4',3='none'
  AgeOfVehicle: 0='2 years',1='3 years',2='4 years',3='5 years',4='6 years',5='7 years',
    6='more than 7',7='new'
  AgeOfPolicyHolder: 0='16 to 17',1='18 to 20',2='21 to 25',3='26 to 30',4='31 to 35',
    5='36 to 40',6='41 to 50',7='51 to 65',8='over 65'
"""


def _validate_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(';').strip()
    if not stripped:
        raise UnsafeSQLError("Empty query.")
    if ';' in stripped:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", stripped, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT / WITH (read-only) queries are allowed.")
    if _BANNED_KEYWORDS.search(stripped):
        raise UnsafeSQLError("Query contains a disallowed keyword (only read-only SELECTs are permitted).")
    if not re.search(r"\blimit\s+\d+", stripped, re.IGNORECASE):
        stripped = f"{stripped}\nLIMIT {MAX_ROWS}"
    return stripped


def run_sql(sql: str) -> dict:
    """Execute an agent-generated SQL query safely and return rows.

    Returns:
        {"ok": True,  "sql": <sql actually run>, "columns": [...], "rows": [...]}
        {"ok": False, "sql": <sql>, "error": "..."}
    """
    try:
        safe_sql = _validate_sql(sql)
    except UnsafeSQLError as e:
        return {"ok": False, "sql": sql, "error": str(e)}

    try:
        con    = get_connection()
        result = con.execute(safe_sql)
        cols   = [d[0] for d in result.description]
        rows   = result.fetchall()
        rows   = [dict(zip(cols, r)) for r in rows[:MAX_ROWS]]
        return {"ok": True, "sql": safe_sql, "columns": cols, "rows": rows}
    except Exception as e:
        return {"ok": False, "sql": safe_sql, "error": f"SQL execution error: {e}"}
