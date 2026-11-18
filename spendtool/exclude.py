"""Rule-based exclusions. Reason codes: CANCELLED, NOT_APPROVED, NON_PROCUREMENT (DUPLICATE is set in match.py)."""
import warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted")
import re
import numpy as np
import pandas as pd

DEFAULT_NP = (r"\b(?:vat|sars|revenue service|income tax|withholding tax|paye|uif|intercompany|inter-company|inter company|"
              r"royalt(?:y|ies)|dividends?|management fees?|interest (?:charge|charges|expense|paid)|bank charges?|"
              r"loan repayment|payroll|salary|salaries|customs duty|import duty)\b")


def apply_exclusions(df, cfg):
    st = cfg.get("status", {})
    approved = {s.lower() for s in st.get("approved", [])}
    cancelled = {s.lower() for s in st.get("cancelled", [])}
    reason = pd.Series("", index=df.index)
    s = df["status_raw"].astype("string").str.strip().str.lower()
    has = s.notna() & (s != "")
    if approved or cancelled:
        reason[has & s.isin(cancelled)] = "CANCELLED"
        reason[has & ~s.isin(cancelled) & ~s.isin(approved)] = "NOT_APPROVED"
    # non-procurement
    pats = []
    if cfg.get("use_default_keyword_rules", True):
        pats.append((["desc", "supplier_raw"], DEFAULT_NP))
    for r in cfg.get("non_procurement_rules", []):
        pats.append((r["columns"] if "columns" in r else [r["column"]], r["regex"]))
    npm = pd.Series(False, index=df.index)
    for cols, rx in pats:
        cre = re.compile(rx, re.I)
        for c in cols:
            col = c if c in df.columns else ("x_" + c if "x_" + c in df.columns else None)
            if col is None:
                continue
            npm |= df[col].astype("string").fillna("").str.contains(cre, na=False)
    reason[(reason == "") & npm] = "NON_PROCUREMENT"
    return reason
