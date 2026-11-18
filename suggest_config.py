#!/usr/bin/env python
"""Draft a config for a NEW company from its files, so the blind-test config takes minutes, not an hour.
    python suggest_config.py --data-dir data --file invoices.csv --taxonomy tree.csv --company C > configs/company_C.yaml
Always read the draft and fix it: it is a guess from column names and values."""
import argparse, re, os
import pandas as pd
import yaml
from spendtool.cleaning import parse_amount, parse_date

KEYS = {
    "id": r"line[_ ]?id|txn|transaction[_ ]?id|record[_ ]?id|^id$|row",
    "date": r"posting|invoice[_ ]?date|^date|doc(ument)?[_ ]?date|txn[_ ]?date|period",
    "amount": r"net|amount|value|total|spend|cost|price",
    "currency": r"curr|ccy",
    "supplier": r"supplier|vendor[_ ]?name|vendor|payee|creditor",
    "desc": r"desc|text|item|product|material[_ ]?name|memo|narrat|particular",
    "status": r"status|state|approval",
    "po": r"^po(?![a-z])|^po[_ ]|purchase[_ ]?order|\\bpo[_ ]?(no|num)",
    "invoice": r"invoice[_ ]?(no|num|id)|inv[_ ]?(no|num)",
}


def pick(cols, pat, avoid=None, df=None, kind=None):
    best = None
    for c in cols:
        if re.search(pat, c, re.I) and not (avoid and re.search(avoid, c, re.I)):
            if kind == "amount" and df is not None:
                if parse_amount(df[c].head(500)).notna().mean() < 0.8:
                    continue
            if kind == "date" and df is not None:
                if parse_date(df[c].head(500))[0].notna().mean() < 0.5:
                    continue
            best = best or c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--file", required=True)
    ap.add_argument("--taxonomy", required=True); ap.add_argument("--company", default="C")
    ap.add_argument("--currency", default="USD")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.data_dir, a.file), dtype=str, keep_default_na=False, na_values=[""], nrows=3000)
    cols = list(df.columns)
    tax = pd.read_csv(os.path.join(a.data_dir, a.taxonomy), nrows=5)
    c = {}
    c["date"] = pick(cols, KEYS["date"], df=df, kind="date")
    c["amount"] = pick(cols, KEYS["amount"], avoid=r"tax|vat|unit|rate", df=df, kind="amount")
    c["currency"] = pick(cols, KEYS["currency"], avoid=r"rate")
    c["supplier"] = pick(cols, KEYS["supplier"], avoid=r"id$|code|type") or pick(cols, KEYS["supplier"], avoid=r"type")
    c["status"] = pick(cols, KEYS["status"])
    c["po"] = pick(cols, KEYS["po"]); c["invoice"] = pick(cols, KEYS["invoice"])
    c["item_code"] = pick(cols, r"material[_ ]?(no|number|code)|sku|item[_ ]?(code|no|number)")
    idc = pick(cols, KEYS["id"]) or cols[0]
    desc = [x for x in cols if re.search(KEYS["desc"], x, re.I) and x != c["supplier"]][:2]
    cfg = {"company": a.company, "reporting_currency": a.currency, "taxonomy": {"file": a.taxonomy},
           "mode": "one_system",
           "sources": [{"name": "inv", "role": "primary", "file": a.file, "id_col": idc, "id_prefix": "",
                        "columns": {k: v for k, v in c.items() if v}, "description": desc, "amount_is_gross": False}]}
    if c["status"]:
        vals = df[c["status"]].fillna("").str.strip().value_counts()
        cfg["status"] = {"approved": ["FILL_ME"], "cancelled": ["FILL_ME"]}
        print("# status values seen: " + ", ".join(f"{k} ({v})" for k, v in vals.head(12).items()))
    print("# taxonomy columns: " + ", ".join(tax.columns))
    print("# CHECK: amount gross or net? currency? status mapping? unused columns worth listing under `extra`?")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
