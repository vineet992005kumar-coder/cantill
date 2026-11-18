#!/usr/bin/env python
"""Check an output file against the brief: python validate_output.py out.csv --config configs/company_X.yaml --data-dir data"""
import argparse, os, sys
import pandas as pd
from spendtool.config import load_config
from spendtool.taxonomy import Taxonomy
from spendtool.ingest import load_all

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--config", required=True); ap.add_argument("--data-dir", default="data")
a = ap.parse_args()
cfg = load_config(a.config)
o = pd.read_csv(a.out, dtype={"exclusion_reason": str}, keep_default_na=False)
inp = load_all(cfg, a.data_dir)
tx = cfg["taxonomy"]; tax = Taxonomy(os.path.join(a.data_dir, tx["file"]), tx.get("levels"))
errs = []
want = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]
if list(o.columns) != want: errs.append(f"columns {list(o.columns)}")
if len(o) != len(inp): errs.append(f"rows {len(o)} != input {len(inp)}")
if o.record_id.duplicated().any(): errs.append("duplicate record_id")
if set(o.record_id) != set(inp.record_id): errs.append("record_id set differs from input")
if not o.include.isin([0, 1]).all(): errs.append("include not 0/1")
inc = o[o.include == 1]; exc = o[o.include == 0]
if (inc.exclusion_reason != "").any(): errs.append("included rows with a reason")
if not exc.exclusion_reason.isin(["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]).all(): errs.append("bad reason code")
if (exc.spend != 0).any(): errs.append("excluded rows with spend != 0")
if not o.confidence.between(0, 1).all(): errs.append("confidence outside 0..1")
paths = {tuple(x for x in p) for p in tax.leaves}
bad = [tuple(x for x in r if x) for r in inc[["l1", "l2", "l3", "l4"][:tax.depth]].itertuples(index=False)]
nbad = sum(1 for b in bad if b not in paths)
if nbad: errs.append(f"{nbad} included rows with a path that is not a tree leaf")
print(f"rows={len(o)} included={len(inc)} spend={o.spend.sum():,.2f}")
print("OK" if not errs else "PROBLEMS:\n  " + "\n  ".join(errs))
sys.exit(1 if errs else 0)
