#!/usr/bin/env python
"""One command:  python run.py --config configs/company_A.yaml --data-dir data --out output/yourname__A.csv"""
import argparse, os
from spendtool.pipeline import run_company, write_output

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data-dir", default="data")
ap.add_argument("--out", required=True)
ap.add_argument("--train-dir", default=None, help="folder with the ORIGINAL files + labels used for training (default: --data-dir)")
ap.add_argument("--zero-label", action="store_true", help="ignore labels: classify from the tree alone")
ap.add_argument("--audit", default=None, help="optional path for a diagnostic CSV (matching, fx flags)")
a = ap.parse_args()
cfg, df = run_company(a.config, a.data_dir, train_dir=a.train_dir, zero_label=a.zero_label)
write_output(df, a.out)
if a.audit:
    df.drop(columns=["desc"]).to_csv(a.audit, index=False)
print("wrote", a.out)
