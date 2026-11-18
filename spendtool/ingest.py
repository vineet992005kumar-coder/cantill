"""Load every source in the config into one standardised frame."""
import os
import numpy as np
import pandas as pd
from .cleaning import parse_amount, parse_vat, parse_date, parse_po, norm_item_no, norm_text

STD = ["record_id", "src", "role", "date", "date_flag", "amount_raw", "net", "currency", "rate", "supplier_raw",
       "vendor_id", "po", "po_item", "invoice", "quantity", "item_code", "status_raw", "desc"]


def _col(df, name):
    return df[name] if name and name in df.columns else pd.Series(np.nan, index=df.index)


def load_source(src, data_dir, cfg):
    df = pd.read_csv(os.path.join(data_dir, src["file"]), dtype=str, keep_default_na=False, na_values=[""])
    c = src.get("columns", {})
    o = pd.DataFrame(index=df.index)
    o["record_id"] = src.get("id_prefix", "") + df[src["id_col"]].astype(str)
    o["src"] = src["name"]
    o["role"] = src.get("role", "primary")
    o["date"], o["date_flag"] = parse_date(_col(df, c.get("date")), src.get("date_formats"))
    amt = parse_amount(_col(df, c.get("amount")))
    o["amount_raw"] = amt
    if src.get("amount_is_gross"):
        vat = parse_vat(_col(df, c.get("vat")), src.get("default_vat", 0.0))
        o["net"] = amt / (1.0 + vat)
    else:
        o["net"] = amt
    o["currency"] = _col(df, c.get("currency")).fillna(cfg["reporting_currency"]).str.strip().str.upper()
    o["rate"] = parse_amount(_col(df, c.get("rate"))) if c.get("rate") else np.nan
    o["supplier_raw"] = _col(df, c.get("supplier"))
    o["vendor_id"] = _col(df, c.get("vendor_id"))
    po, poi = parse_po(_col(df, c.get("po")))
    o["po"] = po
    o["po_item"] = norm_item_no(_col(df, c.get("po_item"))) if c.get("po_item") else poi
    o["invoice"] = _col(df, c.get("invoice"))
    o["quantity"] = parse_amount(_col(df, c.get("quantity")))
    o["item_code"] = _col(df, c.get("item_code"))
    o["status_raw"] = _col(df, c.get("status"))
    desc = df[src.get("description", [])].fillna("").astype(str) if src.get("description") else pd.DataFrame(index=df.index)
    o["desc"] = desc.apply(lambda r: " ".join(x for x in r if x.strip()), axis=1).map(norm_text) if len(desc.columns) else ""
    for ex in src.get("extra", []):
        o["x_" + ex] = df[ex] if ex in df.columns else np.nan
    return o


def load_all(cfg, data_dir):
    parts = [load_source(s, data_dir, cfg) for s in cfg["sources"]]
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["row"] = np.arange(len(out))
    return out


def convert_currency(df, cfg, data_dir):
    """net -> reporting currency. rate = units of line currency per 1 USD; fallback to monthly fx table."""
    rep = cfg["reporting_currency"]
    fx = cfg.get("fx", {})
    table = None
    if fx.get("file") and os.path.exists(os.path.join(data_dir, fx["file"])):
        t = pd.read_csv(os.path.join(data_dir, fx["file"]))
        t.columns = [c.lower() for c in t.columns]
        cur, mon, val = fx.get("currency_col", "currency"), fx.get("month_col", "month"), fx.get("rate_col", "units_per_usd")
        t["_m"] = pd.to_datetime(t[mon].astype(str).str.slice(0, 7), format="%Y-%m", errors="coerce")
        table = t.dropna(subset=["_m"]).rename(columns={cur: "currency", val: "r"})[["currency", "_m", "r"]]
    cur_ok = df["currency"].eq(rep)
    rate = pd.Series(np.nan, index=df.index)
    flag = pd.Series("", index=df.index)
    rate[cur_ok] = 1.0
    own = (~cur_ok) & df["rate"].notna() & (df["rate"] > 0)
    # line rate is per USD; reporting currency may be USD (rate_rep=1) or other (look up)
    rep_per_usd = 1.0 if rep == "USD" else None
    if table is not None and rep != "USD":
        rr = table[table.currency == rep].set_index("_m").r
        rep_per_usd = rr
    mon = df["date"].dt.to_period("M").dt.to_timestamp()
    if table is not None:
        mid = {}
        for cur, g in table.groupby("currency"):
            mid[cur] = g.sort_values("_m")
        need = (~cur_ok) & ~own
        look = pd.Series(np.nan, index=df.index)
        for cur in df.loc[need, "currency"].unique():
            g = mid.get(cur)
            if g is None:
                continue
            idx = df.index[need & (df.currency == cur)]
            m = mon[idx]
            ok = m.notna()
            if ok.any():
                tmp = pd.merge_asof(pd.DataFrame({"_m": m[ok].values, "i": idx[ok]}).sort_values("_m"), g[["_m", "r"]],
                                    on="_m", direction="nearest")
                look[tmp["i"].values] = tmp["r"].values
            if (~ok).any():                       # no date: use the currency's median rate
                look[idx[~ok]] = g["r"].median()
        rate[need] = look[need]
        flag[need & look.notna()] = "fx_table"
    rate[own] = df.loc[own, "rate"]
    rate[(~cur_ok) & rate.isna()] = np.nan
    flag[(~cur_ok) & rate.isna()] = "fx_missing"
    usd = df["net"] / rate                       # amount in USD when rate is per-USD
    if rep == "USD":
        spend = df["net"].where(cur_ok, usd)
    else:
        spend = df["net"].where(cur_ok, usd)     # TODO cross-rate for non-USD reporting with foreign lines
    return spend, flag
