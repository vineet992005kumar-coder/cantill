import os, time
import numpy as np
import pandas as pd
from .config import load_config
from .ingest import load_all, convert_currency
from .suppliers import cluster_suppliers, pool_unmastered
from .items import item_keys as dedup_items
from .exclude import apply_exclusions
from .match import mark_export_duplicates, match_to_primary

OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def item_keys(df):
    """Provisional item key: item code when present, else normalised description (improved in items.py later)."""
    code = df["item_code"].astype("string").str.strip()
    key = np.where(code.notna() & (code != ""), "I:" + code.fillna(""), "D:" + df["desc"].str.replace(r"[^a-z0-9]", "", regex=True).str.slice(0, 40))
    return pd.Series(key, index=df.index)


def link_secondary_items(df, pri_idx, sec_idx, matched):
    """Secondary lines not matched to a primary record: reuse the primary item key when the description is identical."""
    un = [j for j in sec_idx if not matched.get(j)]
    if not un:
        return
    pk = df.loc[pri_idx]
    pk = pk[pk["desc"].str.len() >= 8]
    top = pk.groupby("desc")["item_key"].agg(lambda v: v.value_counts().index[0])
    d = df.loc[un, "desc"]
    hit = d.map(top)
    df.loc[hit.dropna().index, "item_key"] = hit.dropna()


def run_company(cfg_path, data_dir, log=print, train_dir=None, zero_label=False):
    t0 = time.time()
    cfg = load_config(cfg_path)
    df = load_all(cfg, data_dir)
    log(f"loaded {len(df)} records")
    df["spend_conv"], df["fx_flag"] = convert_currency(df, cfg, data_dir)
    sup_cfg = cfg.get("suppliers", {})
    df["supplier_key"] = cluster_suppliers(df["supplier_raw"], df["vendor_id"], sim_thr=sup_cfg.get("sim_thr", 0.80) if sup_cfg.get("fuzzy", True) else 2.0)
    if sup_cfg.get("pool_unmastered"):
        df["supplier_key"] = pool_unmastered(df["supplier_key"], df["vendor_id"])
    df.loc[df["supplier_key"] == "", "supplier_key"] = "S:UNKNOWN"
    df["item_key"] = dedup_items(df, policy=cfg.get("items"), log=log)
    df["exclusion_reason"] = apply_exclusions(df, cfg)
    df["match_method"] = ""
    df["matched_to"] = ""
    if cfg["mode"] == "two_system":
        pri = df[df.role == "primary"]
        sec = df[df.role == "secondary"]
        live_sec = sec[df.loc[sec.index, "exclusion_reason"] == ""]
        dup = mark_export_duplicates(live_sec)
        df.loc[dup[dup].index, "exclusion_reason"] = "DUPLICATE"
        df.loc[dup[dup].index, "match_method"] = "export_repeat"
        live_sec = live_sec[~dup.reindex(live_sec.index).fillna(False)]
        live_pri = pri[df.loc[pri.index, "exclusion_reason"] == ""]
        mt, mm = match_to_primary(live_pri, live_sec, cfg)
        for j, lst in mt.items():
            if lst:
                df.at[j, "exclusion_reason"] = "DUPLICATE"
                df.at[j, "match_method"] = mm[j]
                df.at[j, "matched_to"] = ";".join(df.at[i, "record_id"] for i in lst)
                df.at[j, "supplier_key"] = df.at[lst[0], "supplier_key"]
                df.at[j, "item_key"] = df.at[lst[0], "item_key"]
        link_secondary_items(df, pri.index, sec.index, mt)
        log(f"match: {(mm != '').sum()} secondary lines matched ({mm[mm != ''].value_counts().to_dict()})")
    df["include"] = (df["exclusion_reason"] == "").astype(int)
    df["spend"] = np.where(df["include"] == 1, df["spend_conv"].fillna(0.0), 0.0).round(2)
    from .taxonomy import Taxonomy
    from .classify import classify
    tx = cfg["taxonomy"]
    tax = Taxonomy(os.path.join(data_dir, tx["file"]), tx.get("levels"))
    for c in ("l1", "l2", "l3", "l4"):
        df[c] = ""
    df["confidence"] = 0.0
    out = conf = None
    if not zero_label:
        from .supervised import Supervised, load_training
        tr = load_training(cfg, train_dir or data_dir, tax, log=log)
        if tr is not None:
            model = Supervised().fit(tr, tr["_path"])
            inc = df[df["include"] == 1]
            P = model.predict(inc)
            best = P.argmax(1)
            code_hit = inc["item_code"].map(model.code_map)
            best = np.where(code_hit.notna(), code_hit.fillna(0).astype(int), best)
            conf = np.where(code_hit.notna(), 0.98, P.max(1))
            rows = [list(model.classes_[i]) + [""] * (tax.depth - len(model.classes_[i])) for i in best]
            out = pd.DataFrame(rows, index=inc.index, columns=[f"l{i + 1}" for i in range(tax.depth)])
            conf = pd.Series(conf, index=inc.index)
            log(f"classifier: supervised on {len(tr)} labelled lines")
    if out is None:
        out, conf, _ = classify(df, tax, cfg, log=log)
    for i in range(tax.depth):
        df.loc[out.index, f"l{i + 1}"] = out[f"l{i + 1}"]
    df.loc[conf.index, "confidence"] = conf.round(3)
    log(f"done in {time.time() - t0:.1f}s")
    return cfg, df


def write_output(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df[OUT_COLS].copy()
    out["exclusion_reason"] = out["exclusion_reason"].replace("", np.nan)
    out.to_csv(path, index=False)
