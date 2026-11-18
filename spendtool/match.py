"""Count-once logic for two-system companies: (1) repeated secondary exports, (2) secondary vs primary matching."""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def mark_export_duplicates(sec):
    """Same invoice exported twice: identical invoice+text+net+qty and compatible PO(+item). Returns bool Series."""
    d = sec.copy()
    d["_k"] = (d["invoice"].astype(str) + "|" + d["desc"] + "|" + d["net"].round(2).astype(str) + "|" +
               d["quantity"].astype(str) + "|" + d["supplier_key"].astype(str))
    d["_po"] = (d["po"].fillna("") + "/" + d["po_item"].fillna("")).where(d["po"].notna(), "")
    dup = pd.Series(False, index=d.index)
    for _, g in d.groupby("_k"):
        if len(g) < 2:
            continue
        seen = set()
        has_po = (g["_po"] != "").any()
        first_blank_kept = False
        for i, p in zip(g.index, g["_po"]):
            if p != "":
                if p in seen:
                    dup[i] = True
                seen.add(p)
            else:
                if has_po or first_blank_kept:
                    dup[i] = True
                first_blank_kept = True
    return dup


def match_to_primary(pri, sec, cfg):
    """Return Series (index = sec index) -> list of matched primary indices (or empty), plus method label."""
    m = cfg.get("matching", {})
    tol_abs, tol_rel = m.get("amount_tol_abs", 0.05), m.get("amount_tol_rel", 0.02)
    sim_min = m.get("text_sim_min", 0.35)
    sim_min_nodate = m.get("text_sim_min_nodate", 0.5)
    max_gap = m.get("max_date_gap_days", 90)
    pdate = pri["date"].values.astype("datetime64[D]")
    pdate_ok = ~pd.isna(pri["date"].values)
    pnet = pri["net"].to_dict()
    consumed = set()
    by_po = defaultdict(list)
    for i, po in pri["po"].items():
        if po is not None and not (isinstance(po, float) and np.isnan(po)):
            by_po[po].append(i)
    matched, method = {}, {}

    def close(a, b, rel=0.0):
        return abs(a - b) <= max(tol_abs, rel * abs(b))

    def try_po(j, row, rel, label):
        cand = by_po.get(row.po)
        if not cand:
            return False
        if row.po_item is not None and not pd.isna(row.po_item):
            sub = [i for i in cand if pri.at[i, "po_item"] == row.po_item]
            groups = [sub, cand] if sub else [cand]
        else:
            groups = [cand]
        for grp in groups:
            free = [i for i in grp if i not in consumed]
            for i in free:                                       # single free line
                if close(row.net, pnet[i], rel):
                    matched[j] = [i]; method[j] = label + "_line"; consumed.add(i); return True
            if len(free) > 1 and close(row.net, sum(pnet[i] for i in free), rel):   # several receipts, one invoice line
                matched[j] = free; method[j] = label + "_sum"; consumed.update(free); return True
            for i in grp:                                        # already-consumed line (invoice line vs partial re-receipt)
                if i in consumed and close(row.net, pnet[i], rel):
                    matched[j] = [i]; method[j] = label + "_shared"; return True
        return False

    order = sec.sort_values(["po_item"], na_position="last").index
    for rel, label in ((0.0, "po_exact"), (tol_rel, "po_tol")):
        for j in order:
            if j in matched:
                continue
            r = sec.loc[j]
            if r.po is None or pd.isna(r.po):
                continue
            try_po(j, r, rel, label)

    # fallback: supplier cluster + amount + text similarity, against unconsumed primary lines
    rest = [j for j in sec.index if j not in matched]
    if rest:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
        vec.fit(pd.concat([pri["desc"], sec.loc[rest, "desc"]]))
        P = vec.transform(pri["desc"])
        Sx = vec.transform(sec.loc[rest, "desc"])
        pidx = np.array(pri.index)
        order_p = np.argsort(pri["net"].values)
        pn_sorted = pri["net"].values[order_p]
        psup = pri["supplier_key"].values
        for n, j in enumerate(rest):
            r = sec.loc[j]
            lo = np.searchsorted(pn_sorted, r.net - max(tol_abs, tol_rel * abs(r.net)), "left")
            hi = np.searchsorted(pn_sorted, r.net + max(tol_abs, tol_rel * abs(r.net)), "right")
            cands = [order_p[k] for k in range(lo, hi)]
            cands = [k for k in cands if pidx[k] not in consumed and psup[k] == r.supplier_key]
            sdate_ok = not pd.isna(r.date)
            if sdate_ok:                                   # reject pairs whose dates are far apart
                sd = np.datetime64(r.date, "D")
                cands = [k for k in cands if (not pdate_ok[k]) or abs((pdate[k] - sd).astype(int)) <= max_gap]
            if not cands:
                continue
            sims = (P[cands] @ Sx[n].T).toarray().ravel()
            if sdate_ok:                                   # prefer closer dates among near-ties
                gaps = np.array([abs((pdate[k] - sd).astype(int)) if pdate_ok[k] else 60 for k in cands])
                sims = sims - gaps / 3000.0
            b = int(np.argmax(sims))
            exact = abs(pri.at[pidx[cands[b]], "net"] - r.net) <= tol_abs
            dated = sdate_ok and pdate_ok[cands[b]]
            need = sim_min if dated else sim_min_nodate
            if sims[b] >= need or (len(cands) == 1 and exact and dated):
                i = pidx[cands[b]]
                matched[j] = [i]; method[j] = "fallback_sup_amt_text"; consumed.add(i)
    return pd.Series({j: matched.get(j, []) for j in sec.index}), pd.Series({j: method.get(j, "") for j in sec.index})
