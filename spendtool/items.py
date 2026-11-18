"""Item de-duplication: one item_key per real item.
Links (union-find): (1) same reliable item code, (2) same non-generic description under different codes,
(3) near-identical descriptions (OCR/typing noise) whose numbers agree exactly.
A code is 'unreliable' (a free-text/placeholder code) when it carries many different descriptions; those codes are ignored
and the description decides."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .suppliers import UF

GENERIC = {"", "product", "products", "parts", "part", "service", "services", "no object", "noobject", "n/a", "na", "misc",
           "miscellaneous", "item", "items", "other", "material", "general", "none"}


def _key(desc):
    return re.sub(r"[^a-z0-9]", "", desc)


def _nums(desc):
    return tuple(sorted(re.findall(r"\d+(?:[.,]\d+)?", desc)))


def item_keys(df, sim_thr=0.88, placeholder_min_descs=8, min_desc_len=12, policy=None, log=print):
    """policy (optional, from config `items:`): {column, by_code: [...], by_description: [...], pooled: [...]}
    Rows whose `column` value is in `pooled` share ONE catch-all item (the client's own convention for uncatalogued
    free-text/catalogue lines); `by_description` rows ignore their code and are grouped by description."""
    desc = df["desc"].fillna("")
    code = df["item_code"].astype("string").str.strip()
    code = code.where(code.notna() & (code != ""), pd.NA)
    dk = desc.map(_key)
    pooled = pd.Series(False, index=df.index)
    by_code_rows = pd.Series(False, index=df.index)
    if policy and policy.get("column"):
        col = df.get("x_" + policy["column"], df.get(policy["column"]))
        if col is not None:
            pooled = col.isin(policy.get("pooled", []))
            by_code_rows = col.isin(policy.get("by_code", []))
            code = code.where(~col.isin(policy.get("by_description", [])), pd.NA)
    informative = (dk.str.len() >= min_desc_len) & ~desc.str.strip().isin(GENERIC)
    informative = informative & ~by_code_rows
    # placeholder codes: many distinct descriptions under one code
    nd = pd.DataFrame({"c": code, "d": dk}).dropna(subset=["c"]).groupby("c")["d"].nunique()
    placeholder = set(nd[nd >= placeholder_min_descs].index)
    reliable = code.notna() & ~code.isin(placeholder)
    # nodes: unique codes (reliable) and unique informative descriptions
    nodes, uf = {}, None
    def nid(k):
        if k not in nodes:
            nodes[k] = len(nodes)
        return nodes[k]
    for c in code[reliable].unique():
        nid(("c", c))
    ud = dk[informative].unique()
    for d in ud:
        nid(("d", d))
    uf = UF(len(nodes))
    # code <-> description links
    m = reliable & informative
    for c, d in set(zip(code[m], dk[m])):
        uf.union(nodes[("c", c)], nodes[("d", d)])
    # description <-> description: near-duplicates with the same numbers
    if len(ud) > 1 and sim_thr <= 1.0:
        rep = {}
        for d, t in zip(dk[informative], desc[informative]):
            rep.setdefault(d, t)
        texts = [rep[d] for d in ud]
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
        nums = [_nums(t) for t in texts]
        B = 1500
        for s in range(0, len(ud), B):
            S = (X[s:s + B] @ X.T).tocoo()
            for i, j, v in zip(S.row + s, S.col, S.data):
                if i < j and v >= sim_thr and nums[i] == nums[j]:
                    uf.union(nodes[("d", ud[i])], nodes[("d", ud[j])])
    root_of = lambda k: uf.find(nodes[k])
    freq = {}
    out = []
    for c, d, rel, inf in zip(code, dk, reliable, informative):
        if rel:
            r = root_of(("c", c))
        elif inf:
            r = root_of(("d", d))
        else:
            r = None
        out.append(r)
    # label each cluster by its most common code, else description
    label = {}
    for r, c, d, rel in zip(out, code, dk, reliable):
        if r is not None and r not in label:
            label[r] = ("I:" + str(c)) if rel else ("D:" + d[:40])
    keys = []
    for r, c, d, pl in zip(out, code, dk, pooled):
        if pl:
            keys.append("I:POOLED")
        elif r is not None:
            keys.append(label[r])
        elif pd.notna(c):
            keys.append("P:" + str(c) + ("|" + d[:30] if d else ""))     # placeholder code + short text
        else:
            keys.append("U:" + d[:30])
    return pd.Series(keys, index=df.index)
