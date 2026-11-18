"""Supplier de-duplication: normalise names, union exact keys + vendor ids, then fuzzy char-ngram merge."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LEGAL = {"pty", "ltd", "limited", "sa", "rsa", "co", "company", "inc", "llc", "gmbh", "cc", "proprietary", "the",
         "and", "ag", "bv", "sas", "srl", "spa", "plc", "corp", "corporation", "llp", "lp", "pvt", "private",
         "sdn", "bhd", "oy", "ab", "as", "nv", "kg", "ou", "sp", "zoo", "de", "cv", "sl"}


def norm_name(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", str(x).lower())
    toks = [t for t in s.split() if t not in LEGAL]
    return " ".join(toks)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)


def cluster_suppliers(names, vendor_ids=None, sim_thr=0.80, block_chars=3):
    """names: Series of raw supplier strings (index = row). vendor_ids: optional Series (same index).
    Returns Series of supplier_key (str)."""
    norm = names.map(norm_name)
    keyspace = norm.str.replace(" ", "", regex=False)
    uniq = keyspace[keyspace != ""].unique().tolist()
    u_idx = {k: i for i, k in enumerate(uniq)}
    uf = UF(len(uniq))
    if vendor_ids is not None:                       # same vendor id => same supplier
        d = pd.DataFrame({"k": keyspace, "v": vendor_ids}).dropna(subset=["v"])
        d = d[d.k != ""]
        for _, g in d.groupby("v"):
            ks = [u_idx[k] for k in g.k.unique()]
            for k in ks[1:]:
                uf.union(ks[0], k)
    if len(uniq) > 1 and sim_thr <= 1.0:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        X = vec.fit_transform(uniq)
        B = 2000
        for s in range(0, len(uniq), B):
            S = (X[s:s + B] @ X.T).tocoo()
            for i, j, v in zip(S.row + s, S.col, S.data):
                if i < j and v >= sim_thr and uniq[i][:block_chars] == uniq[j][:block_chars]:
                    uf.union(i, j)
    root = {k: uf.find(i) for k, i in u_idx.items()}
    # label = most frequent normalised raw name of the cluster
    cnt = keyspace[keyspace != ""].value_counts()
    best = {}
    for k, r in root.items():
        if r not in best or cnt[k] > cnt[best[r]]:
            best[r] = k
    out = keyspace.map(lambda k: ("S:" + best[root[k]]) if k in root else "")
    return out


def pool_unmastered(keys, vendor_ids, label="S:UNMASTERED"):
    """Clusters with no master-data vendor ID on any row are pooled into one catch-all supplier (one-time / unregistered
    vendors), mirroring how the client's golden supplier master treats them."""
    has_id = vendor_ids.notna().groupby(keys).transform("any")
    return keys.where(has_id | (keys == ""), label)
