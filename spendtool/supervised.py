"""Learn from labelled examples when a company has them (A: labels_A, B: seed_labels). Optional: skipped for zero-label companies.
Features per line: description words + char n-grams, supplier name token, vendor id, item code, and any `extra` columns of the config."""
import os, re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from .ingest import load_all
from .suppliers import norm_name


def _tokens(df):
    """One 'document' per line: structured tokens the linear model can weight."""
    cols = [c for c in df.columns if c.startswith("x_")]
    toks = []
    sup = df["supplier_raw"].map(norm_name).str.replace(" ", "", regex=False)
    toks.append("sup_" + sup)
    toks.append("ven_" + df["vendor_id"].fillna("na").astype(str))
    toks.append("code_" + df["item_code"].fillna("na").astype(str))
    for c in cols:
        toks.append(c + "_" + df[c].fillna("na").astype(str).str.replace(r"\s+", "_", regex=True))
    out = toks[0]
    for t in toks[1:]:
        out = out + " " + t
    return out


class Supervised:
    def __init__(self, C=0.5):
        self.C = C

    def _fit_vec(self, df):
        self.vw = TfidfVectorizer(token_pattern=r"[a-z0-9_]{2,}", sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=150000)
        self.vt = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, min_df=1, lowercase=False)

    def _X(self, df, fit=False):
        d = df["desc"].fillna("")
        t = _tokens(df)
        if fit:
            self._fit_vec(df)
            return sparse.hstack([self.vw.fit_transform(d), self.vc.fit_transform(d) * 0.7, self.vt.fit_transform(t)]).tocsr()
        return sparse.hstack([self.vw.transform(d), self.vc.transform(d) * 0.7, self.vt.transform(t)]).tocsr()

    def fit(self, df, paths):
        """paths: Series of tuples (one leaf path per row)."""
        self.classes_ = sorted(set(paths))
        ix = {p: i for i, p in enumerate(self.classes_)}
        y = paths.map(ix).values
        X = self._X(df, fit=True)
        self.temp = self._calibrate(X, y)
        self.m = LinearSVC(C=self.C, max_iter=3000).fit(X, y)
        # exact item-code memory (strong for repeat purchases of the same material)
        d = pd.DataFrame({"c": df["item_code"].values, "y": y}).dropna()
        d = d[d.c.astype(str).str.len() > 0]
        mode = d.groupby("c")["y"].agg(lambda v: v.value_counts().index[0])
        purity = d.groupby("c")["y"].agg(lambda v: v.value_counts().iloc[0] / len(v))
        self.code_map = mode[purity >= 0.9].to_dict()
        return self

    def _calibrate(self, X, y, seed=0):
        """Pick the softmax sharpness on a 15% hold-out so that confidence tracks accuracy."""
        rng = np.random.RandomState(seed)
        te = rng.rand(len(y)) < 0.15
        if te.sum() < 50 or len(set(y[~te])) < 2:
            return 3.0
        m = LinearSVC(C=self.C, max_iter=3000).fit(X[~te], y[~te])
        s = m.decision_function(X[te])
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        best, bk = 1e9, 3.0
        pos = {c: i for i, c in enumerate(m.classes_)}
        yt = np.array([pos.get(v, -1) for v in y[te]])
        ok = yt >= 0
        for k in (1, 2, 3, 5, 8, 12, 20, 30):
            Z = np.exp((s - s.max(1, keepdims=True)) * k)
            P = Z / Z.sum(1, keepdims=True)
            ll = -np.log(np.clip(P[np.arange(len(yt))[ok], yt[ok]], 1e-6, 1)).mean()
            if ll < best:
                best, bk = ll, k
        return float(bk)

    def predict(self, df):
        X = self._X(df)
        s = self.m.decision_function(X)
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        Z = np.exp((s - s.max(1, keepdims=True)) * getattr(self, "temp", 3.0))
        P = Z / Z.sum(1, keepdims=True)
        return P


def load_training(cfg, train_dir, tax, log=print):
    spec = cfg.get("labels") or cfg.get("seed_labels")
    if not spec or not train_dir:
        return None
    lp = os.path.join(train_dir, spec["file"])
    if not os.path.exists(lp) or not all(os.path.exists(os.path.join(train_dir, s["file"])) for s in cfg["sources"]):
        log("supervised: training files not found, skipping"); return None
    df = load_all(cfg, train_dir)
    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
    lab = lab.set_index(spec["id_col"])
    lv = spec["level_cols"]
    paths = {}
    for rid, row in lab[lv].iterrows():
        paths[rid] = tuple(x.strip() for x in row if x and x.strip())
    leafset = set(tax.leaves)
    df["_path"] = df["record_id"].map(paths)
    df = df[df["_path"].notna() & df["_path"].isin(leafset)]
    log(f"supervised: {len(df)} labelled training lines")
    return df
