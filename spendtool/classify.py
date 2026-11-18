"""Zero-label classification into a tree read at run time.
Backends (best available is used, all tree-driven, no category names in code):
  1. optional labelled examples (seed labels) -> kNN over item text, blended with the tree score
  2. optional local sentence embeddings (sentence-transformers) if installed
  3. TF-IDF char/word similarity between item text and the leaf paths (always available)
A supplier prior then smooths scores: lines from one supplier tend to sit in a few categories."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _softmax(S, t=0.05):
    Z = (S - S.max(1, keepdims=True)) / t
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _tfidf_scores(texts, leaf_text, extra_corpus=None):
    corpus = list(texts) + list(leaf_text) + list(extra_corpus or [])
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit(corpus)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9]{2,}").fit(corpus)
    def enc(x):
        return sparse.hstack([vc.transform(x), vw.transform(x)]).tocsr() * 0.5
    X, L = normalize(enc(texts)), normalize(enc(leaf_text))
    return (X @ L.T).toarray()


EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _embed_scores(texts, leaf_text, model_name=EMBED_MODEL):
    from sentence_transformers import SentenceTransformer      # optional dependency
    m = SentenceTransformer(model_name)
    E = m.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    L = m.encode(list(leaf_text), normalize_embeddings=True)
    return E @ L.T


def classify(df, tax, cfg=None, seed=None, log=print):
    """df needs columns desc, supplier_key, item_code, include. Returns (l-columns frame, confidence Series).
    seed: optional DataFrame with columns text + leaf_idx (labelled examples) for kNN blending."""
    cfg = cfg or {}
    inc = df[df["include"] == 1]
    text = (inc["desc"].fillna("")).where(inc["desc"].fillna("").str.len() > 0, "")
    key = pd.DataFrame({"t": text.values, "s": inc["supplier_key"].values}, index=inc.index)
    ut = key["t"].drop_duplicates().tolist()
    tix = {t: i for i, t in enumerate(ut)}
    leaf_text = [x.lower() for x in tax.leaf_text]
    from .lexicon import concept_tokens
    backend = cfg.get("classifier", {}).get("backend", "auto")
    ut_aug = [t + " " + concept_tokens(t) for t in ut]
    leaf_aug = [t + " " + concept_tokens(t) * 2 for t in leaf_text]
    S_t = _tfidf_scores(ut_aug, leaf_aug)                       # tf-idf + shared concept tokens (always available)
    S_e = None
    if backend in ("auto", "embed"):
        try:                                                     # free local model, downloaded once; no API key
            S_e = _embed_scores(ut, leaf_text, cfg.get("classifier", {}).get("embed_model", EMBED_MODEL))
        except Exception as ex:
            if backend == "embed":
                raise
            log(f"classifier: embeddings unavailable ({type(ex).__name__}), using tf-idf + lexicon")
    if S_e is not None:
        z = lambda M: (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        we = cfg.get("classifier", {}).get("embed_weight", 0.6)
        S = we * z(S_e) + (1 - we) * z(S_t)
        P = _softmax(S, 1.0)
        log("classifier: local embeddings + tf-idf/lexicon over tree paths")
    else:
        S = S_t
        P = _softmax(S, cfg.get("classifier", {}).get("temp", 0.05))
        log("classifier: tf-idf + lexicon over tree paths")
    from . import llm
    if backend in ("auto", "llm") and llm.available():
        # rank unique (supplier, text) pairs by spend share is not needed: label all, cached on disk
        pairs = key.drop_duplicates()[["s", "t"]]
        items = [(a, b) for a, b in zip(pairs["s"], pairs["t"])]
        labelled = llm.label_items(tax.leaves, items, call_fn=cfg.get("_llm_call"), log=log)
        lab = {it: r for it, r in zip(items, labelled)}
        log(f"classifier: LLM ({llm.provider()}) on {sum(r is not None for r in labelled)}/{len(items)} unique items")
        PLm = np.zeros((len(key), P.shape[1]))
        row_ix0 = key["t"].map(tix).values
        for n, (s_, t_) in enumerate(zip(key["s"], key["t"])):
            r = lab.get((s_, t_))
            if r is not None:
                PLm[n, r[0]] = r[1]
        has = PLm.sum(1) > 0
        PL0 = P[row_ix0]
        PL0[has] = 0.8 * PLm[has] + 0.2 * PL0[has]          # LLM dominates; local score breaks ties
        P_line = PL0
    else:
        P_line = None
    # line-level probabilities then supplier prior
    row_ix = key["t"].map(tix).values
    PL = P_line if P_line is not None else P[row_ix]
    w = cfg.get("classifier", {}).get("supplier_weight", 0.5)
    if w > 0:
        sup = pd.DataFrame(PL, index=key.index).groupby(key["s"]).mean()
        prior = sup.loc[key["s"]].values
        PL = (1 - w) * PL + w * prior
    best = PL.argmax(1)
    srt = np.sort(PL, axis=1)
    conf = np.clip(srt[:, -1] - 0.5 * srt[:, -2], 0, 1)
    rows = np.array([tax.row(i) for i in best])
    out = pd.DataFrame(rows, index=inc.index, columns=[f"l{i+1}" for i in range(tax.depth)])
    return out, pd.Series(conf, index=inc.index), PL
