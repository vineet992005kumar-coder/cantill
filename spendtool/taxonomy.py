import os
import numpy as np
import pandas as pd


class Taxonomy:
    """Tree read from a CSV at run time. Level columns come from the config (or all columns of the file, in order)."""
    def __init__(self, path, level_cols=None):
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
        self.cols = level_cols or list(t.columns)
        t = t[self.cols].apply(lambda c: c.str.strip())
        t = t.drop_duplicates().reset_index(drop=True)
        self.depth = len(self.cols)
        # a leaf is the deepest non-empty path; drop internal rows that are prefixes of longer paths
        paths = [tuple(x for x in r if x) for r in t.itertuples(index=False)]
        pset = set(paths)
        leaves = [p for p in pset if not any(q != p and q[:len(p)] == p for q in pset)]
        self.leaves = sorted(leaves)
        self.leaf_text = [" ".join(dict.fromkeys(p)) for p in self.leaves]   # unique names along the path

    def row(self, i):
        p = self.leaves[i]
        return list(p) + [""] * (self.depth - len(p))
