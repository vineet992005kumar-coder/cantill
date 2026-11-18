"""Parsing and normalisation helpers (amounts, VAT, dates, text, PO refs). No company-specific logic."""
import re
import numpy as np
import pandas as pd

_NUM_RE = re.compile(r"[^\d,.\-+()]")


def parse_amount_one(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = _NUM_RE.sub("", str(x).strip())
    if not s:
        return np.nan
    neg = (s.startswith("(") and s.endswith(")")) or s.endswith("-")
    s = ("-" + s.strip("()+-")) if s.startswith("-") else s.strip("()+-")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:                                     # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = (head.replace(",", "") + "." + tail) if len(tail) in (1, 2) else s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -abs(v) if neg else v


def parse_amount(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(parse_amount_one).astype(float)


def parse_vat(series, default):
    """Accepts 15, 0.15, '15%', blank -> default. Explicit 0 stays 0."""
    def one(x):
        if pd.isna(x) or str(x).strip() == "":
            return default
        v = parse_amount_one(str(x).replace("%", ""))
        if np.isnan(v):
            return default
        return v / 100.0 if v > 1 else v
    return series.map(one).astype(float)


DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%d.%m.%Y", "%Y/%m/%d",
                "%Y%m%d", "%d-%m-%Y", "%d %b %Y", "%b %d, %Y", "%m/%d/%Y"]


def parse_date(series, formats=None, min_year=2000, max_year=2030):
    """Return (dates, flag). flag in {'', 'missing', 'unparsable', 'out_of_range'}."""
    formats = formats or DATE_FORMATS
    s = series.astype("string").str.strip()
    s = s.where(~s.isin(["", "nan", "NaN", "None", "NaT", "0", "00000000"]), pd.NA)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for f in formats:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        src = s[todo].str.slice(0, 10) if f in ("%Y-%m-%d", "%Y/%m/%d") else s[todo]
        out[todo] = pd.to_datetime(src, format=f, errors="coerce")
    flag = pd.Series("", index=series.index)
    flag[s.isna()] = "missing"
    flag[s.notna() & out.isna()] = "unparsable"
    bad = out.notna() & ((out.dt.year < min_year) | (out.dt.year > max_year))
    flag[bad] = "out_of_range"
    out[bad] = pd.NaT
    return out, flag


def norm_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9% ,.;:/\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_PO_RE = re.compile(r"(?<!\d)(\d{8,12})(?!\d)(?:\s*[/\-]\s*(\d{1,6})(?:\.0+)?)?")


def parse_po(series):
    """'4510068836/10.0', 'PO#4510055507', 'PO 4510066595', 4510001342.0 -> (po, item) strings or None."""
    po, item = [], []
    for x in series:
        if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
            po.append(None); item.append(None); continue
        s = str(x).strip()
        if re.fullmatch(r"\d+\.0+", s):
            s = re.sub(r"\.0+$", "", s)
        m = _PO_RE.search(s)
        if m:
            po.append(m.group(1)); item.append(m.group(2))
        else:
            po.append(None); item.append(None)
    return pd.Series(po, index=series.index), pd.Series(item, index=series.index)


def norm_item_no(series):
    """10.0 / '10' / '00010' -> '10'"""
    def one(x):
        if pd.isna(x):
            return None
        s = re.sub(r"\.0+$", "", str(x).strip()).lstrip("0")
        return s or "0"
    return series.map(one)
