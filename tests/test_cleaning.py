import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd, numpy as np
from spendtool.cleaning import parse_amount, parse_vat, parse_date, parse_po


def test_amount():
    r = parse_amount(pd.Series(["5,266.63", "1.234,56", "(100.00)", "abc"]))
    assert r[0] == 5266.63 and r[1] == 1234.56 and r[2] == -100.0 and np.isnan(r[3])


def test_vat():
    assert parse_vat(pd.Series(["15", "0.15", "15%", None, "0"]), 0.15).tolist() == [0.15, 0.15, 0.15, 0.15, 0.0]


def test_dates():
    d, f = parse_date(pd.Series(["19/01/2024", "2022-01-29", "27-Mar-24", None, "31/02/2022"]))
    assert d.dt.strftime("%Y-%m-%d").tolist()[:3] == ["2024-01-19", "2022-01-29", "2024-03-27"]
    assert f.tolist()[3:] == ["missing", "unparsable"]


def test_po():
    po, it = parse_po(pd.Series(["4510068836/10.0", "PO#4510055507", "PO 4510066595", None]))
    assert po.tolist()[:3] == ["4510068836", "4510055507", "4510066595"] and it[0] == "10"
