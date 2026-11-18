import os
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("mode", "one_system")
    cfg.setdefault("status", {})
    cfg.setdefault("matching", {})
    cfg.setdefault("non_procurement_rules", [])
    cfg.setdefault("use_default_keyword_rules", True)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    for need in ("company", "reporting_currency", "sources", "taxonomy"):
        if need not in cfg:
            raise ValueError(f"config missing '{need}'")
    return cfg
