# Cantill.ai Data & ML task: spend pipeline

One command per company (data files are NOT in this repo; put them in `data/`, which is git-ignored):

    pip install -r requirements.txt
    python run.py --config configs/company_A.yaml --data-dir data --out output/yourname__A.csv

A new company needs a new YAML config in `configs/`, not new code. See `configs/company_A.yaml` for the schema.

Pipeline: ingest (config) -> clean amounts/dates/VAT/FX -> supplier clustering -> exclusions ->
two-system count-once matching -> item keys -> classification -> output CSV.

Status: ingest, cleaning, supplier clustering, exclusions and A matching are working (A runs in ~26 s).
Status: complete for A and B; Company C is run with a new config at the blind test (see below).

## Classifier backends (best available is used automatically)
1. **Labels** (A: `labels_A.csv`, B: `seed_labels.csv`): a supervised model learns from them. Keep the ORIGINAL data + labels in `data/`.
2. **No labels** (Company C): tf-idf over the tree + built-in concept lexicon; if `pip install sentence-transformers` works on your machine
   (free model download, no key) a multilingual embedding model is blended in automatically.
3. Optional LLM labeller if `GEMINI_API_KEY` / `GROQ_API_KEY` / `ANTHROPIC_API_KEY` is set (off by default, not needed).
`--zero-label` ignores labels (to measure the no-label path); `--train-dir` points at the folder holding the original files + labels.

## Blind-test commands
    # A2 / B2: new files in data_A2/ and data_B2/ (same layouts); training data stays in data/
    python run.py --config configs/company_A.yaml --data-dir data_A2 --train-dir data --out submission/yourname__A2.csv
    python run.py --config configs/company_B.yaml --data-dir data_B2 --train-dir data --out submission/yourname__B2.csv
    # C: draft a config, fix it, run it
    python suggest_config.py --data-dir data_C --file <file> --taxonomy <tree> --company C > configs/company_C.yaml
    python run.py --config configs/company_C.yaml --data-dir data_C --out submission/yourname__C.csv
    python validate_output.py submission/yourname__C.csv --config configs/company_C.yaml --data-dir data_C
Copy `configs/company_C.yaml` into `submission/` too. If A2/B2 file names differ from the originals, edit only the `file:` fields in the config copy you use.
