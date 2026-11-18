# Approach

**AI tools used:** Claude (Anthropic) helped write and review the code during the build. The tool itself calls no paid API and needs no key.
An optional LLM labeller (`spendtool/llm.py`, Gemini/Groq/Anthropic via env var) exists but is off unless a key is set; it was not used for any result below and was only plumbing-tested with a mock model.

## Pipeline (one command, config per company)
ingest (YAML config: files, columns, formats) -> clean amounts / VAT / dates / FX -> supplier clustering -> exclusions
-> count-once matching (two-system companies) -> item de-duplication -> classification -> output CSV.
No category names in code; the tree is read from a file at run time.

## Matching and de-duplication
- **Tool vs ERP (A):** PO + PO item parsed from messy references, net amount within 5 cents, then 2%, then several receipts summed against one invoice line. Lines with no usable PO fall back to supplier cluster + amount + description similarity. True pairs have dates within ~35 days, so fallback pairs more than 90 days apart are rejected; undated pairs need higher text similarity. Matched tool lines are DUPLICATE; repeated exports of one invoice (same invoice, text, amount, qty, PO item) are DUPLICATE too. A result: 23,712 duplicates, 4,305 tool-only lines counted, 315 not approved, 188 cancelled.
- **Suppliers:** normalised names (legal words and spaces removed), ERP vendor IDs link names, char n-gram fuzzy merge. Clusters with no vendor ID anywhere are pooled as one catch-all, as in the client's own master. Against labels_A: ARI 0.973 (homogeneity 1.0).
- **Items:** per-company policy by ERP spend type (code / description / pooled), near-duplicate descriptions merged only when their numbers agree. Against labels_A: ARI 0.997.
- **Exclusions:** status columns, plus regex rules in the config. B rules were derived from the 3,000 seed labels: recall 100%, precision 99.1% in-sample.

## Classification
1. **With labels (A, B):** linear SVM on description words and char n-grams, supplier, vendor ID, item code and extra columns, plus an exact item-code memory. Confidence is calibrated on a hold-out. Cross-validated (labels from the same company):
   A, random split: L1 99.9%, L2 99.3%, L3 98.5%. A, split by material code (unseen items): L1 98.5%, L2 96.6%, L3 95.5%.
   B, 5 held-out analyst labels: L1 97%, L3 95-97% (spend-weighted 97-99%).
2. **No labels (C, and A with labels hidden):** tf-idf over the tree paths expanded with a built-in procurement concept lexicon, blended with a local multilingual sentence-embedding model if `sentence-transformers` is installed (free, no key; not available in my build sandbox, so **never measured**), then smoothed by supplier.
   Measured on A with labels hidden (tf-idf + lexicon, no embeddings): L1 45% of lines / 61% of spend, L2 18% / 27%, L3 15% / 21%. This is weak on abbreviated mining text; expect a generic indirect-spend tree to do better.

## Runtime and cost
A (65.5k records, trains on 37k labelled lines): 66 s. B (41k lines): 12 s. Dry run on a synthetic 11k-line company with an unseen 4-level tree: 3 s. Cost: none (no API).
