# Decisions

## Company A: data findings and choices
- **PO references** appear as `4510068836/10.0`, `PO#4510055507`, `PO 4510066595` or bare numbers. Parsed to PO + item; 5,199 tool lines have none.
- **Matching tool -> ERP**: (1) PO (+item) with net amount within 5 cents, (2) PO with 2% tolerance, (3) several free ERP receipts summed against one invoice line, (4) fallback on same supplier cluster + net amount (2%) + description similarity. Matched tool lines are excluded as DUPLICATE; the ERP record is counted.
- **Fallback guard**: true PO matches have invoice and receipt dates within ~35 days (99.9% within 120 days). Fallback pairs whose dates are known and more than 90 days apart are rejected. When a date is missing a stricter text similarity (0.5 instead of 0.35) is required.
- **Export repeats in the tool file**: identical invoice + text + net + quantity + supplier AND the same PO item (or no PO item). Lines that share text and amount but carry different PO items are different purchases, so they are kept.
- **VAT**: `15`, `0.15`, `15%` all parsed; blank -> 15%; an explicit 0 stays 0.
- **Statuses**: only Approved/Paid count. Cancelled -> CANCELLED; any other status -> NOT_APPROVED. Precedence: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
- **Negative ERP values (434 lines)** are kept as negative spend (returns/reversals reduce spend). labels_A classifies them like normal lines.
- **Suppliers**: names normalised (legal suffixes removed, spaces removed), ERP vendor IDs link names, then char-n-gram fuzzy merge. On A this reproduces the golden key for 69 of 70 suppliers exactly (homogeneity 1.0).
  The client's golden key SG-f85711 lumps 122 different names (20,891 ERP lines). Those names are exactly the ones that never carry a vendor ID anywhere in the ERP file: one-time / unregistered vendors that the client pools into a catch-all. Decision: clusters with no vendor ID on any row are pooled into one `S:UNMASTERED` key (config flag `suppliers.pool_unmastered`, on for A only). Measured on A's ERP lines against labels_A: ARI 0.973, homogeneity 1.0 (without the flag: ARI 0.13).
- **Items**: golden items follow the ERP's `spend_type`. Material lines: one item per material number. Service lines: the code differs per PO but the description repeats, so one item per description (this merges different codes that mean the same thing). Free Text and Catalogue lines have no real item identity and the client pools them into one catch-all item. Configured in `items:` of the A config. ARI vs golden 0.997 (item keys exist only for lines with a material number). Unmatched tool lines reuse an ERP item key when the description is identical, otherwise get a description-based key.
- These conventions were read off labels_A, so the numbers above are in-sample. They rest on the client's own master-data pattern, not on individual records.

## Company B: data findings and choices
- **Dates**: `invoice_date` is unreliable (e.g. 2018 on a 2024 posting), so `posting_date` drives the FX month and all date logic.
- **Currency**: `currency_rate` is units of line currency per 1 USD, so USD = amount / rate. USD lines are unchanged. 3,343 non-USD lines have no rate and use the monthly `fx_rates_usd.csv` (nearest month if the exact month is missing). Three lines in currency `RCN` have no rate anywhere: they are flagged `fx_missing` and carry spend 0.
- **Suppliers**: only vendor IDs exist, so the vendor ID is the supplier key and fuzzy name merging is switched off (it would merge V500009 with V500010).
- **Exclusions**: B has no status column, so only NON_PROCUREMENT applies. Rules were derived from the 3,000 analyst seed labels and checked against them: GL descriptions `I/C Payable`, `IC Payables`, `IC<number>`; supplier types 5406, 2742 and blank; descriptions mentioning taxes/GST/sales tax; descriptions of the form `sl-nnnnn-nnnn-nnnn-`. On the seeds: recall 100%, precision 99.1%. This is in-sample, so real recall on unseen lines will be somewhat lower.
- **Customs duty / import duty is kept as spend**: the analyst labelled it as spend and the tree has a "Customs Clearance & Duties" category. VAT/income tax lines are excluded.
- **Reversal pairs** (same invoice, equal and opposite amounts) are kept; they net to zero.

## Cross-company choices
- Legal-form words (pty, ltd, limited, llc, inc, gmbh, ...) are ignored when matching supplier names, so "Alpha Foods Ltd" and "Alpha Foods LLC" become one supplier. This is needed for typing variants such as "(PTY) LTD" vs "PTY LTD", but it can merge genuinely separate legal entities with the same trading name.

## Classification choices
- Labels exist for A and B, so a supervised model is used there (the brief allows it; only C is label-free). The model is retrained from the labels at each run, so no trained artefact or data is stored in the repo.
- Confidence: SVM scores are turned into probabilities with a temperature picked on a hold-out so confidence tracks accuracy; lines whose item code was seen with a consistent category get 0.98.
- Zero-label quality is limited without a language model. The built-in lexicon helped only a little (A L1 36% -> 45% of lines). Embeddings are used when installed but were not measurable here.
- Lines excluded from spend get no category (blank l1..l4).
- Known limits: undated tool lines matched by fallback rest on text similarity; 3 lines in an unknown currency (`RCN`) carry spend 0.
