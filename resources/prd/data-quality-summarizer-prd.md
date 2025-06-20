# Product Requirements Document (PRD)

## 1 Purpose
Design and implement an **offline summariser** that ingests a CSV (~100 k rows) containing raw data-quality check results and produces:

1. A *row-oriented* summary CSV (`full_summary.csv`).
2. A natural-language digest text file (`nl_all_rows.txt`) containing one sentence per summary row.

These artefacts will be indexed as the **knowledge base** for a downstream LLM-powered chatbot, enabling rich Q&A on dataset health without ever exposing the original 100 k-row file.

---

## 2 Context & Goals
* Operates on consumer-grade machines (<8 GB RAM).
* Avoids sending the raw CSV to the LLM.
* Enables both high-level conversation (trend, worst rules) and deep drill-down (any rule, any dataset) entirely from the prepared artefacts.

---

## 3 Input Specifications
| Column | Type | Notes |
|--------|------|-------|
| `source` | string | Data source system |
| `tenant_id` | string | Tenant / client identifier |
| `dataset_uuid` | string | Stable dataset key |
| `dataset_name` | string | Human-readable name |
| `business_date` | string (ISO) | Row-level execution date |
| `dataset_record_count` | number | Total rows in dataset (raw) |
| `rule_code` | number | Foreign key to rule-metadata mapping |
| `level_of_execution` | string | ATTRIBUTE / DATASET etc. |
| `attribute_name` | string | Populated when `level_of_execution = ATTRIBUTE` |
| `results` | JSON (string) | e.g. `{ "result":"Pass" , ... }` |
| `context_id` | string | Upstream batch context |
| `filtered_record_count` | number | Rows considered after filters |

### Rule-Metadata Mapping (separate JSON)
Keyed by `rule_code`, provides:
`rule_name , rule_type , dimension , rule_description , category (1-4)`.

---

## 4 Processing Pipeline
1. **Chunked Ingestion**
   * Use **pandas** with `dtype` map and `chunksize≈20 000`.
   * Reason: keep memory footprint <1 GB and leverage familiar API.
2. **Streaming Aggregation**
   * Accumulator key: `(source, tenant_id, dataset_uuid, dataset_name, rule_code)`.
   * For each row update counts:
     * `pass_count_total`, `fail_count_total`.
     * Rolling windows ending at **latest `business_date` in the file**:
       * 1-month (30 days), 3-month (90 days), 12-month (365 days).
3. **Enrichment**
   * Join accumulator with rule-metadata.
   * Compute derived metrics & insights:
     * Fail-rates for each window.
     * `trend_flag` ↑/↓/= based on `fail_rate_1m` vs `fail_rate_3m`.
     * `last_execution_level` = most frequent `level_of_execution`.
4. **Export Artefacts** (see §6).

---

## 5 Output Schema – `full_summary.csv`
| # | Column | Description |
|---|---------|-------------|
| 1 | `source` | Data source system |
| 2 | `tenant_id` | Tenant identifier |
| 3 | `dataset_uuid` | Dataset UUID |
| 4 | `dataset_name` | Dataset name |
| 5 | `rule_code` | Rule identifier |
| 6 | `rule_name` | e.g. *ROW_COUNT* |
| 7 | `rule_type` | DATASET / ATTRIBUTE |
| 8 | `dimension` | Correctness, etc. |
| 9 | `rule_description` | Verbose description |
| 10 | `category` | Category 1–4 |
| 11 | `business_date_latest` | Max `business_date` seen for this key |
| 12 | `dataset_record_count_latest` | From latest row |
| 13 | `filtered_record_count_latest` | From latest row |
| 14 | `pass_count_total` | Cumulative passes |
| 15 | `fail_count_total` | Cumulative fails |
| 16 | `pass_count_1m` | Passes in last 30 days |
| 17 | `fail_count_1m` | Fails in last 30 days |
| 18 | `pass_count_3m` | Passes in last 90 days |
| 19 | `fail_count_3m` | Fails in last 90 days |
| 20 | `pass_count_12m` | Passes in last 365 days |
| 21 | `fail_count_12m` | Fails in last 365 days |
| 22 | `fail_rate_total` | `fail_count_total / (pass+fail)_total` |
| 23 | `fail_rate_1m` | Analogous |
| 24 | `fail_rate_3m` | Analogous |
| 25 | `fail_rate_12m` | Analogous |
| 26 | `trend_flag` | ↑ if `fail_rate_1m` > `fail_rate_3m` + ε; ↓ if < −ε; = otherwise |
| 27 | `last_execution_level` | Most common `level_of_execution` for key |

Total rows ≈  `(# distinct datasets) × (# distinct rule_code)`.

---

## 6 Generated Artefacts & Locations
| File | Format | Purpose |
|------|--------|---------|
| `artifacts/full_summary.csv` | CSV | Authoritative, row-oriented summary; audit & exploration |
| `artifacts/nl_all_rows.txt` | UTF-8 text | LLM-friendly prompt sentences (one per CSV row) |

> Both files reside in `resource/artifacts/` (created if absent).

### NL Sentence Template (per row)
```
• On {business_date_latest}, dataset “{dataset_name}” (source: {source}, tenant: {tenant_id}, UUID: {dataset_uuid}) under rule “{rule_name}” [{rule_code}] recorded {fail_count_total} failures and {pass_count_total} passes overall (fail-rate {fail_rate_total:.2%}; 1-month {fail_rate_1m:.2%}, 3-month {fail_rate_3m:.2%}, 12-month {fail_rate_12m:.2%}) — trend {trend_flag}.
```

Example
```
• On 2025-02-24, dataset “EFGH_Dataset” (source: ABC System, tenant: XYZC, UUID: 2345sdfs) under rule “ROW_COUNT” [202] recorded 2 failures and 1 589 passes overall (fail-rate 0.13 %; 1-month 0.13 %, 3-month 0.08 %, 12-month 0.05 %) — trend ↑.
```

---

## 7 Performance & Memory Considerations
* **Chunk size** tunable; default 20 k rows balances I/O and overhead.
* `dtype` mapping prevents expensive type inference.
* Only accumulator dict lives in memory (~20 datasets × rules ≪ 1 MB).

---

## 8 Logging & Observability
* **INFO** – start/end of each chunk, current accumulator size.
* **DEBUG** – periodic sample accumulator entry.
* **WARN** – unrecognised `rule_code`, malformed `results` JSON.
* Use `structlog` or `logging` with JSON handler for easy grep.

---

## 9 Folder & Module Layout
```
src/
  ingestion.py        # chunked CSV reader
  aggregator.py       # streaming aggregation & metric calc
  rules.py            # rule-metadata loader & utils
  summarizer.py       # CSV + NL generation
  __main__.py         # CLI entry point
resource/
  artifacts/          # full_summary.csv, nl_all_rows.txt
  prd/                # this PRD
  prompts/            # design discussion etc.
```

---

## 10 Non-Functional Requirements
* **Runtime** < 2 min for 100 k rows on 4-core laptop.
* **Memory** < 1 GB.
* **File size** – summary CSV < 2 MB typical.
* **Code quality** – unit tests ≥ 80 % coverage; follow logging & exception-handling guidelines.

---

## 11 Out of Scope
* Real-time ingestion (streaming beyond file batch).
* Writing back to databases or analytics warehouses.
* Direct API calls to LLM during summariser run.

---

## 12 Open Questions (none)
All prior discussion points have been incorporated.

---

*Document version: v1.0 – 2025-06-20*
