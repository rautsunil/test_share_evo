# Semantic Layer Architecture — Design Concept & Contribution Note

**Status:** Concept / Reference Design — illustrative, not yet built end-to-end | **Doc type:** Architecture Discussion Paper

> This is a design concept, worked through in detail using a CRM scenario as a concrete, easy-to-follow example — not a description of an existing production system. It's shared as a reference point for discussion, and as a starting point for contributing to a broader, domain-agnostic semantic layer effort.

> Paste this directly into a Confluence page. Headings map to Confluence's heading styles (H1/H2/H3), and every table below renders as a native Confluence table. Suggested page tree: make each **H2** its own child page if the page gets too long for one screen.

---

## 1. Executive Summary

Today, when a Marketing Manager asks "show me premium customers in Korea likely to upgrade to Galaxy Fold," that question has to travel through several hands — an analyst writes SQL, checks with someone about what "premium" even means, waits for a data pull, and often gets a slightly different answer than the last person who asked a similar question last month.

This paper walks through a **Semantic Layer** design: a single, governed definition of what business terms mean (Customer, Premium, Upgrade Propensity, CLV...), sitting between AI/BI tools and a data warehouse. An LLM assistant turns a plain-English question into that shared vocabulary, a set of engines translates the vocabulary into safe, optimized SQL, and the warehouse returns the answer — all using **one** definition of a term like "premium customer" everywhere, every time.

The example throughout uses a CRM scenario purely because it's concrete and easy to reason about. The pattern itself — entities, metrics, rules, relationships, dimensions, and a resolution/orchestration/SQL-generation pipeline in front of them — is domain-agnostic. Section 10 below outlines how the same skeleton extends to a general-purpose semantic layer covering any business domain, not just CRM.

**In one sentence:** stop re-deriving business logic in every dashboard and every prompt — define it once, govern it, and let both humans and AI agents query it safely.

---

## 2. Problem Statement & Goals

| Problem today | What this architecture fixes |
|---|---|
| Every analyst/dashboard defines "CLV" or "premium" slightly differently | One governed metric/entity catalog (Component 7) |
| LLMs hallucinate table names, joins, or numbers when asked to write SQL directly | LLM never writes SQL — it only expresses *intent*; engines resolve it against the semantic model |
| No consistent security enforcement across BI tools and AI agents | Row/column-level security enforced centrally in the SQL Generation Engine |
| Business context (brand rules, compliance docs) lives in slide decks, not in the pipeline | RAG-based Business Knowledge layer (Component 8) makes it queryable |
| Customer data is scattered across Samsung Account, eStore, CRM, App analytics | Golden Customer 360 (Component 10) unifies it into one trusted record |

**Goals:** single source of business truth, governed & secure access, consistent metrics across BI/AI/apps, faster analytics development, and measurably better decisions.

---

## 3. High-Level Flow (Plain Language)

```
Person asks a question in English
   → LLM figures out what they mean (entities, metrics, filters)
      → Semantic Layer Engine looks up what those terms *officially* mean
         → Orchestration Engine decides which tables/joins are needed
            → SQL Generation Engine writes safe, dialect-correct SQL
               → BigQuery runs it and returns rows
                  → LLM turns rows back into a plain-English answer + chart
```

Everything in the middle is backed by two knowledge stores that don't do computation themselves but feed every engine: the **Semantic Layer** (structured business definitions) and **Business Knowledge / RAG** (unstructured policy & context docs).

---

## 4. Component Deep-Dive

Each component below follows the same template: **Job → Sub-components → Input → Output → Possible implementation approach → Worked example → Evaluation metrics → Guardrails.**

---

### Component 1 — Users & Consumers

**Job:** The people and systems that originate business questions and consume the final answer. This is the *demand side* of the architecture.

| Sub-component | Job | Example |
|---|---|---|
| Marketing Managers | Ask campaign/segment questions in plain English | "Who should get the Fold 7 upgrade offer this month?" |
| Data Analysts | Ask deeper, exploratory questions; validate answers | "Break down upgrade propensity by region and age group" |
| CRM Applications | Programmatically request customer scores | Salesforce plugin pulls a customer's propensity score on page load |
| BI / Dashboards | Auto-refresh charts using the same governed metrics | A Looker/Tableau tile bound to the "CLV" metric definition |
| AI Agents / LLMs | Other autonomous agents that need trusted CRM facts | A retention agent asks "is this customer high-risk churn?" |

**Input:** none — this is the origin point. The trigger is a human or system intent.
**Output:** a natural-language (or structured API) question, e.g. `"Show premium customers in Korea likely to upgrade to Galaxy Fold"`. Later it *receives* the output of the whole pipeline: NL summary + chart + table.

**Possible implementation approach:** a chat UI (Slack/Teams bot or in-app assistant), a REST/GraphQL endpoint for CRM apps, and a semantic-layer connector for BI tools (so Tableau/Looker query the same metric catalog instead of raw tables).

**Evaluation metrics:** query success rate, user CSAT on answers, % of questions resolved without human/analyst intervention.
**Guardrails:** SSO/authentication, role-based access before a question is even accepted, per-user/team rate limiting to prevent runaway usage.

---

### Component 2 — LLM / AI Assistant

**Job:** Natural Language Understanding. It does **not** write SQL or touch data — its only job is to turn fuzzy human language into a precise, structured request.

| Sub-component | Job | Example |
|---|---|---|
| Intent understanding | Classify what kind of ask this is (lookup, aggregation, ranking, comparison) | Detects this is a "ranked list of customers" request |
| Entity/metric extraction | Pull out entities, metrics, filters, time windows, segments | `Country=KR`, `metric=upgrade_propensity`, `product=Galaxy Fold` |
| Semantic Engine requester | Package the extracted concepts and hand off to Component 3 | Emits the JSON below |

**Input (hypothetical):**
```
"Show premium customers in Korea likely to upgrade to Galaxy Fold"
```

**Output (hypothetical, sent to Component 3):**
```json
{
  "intent": "ranked_list",
  "entity": "customer",
  "filters": [
    { "field": "country", "op": "=", "value": "KR" },
    { "field": "customer_segment", "op": "=", "value": "premium" },
    { "field": "target_product", "op": "=", "value": "Galaxy Fold" }
  ],
  "metric": "upgrade_propensity_score",
  "sort": "desc",
  "limit": 100
}
```

**Possible implementation approach:** an LLM (Claude/GPT-class model) with tool-use/function-calling, a system prompt containing few-shot examples of good extractions, and a JSON-schema the model is forced to output against — this is the same discipline as structured-output prompting: define the schema, validate on the way out, and reject/retry on malformed output.

**Worked example:** the raw sentence above is parsed, entities are resolved to *field names it doesn't yet know are real* (that's Component 3's job) — the LLM only knows the vocabulary, not the physical schema.

**Evaluation metrics:** intent-classification accuracy, entity-extraction F1 score, hallucination rate (checked via an LLM-as-Judge grader comparing extracted JSON to a gold-labeled test set), clarification-question rate (how often it has to ask "did you mean X?").
**Guardrails:** strict JSON-schema validation before forwarding, confidence threshold — below it, ask a clarifying question instead of guessing, PII scrubbing on the raw input before logging.

---

### Component 3 — Semantic Layer Engine (Business Resolution Engine)

**Job:** The "business dictionary lookup." It takes the LLM's structured request and resolves every term against the **governed** semantic model (Component 7) — checking the term actually exists, means what the requester thinks, and that the requester is allowed to see it.

| Sub-component | Job | Example |
|---|---|---|
| Semantic model loader | Load the current YAML/version-controlled definitions | Loads `customer.yaml`, `metrics.yaml` |
| Concept resolver | Map `"premium"` → the governed rule `CLV > $1000` | Confirms `premium` = `clv_segment = 'premium'` |
| Permission/context validator | Checks the requesting user/role can see this data | Marketing Manager role can see aggregate scores, not raw PII |
| Metadata returner | Emits the fully-resolved, canonical request | See output below |

**Input:** the JSON from Component 2.

**Output (hypothetical):**
```json
{
  "resolved_entity": "customer",
  "resolved_metric": { "id": "upgrade_propensity_score", "table": "feature_store.rec_scores" },
  "resolved_filters": [
    { "field": "country_code", "op": "=", "value": "KR" },
    { "field": "clv_usd", "op": ">", "value": 1000 },
    { "field": "recommended_product", "op": "=", "value": "SM-F958_GalaxyFold7" }
  ],
  "permission": "granted",
  "row_limit_allowed": 5000
}
```

**Possible implementation approach:** a YAML-based, Git-versioned business-concept registry — a schema-agnostic catalog with a resolver service in front of it, plus an RBAC check against the requesting identity. (This is the same registry pattern used in feature-store designs more broadly, not something specific to CRM.)

**Worked example:** `"premium"` is meaningless to a database — this engine is what turns it into `clv_usd > 1000`, consistently, everywhere it's asked.

**Evaluation metrics:** concept-resolution accuracy (does it map to the *correct* governed definition), permission-check latency, cache hit-rate on repeated lookups.
**Guardrails:** deny-by-default access (unknown term/role = reject, don't guess), schema versioning so definition changes don't silently break old dashboards, automatic conflict detection if two rules contradict.

---

### Component 4 — Semantic Orchestration Engine (Query Planning Engine)

**Job:** Turns "what" into "how" — decides which physical tables to touch, how to join them, and how to make the query efficient, *before* any SQL is written.

| Sub-component | Job | Example |
|---|---|---|
| Table/relationship resolver | Determine which tables are needed and how they join | `customer` joins `rec_scores` on `customer_id` |
| Business rule applier | Apply calculation logic (e.g., propensity formula) | Applies the governed CLV formula from Component 7 |
| Logical plan builder | Build a join/filter/aggregation plan (a DAG) | See JSON below |
| Cost/performance optimizer | Push down filters, pick partitions, estimate cost | Prunes to `country_code=KR` partition only |

**Input:** the resolved metadata from Component 3.

**Output (hypothetical logical plan):**
```json
{
  "tables": ["golden_customer_360", "feature_store.rec_scores"],
  "joins": [{ "left": "golden_customer_360.customer_id", "right": "rec_scores.customer_id", "type": "inner" }],
  "filters": ["country_code = 'KR'", "clv_usd > 1000", "recommended_product = 'SM-F958_GalaxyFold7'"],
  "aggregations": [],
  "sort": [{ "field": "upgrade_propensity_score", "dir": "desc" }],
  "estimated_scan_gb": 0.8,
  "estimated_cost_usd": 0.004
}
```

**Possible implementation approach:** a rule-based/cost-aware query planner (comparable to how dbt or a BI semantic layer compiles a metric request into a join plan), with BigQuery dry-run used to estimate bytes scanned before execution.

**Evaluation metrics:** plan-vs-actual cost variance, join correctness rate (caught via automated tests on known query patterns), % of queries needing manual re-planning.
**Guardrails:** hard cap on estimated bytes scanned before a plan is allowed to proceed, max join fan-out limit, query timeout budget.

---

### Component 5 — SQL Generation Engine (SQL Translation Engine)

**Job:** Compiles the logical plan into actual, dialect-correct, secure SQL. This is the *only* component that ever produces SQL text.

**Input:** the logical plan from Component 4.

**Output (hypothetical SQL):**
```sql
SELECT
  c.customer_id,
  c.customer_name,
  r.upgrade_propensity_score
FROM golden_customer_360 c
JOIN feature_store.rec_scores r
  ON c.customer_id = r.customer_id
WHERE c.country_code = 'KR'
  AND c.clv_usd > 1000
  AND r.recommended_product = 'SM-F958_GalaxyFold7'
ORDER BY r.upgrade_propensity_score DESC
LIMIT 100;
```

**Possible implementation approach:** a templated SQL compiler (similar in spirit to dbt's Jinja compilation) with a BigQuery dialect adapter, plus a security-filter injector that automatically appends row-level (`country`, `business unit`) and column-level (mask PII columns for non-privileged roles) predicates.

**Evaluation metrics:** SQL validity rate (parses & runs without error), % of queries requiring a security-filter injection that were caught correctly, execution success rate.
**Guardrails:** parameterized templates only (no raw string concatenation → no injection risk), a strict **read-only** allowlist (no `INSERT`/`UPDATE`/`DELETE`/`DROP` ever generated), automatic row/column security filter injection that cannot be bypassed by the LLM's request.

---

### Component 6 — Data Warehouse (BigQuery)

**Job:** Executes the SQL and returns results. The engine of record.

**Input:** the SQL string from Component 5.
**Output (hypothetical result set):**

| customer_id | customer_name | upgrade_propensity_score |
|---|---|---|
| C-10293 | Ji-Hoon Kim | 0.93 |
| C-10442 | Soo-Yun Park | 0.89 |
| C-10871 | Min-Jae Lee | 0.87 |

**Possible implementation approach:** BigQuery datasets, partitioned by date/country and clustered by customer_id for cost efficiency; IAM roles bound to service accounts per calling engine.
**Evaluation metrics:** query latency (p50/p95), cost per query, uptime/error rate.
**Guardrails:** IAM least-privilege per service account, column-level security policies, budget/quota alerts, slot reservation so one bad query can't starve others.

---

### Component 7 — Semantic Layer (Business Meaning & Logic)

This is the **shared dictionary** everything else reads from — not a compute engine, a governed knowledge store.

| Sub-component | Job | Example |
|---|---|---|
| Business Entities | Canonical list of "things" the business talks about | Customer, Product, Campaign, Order, Device, Recommendation |
| Business Metrics | Formula + definition for every KPI | `CLV = SUM(order_value) over lifetime`; Upgrade Propensity, Email CTR |
| Business Rules | Conditions that encode business logic | "Premium Customer" = `CLV > $1000`; "Campaign Eligible" = no contact in 30 days |
| Relationships | How entities join (cardinality, keys) | `Customer 1—N Order`, `Customer 1—N Device` |
| Dimensions | Attributes used to slice/filter | Country, Age Group, Device Family, Channel, Time |
| Semantic Model Definitions | The actual YAML/LookML/Cube files, version controlled | `metrics/upgrade_propensity.yaml` in Git |

**Input:** definitions authored by data engineers/analysts (YAML files, PRs).
**Output:** consumed by Components 3 and 4 on every single query — this is *read*, not computed, at query time.

**Possible implementation approach:** a Git repository of YAML metric/entity/rule definitions (schema-agnostic, so it isn't tied to CRM data models specifically), a CI pipeline that lints and validates definitions on every PR, and a resolver API in front for runtime lookups.

**Evaluation metrics:** definition coverage (% of dashboard/AI queries that hit a governed metric vs. an ad-hoc one), definition drift (how often two teams redefine the same term), review cycle time for new metric requests.
**Guardrails:** mandatory PR review before a metric/rule goes live, semantic versioning so old dashboards don't break silently, automated conflict detection (e.g., two rules both claiming to define "premium").

---

### Component 8 — Business Knowledge (RAG)

**Job:** Supplies *unstructured* business context — the stuff that isn't a number, like brand tone rules or compliance requirements — to the LLM and orchestration engines via retrieval-augmented generation.

| Sub-component | Job | Example |
|---|---|---|
| Document store | Holds source docs | Campaign Playbooks, Brand Guidelines, Compliance & Legal Docs |
| Embedding pipeline | Chunk + embed documents | Splits a 40-page brand guide into ~200 searchable chunks |
| Vector retriever | Finds relevant chunks for a given query | Retrieves the "age-based messaging" rule for a campaign targeting teens |

**Input:** raw documents (PDF/Docx/Confluence pages).
**Output (hypothetical):** `"Retrieved: 'Messaging to customers under 18 must not reference financing or credit offers.'"` — injected as context into Component 2's reasoning.

**Possible implementation approach:** an embedding model + vector database (e.g., Vertex AI Search or pgvector), a chunking/ingestion pipeline, and a retriever that's called before the LLM finalizes its interpretation of ambiguous requests. This is the same discipline as a citation-grounding verifier (claim → retrieval → verdict) — nothing is asserted without a retrieved source.

**Evaluation metrics:** retrieval precision/recall against a labeled test set, groundedness score (does the final answer's claim actually trace back to a retrieved chunk — checked with an LLM-as-Judge grounding verifier), staleness (age of the most-cited documents).
**Guardrails:** source allow-listing (only approved document repositories are indexed), mandatory citation — no ungrounded claims are allowed to reach the user, freshness checks that flag docs older than a review threshold.

---

### Component 9 — Data Sources

**Job:** The raw systems of record that feed everything downstream.

| Sub-component | Job | Example |
|---|---|---|
| Samsung Account | Profile, device, preference data | User's registered Galaxy devices |
| Samsung eStore | Orders, payments, returns | Purchase of Galaxy S24 six months ago |
| CRM / Campaign Data | Email/SMS/push sends & conversions | Opened last 3 promo emails |
| Web / App Analytics | Page views, searches, behavior | Viewed the Galaxy Fold 7 page 4 times this week |
| Feature Store | Precomputed scores | `clv_usd`, `upgrade_propensity_score` |
| Recommendation Models | Next-best-product/affinity outputs | "82% affinity to Fold form factor" |

**Input:** live user/transaction/behavioral events.
**Output:** raw and semi-structured feeds into the Data Platform (Component 11) and onward into Golden Customer 360 (Component 10).

**Possible implementation approach:** source APIs/CDC connectors, event streaming (Pub/Sub), batch extracts for legacy systems.
**Evaluation metrics:** data freshness SLA (e.g., events land within 15 minutes), completeness rate, schema-drift alert count.
**Guardrails:** consent-flag propagation from the very first ingestion point, PII encryption at rest and in transit, source-level access control.

---

### Component 10 — Golden Customer 360 (Trusted, Curated & Unified)

**Job:** Turns scattered per-source records into a single trusted customer profile.

| Sub-component | Job | Example |
|---|---|---|
| Unified Customer ID | Resolves identity across systems | Same person across Account + eStore + CRM gets one ID |
| Profile & Demographics | Curated attributes | Age band, region, language |
| Registered Devices | Owned/linked devices | Owns S22, no Fold |
| Purchase History | Order-level rollup | 3 purchases, last one 8 months ago |
| Engagement Signals | Cross-channel activity | Opened 6/10 recent emails |
| CLV & Propensity | Computed value + likelihood scores | `clv_usd=1450`, `upgrade_propensity=0.87` |
| Recommendation Scores | Model output per product | Fold 7 affinity 0.82 |
| Campaign History | What they've been sent, and response | 2 promos sent, 1 clicked |
| Channel Preferences | Best channel/time to reach them | Prefers push notification, evenings |
| Quality checks (bottom strip) | Dedup, standardize, enrich, consent, govern | Merges 2 duplicate accounts into 1 |

**Input:** raw feeds from Component 9.
**Output (hypothetical golden record):**
```json
{
  "customer_id": "C-10293",
  "country_code": "KR",
  "clv_usd": 1450,
  "upgrade_propensity_score": 0.93,
  "owned_devices": ["Galaxy S22"],
  "consent_marketing": true,
  "preferred_channel": "push"
}
```

**Possible implementation approach:** identity resolution (deterministic keys + probabilistic matching for edge cases), dbt/Delta Lake transformations for dedup/standardize/enrich, a consent flag that gates whether a record is even eligible to be used downstream.

**Evaluation metrics:** match precision/recall (identity resolution correctness), duplicate rate after processing, data-quality score (completeness + validity + consistency composite).
**Guardrails:** consent must be `true` before a record can be used in any targeting query, retention/deletion policy enforcement, anomaly detection on sudden score swings (e.g., propensity jumping from 0.1 to 0.9 overnight gets flagged for review, not blindly trusted).

---

### Component 11 — Data Platform

**Job:** The infrastructure backbone everything else runs on.

| Sub-component | Job | Example |
|---|---|---|
| BigQuery / Data Warehouse | Query execution | Component 6 |
| Data Lake (GCS) | Raw/staged storage | Landing zone for CDC events |
| ETL/ELT Pipelines (Dataflow) | Transform raw → curated | Builds Golden Customer 360 nightly + streaming |
| Orchestration (Airflow) | Schedules/monitors pipelines | DAG runs feature-store refresh every 4 hours |
| Data Catalog & Lineage (DataHub) | Discoverability & trust | Shows which dashboards depend on `clv_usd` |

**Input:** raw + curated data flows from Components 9 & 10.
**Output:** reliable, monitored, discoverable pipelines that all other components depend on.

**Possible implementation approach:** GCP-native — BigQuery + GCS + Dataflow + Cloud Composer (managed Airflow) + DataHub for catalog/lineage, all defined as Infrastructure-as-Code (Terraform), with drift-triggered retraining hooks for ML feature pipelines.
**Evaluation metrics:** pipeline SLA adherence (% of DAGs completing on time), lineage coverage (% of tables with documented lineage), cost per TB processed.
**Guardrails:** IaC-only change control (no manual console edits), backup/disaster-recovery policy, cost governance (budget alerts per pipeline/team).

---

## 5. How the Components Coordinate — End-to-End Walkthrough

Using one running example so you can see the handoffs:

> **Question:** *"Show premium customers in Korea likely to upgrade to Galaxy Fold."*

| Step | Component | What happens | Handoff artifact |
|---|---|---|---|
| 1 | **1 → 2** | Marketing Manager types the question into the assistant | Plain text |
| 2 | **2** | LLM extracts intent, entities, filters | Structured JSON request |
| 3 | **2 → 8** | LLM checks Business Knowledge for any relevant policy (e.g., age-targeting rules) before finalizing intent | Retrieved policy snippet |
| 4 | **2 → 3** | Structured request sent to Semantic Layer Engine | Semantic request JSON |
| 5 | **3 ↔ 7** | Engine looks up "premium," "upgrade propensity," "customer" against the governed catalog | Resolved metadata |
| 6 | **3 → 4** | Resolved metadata passed to Orchestration Engine | Resolved metadata JSON |
| 7 | **4** | Builds logical plan: which tables, which joins, cost estimate | Logical query plan |
| 8 | **4 → 5** | Plan passed to SQL Generation Engine | Logical plan JSON |
| 9 | **5** | Compiles plan into secure BigQuery SQL, injects row/column security filters | SQL string |
| 10 | **5 → 6** | SQL sent to BigQuery | SQL |
| 11 | **6** | Executes against Golden Customer 360 (built by 9→10→11) | Result rows |
| 12 | **6 → 2** | Results flow back up to the LLM | Tabular results |
| 13 | **2 → 1** | LLM turns rows into a plain-English answer + chart | "Here are 100 customers in Korea most likely to upgrade to Galaxy Fold, led by Ji-Hoon Kim (93% propensity)…" plus a table/chart |

**The key coordination principle:** each engine only trusts the *structured artifact* handed to it by the previous engine — never raw natural language, and never freeform SQL from the LLM. That's what prevents hallucinated table names or made-up metrics from ever reaching the database.

---

## 6. Problems Encountered While Working Through This (Honest Take)

Worth being upfront: none of this is a solved problem yet. Working through the design surfaced real, unresolved friction points — flagging them here on purpose, because a credible architecture proposal should name its hard parts, not paper over them.

| Area | Problem | Where it currently stands |
|---|---|---|
| Natural language understanding (2) | People phrase the same question a dozen different ways; entity/metric extraction is reliable on clear questions but degrades fast on ambiguous or compound ones | Confidence-threshold + clarification-question fallback helps, but hasn't been stress-tested against messy real-world phrasing at scale |
| Semantic governance (7) | Two teams will always want to define "premium" or "active customer" slightly differently — the catalog doesn't resolve organizational disagreement, only makes it visible | Needs a real review/ownership process (not just tooling) before it can be trusted; this is more a people problem than a technical one |
| Cold start | The semantic layer is only as useful as the definitions loaded into it — an empty or half-populated catalog provides little value on day one | Suggests starting narrow (10–20 high-value metrics) rather than trying to model everything up front |
| Query cost & performance (4, 5, 6) | Even with a planning/optimization step, a badly-shaped question can still generate an expensive multi-join query | Cost caps catch the worst cases, but "expensive but technically valid" queries still slip through — needs more real query patterns to tune against |
| Identity resolution (10) | Merging customer identities across systems is genuinely hard — sparse or conflicting data leads to false merges or missed matches | Human-review queue for low-confidence matches is a mitigation, not a fix; match quality depends heavily on source data quality, which varies |
| Retrieval grounding (8) | Retrieval isn't always precise — a plausible-looking but wrong chunk can get pulled in, and the LLM may lean on it with unwarranted confidence | Mandatory-citation guardrail helps catch this at review time, but doesn't prevent it at generation time |
| Latency stacking | Five sequential engines (NLU → resolution → orchestration → SQL → warehouse) each add latency; it adds up | Caching resolved metadata and common query plans is the likely answer, but hasn't been designed in detail yet |
| Evaluation itself | "Is this answer correct?" is subjective for open-ended natural-language questions — automated metrics only go so far | Needs a human-in-the-loop review sample alongside the automated scores in Section 7, not either one alone |
| Generalizing beyond CRM (Section 9) | The entity/metric schema needs to flex across very different domains without becoming so abstract it's useless | Still an open design question — the CRM version above is the first concrete test case, not proof it generalizes cleanly |

None of these are reasons to stop — they're the reason to treat this as a phased, learn-as-we-go rollout (Section 10) instead of a big-bang launch. The direction feels right; the details above are exactly what a pilot phase should be built to surface and fix early.

## 7. Cross-Cutting Evaluation Framework

| Layer | Primary metric | Target |
|---|---|---|
| NLU (Component 2) | Entity/intent extraction F1 | ≥ 0.90 |
| Semantic resolution (3) | Concept resolution accuracy | ≥ 0.98 |
| Query planning (4) | Cost-estimate vs. actual variance | ≤ 15% |
| SQL generation (5) | SQL validity + execution success rate | ≥ 0.99 |
| Warehouse (6) | p95 query latency | ≤ 5s |
| Semantic Layer (7) | % queries hitting a governed metric | ≥ 95% |
| RAG (8) | Groundedness (claims traced to source) | 100% (zero ungrounded claims) |
| Golden 360 (10) | Identity-match precision | ≥ 0.97 |
| Platform (11) | Pipeline SLA adherence | ≥ 99% |

A monthly **LLM-as-Judge audit** (scoring a sample of end-to-end conversations on correctness, groundedness, and security-compliance) should sit on top of these component-level metrics — a multi-dimension quality-scoring pattern that generalizes well beyond this one use case.

---

## 8. Guardrails Summary (Governance)

These are the guardrails the design points to *so far*, mapped directly to the problems in Section 6 — the list should be expected to grow as real usage surfaces edge cases none of us have thought of yet:

- **Never let the LLM write or execute SQL directly** — it only ever produces structured intent.
- **Deny-by-default** on every permission check (Components 3, 5, 6).
- **Read-only SQL allowlist** — no DML/DDL can ever be generated.
- **Consent-gating** — no customer record is usable downstream unless `consent_marketing = true`.
- **Mandatory grounding** — RAG-sourced claims must cite a retrieved document; no ungrounded assertions reach the end user.
- **Version control everywhere** — every metric, rule, and entity definition is Git-managed and PR-reviewed.
- **Cost/row caps** — hard limits on estimated bytes scanned and result-set size before execution.

---

## 9. Generalizing This Pattern Beyond CRM

The CRM scenario above is one instantiation of a pattern that isn't inherently CRM-specific. Here's how each piece generalizes to a domain-agnostic semantic layer:

| CRM-specific version (this doc) | General-purpose equivalent |
|---|---|
| Component 7 entities: Customer, Product, Campaign, Order, Device | Any domain's entities: Employee, Shipment, Invoice, Asset, Ticket — the entity catalog is just config, not hardcoded logic |
| Component 7 metrics: CLV, Upgrade Propensity, Email CTR | Any KPI in any domain: On-Time Delivery Rate, Inventory Turnover, MRR, Defect Rate — same YAML-metric pattern, different formulas |
| Component 9 sources: Samsung Account, eStore, CRM data | Any source system: ERP, HRIS, IoT telemetry, finance ledgers — the ingestion pattern (connector → raw layer) doesn't change |
| Component 10 "Golden Customer 360" | A general "Golden Record" pattern — could just as easily be a Golden Product 360, Golden Vendor 360, or Golden Asset 360 using the identical dedup/standardize/enrich/consent pipeline |
| Components 2–5 (NLU → Resolution → Orchestration → SQL) | Fully domain-agnostic already — none of these engines contain CRM logic; they only operate on whatever entities/metrics/rules Component 7 defines |
| Component 8 (RAG / Business Knowledge) | Same retrieval pattern works for any policy corpus — compliance docs, SOPs, engineering runbooks, not just brand/campaign guidelines |

**The core insight worth contributing:** the amount of CRM-specific code in this architecture is small — it's really only the *content* of the YAML definitions (Component 7) and the specific source connectors (Component 9). The engines (2–6) and the orchestration logic are reusable infrastructure. That's the argument for building this once as a general-purpose semantic layer, with CRM as the first of several domain packs plugged into it, rather than as CRM-only infrastructure.

## 10. Suggested Next Steps

| Phase | Scope | Duration |
|---|---|---|
| 1 | Stand up Semantic Layer (Component 7) for top 10 metrics/entities | 4–6 weeks |
| 2 | Build Semantic Layer Engine + Orchestration Engine (3, 4) | 4–6 weeks |
| 3 | SQL Generation Engine + security filter injection (5) | 3–4 weeks |
| 4 | Golden Customer 360 pipeline (9, 10, 11 hardening) | 6–8 weeks |
| 5 | LLM Assistant integration + RAG knowledge layer (2, 8) | 4–6 weeks |
| 6 | Pilot with one BI dashboard + one CRM use case, then scale | Ongoing |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Metric definitions drift out of sync with business reality | Scheduled quarterly review + owner sign-off per metric |
| LLM misinterprets ambiguous questions | Confidence threshold triggers a clarifying question instead of a guess |
| Cost overruns from expensive queries | Hard cost caps in Orchestration Engine + BigQuery budget alerts |
| Identity-resolution errors merge two different customers | Human-review queue for low-confidence matches |
| RAG retrieves outdated policy docs | Freshness checks + document expiry flags |


---

## 12. Where This Stands

This isn't a finished or fully validated architecture — several of the harder problems in Section 6 are still open, and the CRM example is one worked-through scenario, not a proven-at-scale system. What it does offer is a coherent, end-to-end direction: a clear separation between *what a question means* (semantic layer) and *how to answer it safely* (the engines), with guardrails and evaluation criteria attached to every piece rather than bolted on afterward. The intent in sharing it is to compare notes against the direction of the broader semantic layer effort already underway, and to fold in whatever's already been learned there.
