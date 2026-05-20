# AMACE — Prior-Art Search Query Terms for Patent Officer

**Purpose:** Strategic search queries for the patent officer's proprietary prior-art search tool, with novelty positioning notes.
**Companion to:** AMACE Patent Abstract for Prior-Art Search (already shared)
**Confidential — April 2026**

---

## How to Use This Document

For each query below, the officer should:

1. **Run the query** in the proprietary prior-art tool, ProArt, Google Patents, Espacenet, and IEEE Xplore.
2. **Read the "Novelty Positioning" line** before reviewing hits — this is the framing that lets the officer recognize whether a hit is truly blocking or only adjacent.
3. **Flag a hit as blocking only if it discloses the full combination** described in the novelty line. Adjacent or partial hits actually strengthen the patent (they prove the field is active but has not solved the specific problem).

The queries are ordered from **broadest (system-level)** to **most specific (component-level)**, so the officer can prune the search tree efficiently.

---

## SYSTEM-LEVEL QUERIES (Cover the Overall Invention)

### Query 1 — The Core System Claim

**Search string:**
> `("cross-stratum" OR "cross-layer" OR "multi-layer" OR "multi-stratum") AND (anomaly OR deviation) AND (correlation OR causal) AND (validation OR ablation OR counterfactual)`

**Novelty positioning:**
Our system is the first to unify anomaly detection, correlation, and causal validation across **three distinct abstraction layers** of a software system (raw events, derived features, ML-model predictions) in a single autonomous pipeline — every prior-art system operates within only one layer.

---

### Query 2 — Cross-Layer Monitoring Pipeline

**Search string:**
> `("observability" OR "monitoring") AND ("infrastructure" AND "machine learning" AND "feature") AND (correlation OR "root cause") AND autonomous`

**Novelty positioning:**
Our system is the first end-to-end autonomous pipeline that connects an infrastructure anomaly, a feature-store distribution shift, and an ML-model prediction drift as a single causal chain — existing observability tools treat these as three separate, human-bridged disciplines.

---

### Query 3 — Universal Anomaly Representation

**Search string:**
> `("anomaly" OR "alert" OR "deviation") AND ("universal" OR "unified" OR "schema-agnostic" OR "common format") AND (representation OR tensor OR structure) AND (entity AND temporal AND dimensional)`

**Novelty positioning:**
Our Deviation Signature Tensor is the first data structure that represents anomalies from **any monitoring source and any abstraction layer** in a single mathematically-comparable seven-component format (stratum, entity set, temporal envelope, dimensional locus, magnitude, confidence interval, provenance hash) — prior art standardizes raw telemetry (OpenTelemetry, CEF, STIX) but not detected deviations.

---

### Query 4 — Counterfactual Causal Validation for Anomalies

**Search string:**
> `(counterfactual OR ablation OR "data removal") AND (anomaly OR alert OR incident) AND (causation OR causal) AND (sufficiency OR exclusivity OR minimality)`

**Novelty positioning:**
Our system is the first to validate causal relationships between anomalies by **physically removing data and recomputing** the dependent anomaly, applying a three-test protocol (exclusivity, sufficiency, minimality) — prior art either assumes a known causal graph (Pearl's do-calculus), requires intervention (A/B testing), or works only on continuous time-series (Granger causality), none of which apply to discrete cross-stratum anomaly events.

---

## COMPONENT-LEVEL QUERIES (Cover Individual Patentable Claims)

### Query 5 — Information-Theoretic Dimensional Localization

**Search string:**
> `("beam search" OR "best-first search") AND ("minimum description length" OR MDL OR "information gain" OR entropy) AND (anomaly OR outlier OR "subgroup discovery") AND (dimension OR slice OR cohort)`

**Novelty positioning:**
Our Entropic Relevance Decomposition is the first to use an MDL-inspired *Surprise minus Cost* scoring (in bits) inside a beam search specifically for **anomaly sub-group localization** — prior subgroup-discovery work (PRIM, SD-Map, SUBGROUP) optimizes for rule coverage, and decision-tree information gain optimizes for prediction, neither for anomaly concentration.

---

### Query 6 — Four-Axis Cross-Layer Correlation Scoring

**Search string:**
> `(Jaccard OR overlap) AND ("temporal kernel" OR "Gaussian kernel" OR "propagation delay") AND ("semantic similarity" OR "cosine similarity" OR embedding) AND (correlation OR matching) AND alert`

**Novelty positioning:**
Our Cross-Stratum Causal Similarity Tensor is the first to combine **four orthogonal axes** (surprise-adjusted entity Jaccard + learned-propagation-delay Gaussian temporal kernel + hierarchical dimensional matching + cosine semantic embedding affinity) into a single correlation score across heterogeneous strata — existing AIOps tools (Moogsoft, BigPanda, PagerDuty) use at most two axes (time proximity + tag similarity) within a single layer.

---

### Query 7 — Schema Inference for Anomaly Correlation (Not Cataloging)

**Search string:**
> `("schema inference" OR "automatic schema" OR "field classification") AND (entropy OR cardinality OR "Shannon") AND (entity OR dimension OR metric OR timestamp) AND (monitoring OR anomaly OR correlation)`

**Novelty positioning:**
Our schema inference engine is the first to classify fields into **correlation-operable roles** (entity / timestamp / dimension / metric) for the purpose of enabling automated cross-source anomaly correlation — existing schema-discovery tools (Apache Atlas, AWS Glue, Alation, DataHub) classify fields for cataloging, search, and governance, never producing the cross-source entity-identity map that anomaly correlation requires.

---

### Query 8 — Self-Calibrating Detection via Synthetic Anomalies

**Search string:**
> `("synthetic anomaly" OR "synthetic deviation" OR "controlled fault injection") AND (threshold OR sensitivity OR "false positive rate") AND (calibration OR adjustment OR tuning) AND (detector OR monitoring)`

**Novelty positioning:**
Our system is the first to inject **statistical anomalies of known magnitude and known causal relationship across strata** to continuously calibrate cross-stratum correlation sensitivity — chaos-engineering tools (Netflix Chaos Monkey, Gremlin) inject failures or load, never calibrated statistical deviations, and never to tune a multi-layer correlation engine.

---

### Query 9 — Distribution Shift Detection in Feature Pipelines

**Search string:**
> `("population stability index" OR PSI OR "Kolmogorov-Smirnov" OR Wasserstein) AND ("distribution drift" OR "distribution shift") AND ("feature store" OR "feature pipeline" OR "derived feature") AND (anomaly OR detection)`

**Novelty positioning:**
Our system is the first to apply distribution-shift detection (PSI) **as one engine within a three-stratum unified pipeline** that emits the same universal DST output as the raw-event detector (MAD) and the model-prediction detector (prediction intervals) — Evidently AI and NannyML use PSI in isolation for ML monitoring only, never feeding a cross-stratum correlation engine.

---

### Query 10 — Robust Outlier Detection in Streaming Systems

**Search string:**
> `("median absolute deviation" OR MAD OR "modified z-score" OR "robust z-score") AND ("sliding window" OR streaming OR "rolling baseline") AND (anomaly OR outlier) AND ("auto threshold" OR "self-calibrating" OR adaptive)`

**Novelty positioning:**
Our system is the first to combine MAD-based robust scoring with **self-calibrated thresholds for cross-stratum correlation accuracy** (rather than for single-metric accuracy) — MAD-based detection exists in academic literature and tools like Twitter AnomalyDetection, but never with thresholds tuned by injecting cross-layer synthetic scenarios into the full correlation pipeline.

---

### Query 11 — Predictive Pattern Matching from Past Incidents

**Search string:**
> `("precursor pattern" OR "incident pattern" OR "cascade prediction") AND ("sequence library" OR "pattern library" OR "incident database") AND (predict OR forecast OR "early warning") AND (anomaly OR alert OR incident)`

**Novelty positioning:**
Our Predictive Precursor Engine is the first to detect partial matches against **cross-stratum causal-graph patterns** (sequences of validated DSTs spanning S₀, S₁, S₂) with temporal-decay-weighted completion probability — Complex Event Processing tools (Esper, Flink CEP) detect raw-event sequences in a single layer, never validated causal-chain patterns spanning multiple abstraction strata.

---

### Query 12 — Missing/Absent Expected Behavior Detection

**Search string:**
> `("missing correlation" OR "absent response" OR "expected behavior" OR "anti-correlation") AND (monitoring OR detection) AND (anomaly OR alert OR incident)`

**Novelty positioning:**
Our anti-correlation detector is the first to alert on the **absence of expected cross-stratum relationships** (e.g., a latency spike that did not trigger the autoscaler that should have responded) — heartbeat monitoring and dead-man's-switch detect missing events, never missing learned relationships between events across strata.

---

### Query 13 — LLM Reasoning Constrained by Statistical Evidence

**Search string:**
> `("language model" OR LLM OR "large language model") AND ("root cause" OR "causal" OR diagnosis) AND (constraint OR "hallucination" OR grounding) AND (retrieval OR RAG OR "retrieval-augmented")`

**Novelty positioning:**
Our system is the first to constrain LLM-generated mechanistic hypotheses inside a **mathematically-bounded statistical framework** (trust discount ω < 1.0, four-layer hallucination containment: retrieval-only generation, citation embedding-similarity verification, statistical direction/temporal/entity consistency checks, and re-submission of LLM-bridged correlations to the same physical ablation protocol applied to statistical correlations) — Microsoft RCACopilot and PagerDuty Copilot use LLMs for root cause analysis but never subject the LLM's output to a statistical veto or to counterfactual ablation.

---

### Query 14 — Causal Graph Construction from Observational Anomaly Data

**Search string:**
> `("causal graph" OR DAG OR "directed acyclic graph") AND (construction OR assembly OR inference) AND (anomaly OR alert OR incident) AND ("root cause" OR "in-degree")`

**Novelty positioning:**
Our Causal Insight Graph Synthesis is the first to assemble a directed acyclic graph from **statistically validated cross-stratum causal edges** (each individually proven via ablation) with automatic root identification via in-degree analysis and **entity-set intersection across the entire chain** to identify users affected by every step — Microsoft Gandalf and CMDB-based dependency maps build graphs from infrastructure topology, not from validated anomaly causation, and operate within a single stratum.

---

### Query 15 — Observable API Schema with Causal Provenance

**Search string:**
> `(API OR endpoint OR "REST") AND (schema OR output OR response) AND ("causal" OR "root cause" OR ablation) AND (provenance OR "hash chain" OR audit)`

**Novelty positioning:**
Our system exposes an externally observable API output containing four-axis correlation scores, ablation test results, stratum-labeled causal chains, and a SHA-256 provenance hash chain — making infringement **detectable from outside the system** without source-code access, which no other monitoring-tool API enables because no other system performs these computations.

---

## STRATEGIC NOTES FOR THE PATENT OFFICER

**On framing the search:**

The novelty of AMACE does **not** rest on any single mathematical technique. MAD, PSI, Jaccard, beam search, MDL, RAG, cosine similarity, and bootstrapping are all individually well-known in their respective fields. The patent novelty rests entirely on the **specific combination and the specific application** — applying these techniques **across three abstraction strata, in a single pipeline, with counterfactual ablation as the validation step.**

Therefore: a prior-art hit that discloses MAD-based anomaly detection alone, or PSI-based drift detection alone, or RAG-grounded LLM reasoning alone, is **not blocking**. It is only blocking if it discloses the *combination with cross-stratum correlation and ablation-based causal validation*.

**The three queries most likely to surface true blocking art (in order):**

1. **Query 4** (counterfactual ablation for anomaly causation) — this is our strongest individual novelty (90–95% grant probability). If anyone has published this, our central claim weakens.
2. **Query 6** (four-axis correlation) — also 90–95% grant probability. Four orthogonal axes in a single tensor across strata is unprecedented.
3. **Query 1** (the core system claim) — the broadest umbrella query.

If Queries 4 and 6 return no blocking art, the patent has a strong foundation regardless of what other queries return, because the **system-level claim (DST + four-axis CSCT + computational ablation in combination)** stands on Queries 3, 4, and 6 together.

**The "categorical gap" argument:**

When summarizing findings, the officer can emphasize that across 15 capability dimensions (per the Comparison Table in our novelty map), the closest competitor (Evidently AI) provides 2 of 15 — and both of those are single-stratum only. The gap between AMACE and the prior art is not incremental; it is categorical. This framing is important when the search returns adjacent-but-not-blocking art, because it reframes that art as **supporting** evidence that the field is active but has not solved the problem AMACE solves.

---

*Prepared for patent officer review. Confidential.*
