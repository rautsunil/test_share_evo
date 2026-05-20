# AMACE — Patent Abstract for Prior-Art Search

**For:** Patent review team — ProArt / Google Patents / Espacenet search
**Status:** Draft abstract for Zone-1 novelty screening
**Confidential — April 2026**

---

## 1. Title (Working)

**System and Method for Autonomous Multi-Stratum Anomaly Correlation and Counterfactually-Validated Causal Insight Generation in Distributed Computing Environments**

Short form: **AMACE — Autonomous Multi-stratum Anomaly Correlation Engine**

---

## 2. Abstract (Patent-Style)

A computer-implemented system and method is disclosed for autonomously detecting, correlating, and causally validating anomalies that occur concurrently across multiple, heterogeneous abstraction layers ("strata") of a distributed computing environment — specifically a raw-event stratum (S₀), a derived-feature stratum (S₁), and a model-prediction stratum (S₂). The system ingests data from arbitrary monitoring sources without manual schema configuration, classifying each field into an entity, timestamp, dimension, or metric role using an entropy- and cardinality-based statistical classifier. Each stratum is monitored by a stratum-adapted detector — robust median-absolute-deviation scoring for raw events, population-stability-index distribution-shift scoring for derived features, and prediction-interval scoring for model outputs — whose sensitivity thresholds are continuously self-calibrated by injecting synthetic anomalies of known magnitude. Every detection is emitted as a stratum-agnostic, seven-component data structure ("Deviation Signature Tensor") encoding the stratum origin, the resolved entity set, a temporal envelope, a dimensional locus, a magnitude vector, a bootstrapped confidence interval, and a cryptographic provenance hash. A dimensional sub-group localizer based on information-theoretic beam search identifies the precise affected cohort. A four-axis correlation engine then scores every pair of cross-stratum tensors on entity coincidence, temporal precedence, dimensional coherence, and learned semantic affinity, producing a unified correlation tensor. Candidate correlations are submitted to a computational ablation protocol that physically removes the candidate cause's entities from the dataset and re-evaluates the candidate effect, applying exclusivity, sufficiency, and minimality tests with bootstrapped confidence intervals to establish causation rather than mere correlation. Validated causal edges are assembled into a directed acyclic insight graph with root-cause identification, predictive precursor pattern matching for early warning, anti-correlation detection of absent expected responses, and a constrained retrieval-augmented language-model layer that generates mechanistic explanations bounded by the statistically proven causal structure.

---

## 3. The Big Problem It Solves (Plain Language)

**The problem no existing tool solves:**

Modern distributed systems generate signals at three fundamentally different levels of abstraction:

1. **Raw operational events** — server requests per second, cache hits, error rates, latency.
2. **Derived features** — aggregated, transformed signals such as cache-hit ratio distributions, user-engagement scores, feature-store values.
3. **Model predictions** — outputs of machine-learning models such as churn scores, recommendation rankings, fraud probabilities.

When something breaks, the symptoms appear across **all three levels simultaneously** — a configuration change at the raw-event layer cascades into a feature-distribution shift, which then drifts a downstream ML model's predictions, which finally damages business metrics.

**Today, no monitoring system can mechanically connect anomalies across these three layers.** Existing tools (DataDog, Splunk, Moogsoft, BigPanda, Evidently AI, MLflow, ServiceNow ITOM, PagerDuty AIOps, New Relic, Dynatrace) each operate inside a single stratum, emit alerts in incompatible proprietary formats, and rely on a human engineer to manually correlate dashboards across tools. Even within a single stratum, these tools report *correlations* — they cannot computationally distinguish correlation from coincidence or from genuine causation.

The result is: incidents take hours to diagnose, root causes are guessed at rather than proven, downstream model degradation is discovered only after business impact, and the same cascade pattern is rediscovered manually each time it recurs.

**AMACE addresses this by providing the first end-to-end pipeline that (a) unifies anomaly representation across heterogeneous strata, (b) correlates anomalies across strata using a four-axis tensor, and (c) validates correlations as causal using physical data ablation — without requiring any pre-defined causal graph, manual schema configuration, or human-authored correlation rules.**

---

## 4. What It Does (One-Paragraph Lay Description)

AMACE is an autonomous diagnostic engine that watches a distributed system from three altitudes at once — the raw infrastructure, the derived analytical features, and the machine-learning models — using a different statistical detector best suited to each altitude. Whenever any detector fires, AMACE converts the alert into a single common format so that alerts from the three altitudes can be mathematically compared. It then asks four questions of every pair of cross-altitude alerts: *Do they affect the same users? Did one start before the other in a way consistent with the system's learned propagation delays? Are they concentrated in the same sub-population? Do their underlying metrics historically move together?* If the combined answer is strong enough, AMACE then performs a counterfactual experiment — it removes the suspected cause's data from the dataset, re-runs the effect's anomaly calculation, and measures how much of the effect disappears. If a significant fraction disappears, causation is considered proven. The validated cause-effect edges are assembled into a graph, the root is identified, the affected user cohort is intersected across the entire chain, similar past patterns are matched to predict where the cascade is heading, and a constrained language-model layer attaches a mechanistic explanation that is forbidden from contradicting any statistically proven fact.

---

## 5. Searchable Novelty Hooks (For ProArt / Google Patents Keyword Search)

The patent team should search the following discrete inventive concepts. Each is independently potentially patentable and each defines a distinct prior-art search query.

| # | Novelty Hook | Primary Search Phrases | Closest Known Prior Art (Distinguish Against) |
|---|---|---|---|
| **N1** | Zero-configuration schema inference for **correlation operability** (not for cataloging) | "entropy based field classification", "automatic schema inference anomaly correlation", "cardinality ratio dimension metric classifier" | Apache Atlas, AWS Glue, Alation, DataHub, Confluent Schema Registry — all do cataloging, not correlation. |
| **N2** | **Three-stratum** anomaly detector with stratum-adapted methods (MAD / PSI / prediction intervals) under one system | "multi-stratum anomaly detection", "raw event feature model anomaly detection unified", "stratum-adaptive threshold calibration" | DataDog, CloudWatch, Evidently AI, MLflow, Monte Carlo — each single-layer only. |
| **N3** | **Deviation Signature Tensor (DST)** — universal 7-component anomaly representation (stratum, entity set, temporal envelope, dimensional locus, magnitude vector, confidence interval, provenance hash) | "universal anomaly representation", "anomaly tensor entity temporal dimensional magnitude", "cross-source anomaly schema" | OpenTelemetry, CEF, STIX, SNMP traps, syslog — all standardize telemetry/events, not deviations. |
| **N4** | **Self-calibrating detection thresholds via synthetic anomaly injection** (calibrating cross-stratum correlation, not single-metric accuracy) | "synthetic anomaly injection threshold calibration", "false positive rate self calibration anomaly detector", "synthetic deviation cross-stratum calibration" | Netflix Chaos Monkey, Gremlin — inject failures, not calibrated statistical anomalies. |
| **N5** | **Entropic Relevance Decomposition (ERD)** — MDL-inspired beam search for dimensional anomaly localization (Surprise − Cost in bits) | "information theoretic dimensional drill down", "MDL beam search anomaly subgroup", "surprise minus cost anomaly localization" | OLAP (Codd), PRIM, SD-Map, decision trees (ID3, C4.5), DataDog Watchdog — different objectives or human-directed. |
| **N6** | **Four-axis Cross-Stratum Causal Similarity Tensor (CSCT)** — Jaccard-with-surprise + Gaussian temporal kernel on learned propagation delay + hierarchical dimensional matching + cosine semantic affinity | "four axis cross stratum correlation", "surprise adjusted Jaccard anomaly", "learned propagation delay Gaussian kernel anomaly correlation" | Moogsoft, BigPanda, Splunk SPL correlations, Granger causality — at most 2 axes, same-layer only. |
| **N7** | **Computational Ablation Protocol (CAP)** — three-test counterfactual validation by **physical data removal and recomputation** with bootstrapped CIs (exclusivity, sufficiency, minimality) | "counterfactual ablation anomaly validation", "data removal recomputation causal proof", "exclusivity sufficiency minimality anomaly causal test" | Granger causality, Pearl do-calculus, A/B testing, SHAP/LIME — none physically ablate observational anomaly data. |
| **N8** | **Causal Insight Graph Synthesis (CIGS)** — automated DAG construction from validated cross-stratum edges with root-cause identification and entity-set intersection across the chain | "automated causal graph construction cross stratum", "validated edge DAG root cause anomaly", "entity intersection causal chain" | FTA/ETA (manual), CMDB dependency mapping, Microsoft Gandalf (single stratum), CloudWatch ServiceLens. |
| **N9** | **Cross-Stratum Predictive Precursor Pattern matching** with temporal-decay-weighted completion probability | "precursor sequence library cross stratum", "predictive cascade pattern matching anomaly", "incident pattern completion probability decay" | CEP (Esper, Flink CEP), GSP/PrefixSpan sequence mining, Hawkes processes — single-layer event patterns. |
| **N10** | **Anti-correlation detection** — flagging the **absence** of expected cross-stratum correlations (e.g., autoscaler that did not trigger) | "anti correlation missing expected response", "absent correlation anomaly detection", "expected companion anomaly missing" | Heartbeat / dead-man's-switch monitoring — detect missing events, not missing relationships. |
| **N11** | **Constrained LLM hypothesis generation with statistical veto** — four-layer hallucination containment (retrieval-only, citation embedding similarity, statistical direction/temporal/entity consistency, trust-discount factor ω < 1.0); LLM-bridged correlations are themselves subjected to CAP ablation | "constrained LLM root cause hallucination containment", "retrieval augmented causal hypothesis statistical constraint", "LLM trust discount factor causal validation", "LLM hypothesis ablation" | Lewis et al. 2020 (RAG), Microsoft RCACopilot, PagerDuty Copilot — none constrain LLM output by statistical causal proof or subject LLM bridges to ablation. |
| **N12** | **Observable API output with provenance hash chain** — externally inspectable output schema containing four-axis scores, ablation results, stratum-labeled causal chain, and cryptographic provenance | "anomaly correlation API provenance hash", "ablation result API schema causal", "four axis correlation API output" | REST/JSON, CEF, STIX, SOX/HIPAA audit trails — none expose multi-axis cross-stratum causal evidence. |

---

## 6. Recommended ProArt / Google Patents Search Strategy

**Primary CPC / IPC classes to scope:**

- `G06F 11/07` — Error detection / failure diagnosis
- `G06F 11/30` — Monitoring
- `G06F 11/34` — Recording or statistical evaluation of computer activity
- `G06N 5/04` — Inference / reasoning
- `G06N 7/00` — Probabilistic / statistical computing
- `H04L 41/06` — Network fault management
- `H04L 43/00` — Network monitoring

**High-priority composite queries** (run each independently for each novelty hook N1–N12):

1. `("multi-stratum" OR "cross-layer" OR "cross-stratum") AND (anomaly OR deviation) AND (correlation OR causation)` — core hit, scopes N3, N6.
2. `("counterfactual" OR "ablation") AND (anomaly OR alert) AND (validation OR proof) AND (entity OR cohort)` — scopes N7.
3. `(MDL OR "minimum description length" OR "information theoretic") AND ("beam search" OR "drill-down") AND anomaly` — scopes N5.
4. `("population stability index" OR PSI) AND ("median absolute deviation" OR MAD) AND ("anomaly detection" OR monitoring)` — scopes N2.
5. `(LLM OR "language model") AND (anomaly OR "root cause") AND (constraint OR "hallucination") AND (statistical OR causal)` — scopes N11.
6. `("synthetic anomaly" OR "synthetic deviation") AND (threshold OR calibration) AND injection` — scopes N4.
7. `("missing correlation" OR "absent response" OR "anti-correlation") AND monitoring` — scopes N10.
8. `("precursor pattern" OR "incident precursor" OR "cascade pattern") AND (predict OR forecast) AND (sequence OR library)` — scopes N9.

**Decision rule for moving to Zone 2 (drafting):**

If **no single document** in ProArt or Google Patents discloses the combination of: (a) cross-stratum DST representation + (b) four-axis CSCT scoring + (c) computational ablation validation — then the **system-level claim (N3 + N6 + N7 in combination)** is novel and should proceed to Zone 2. These three together form the core invention; the remaining nine novelties are supporting independent and dependent claim opportunities.

---

## 7. One-Line Pitch (For The Search Report Cover Note)

> AMACE is a monitoring and diagnostic system that, for the first time, mechanically detects, mathematically correlates, and counterfactually proves causal links between anomalies occurring concurrently in raw infrastructure events, derived analytical features, and machine-learning model predictions — replacing the human engineer who today does this manually across incompatible dashboards.

---

*Prepared for patent prior-art screening. Confidential.*
