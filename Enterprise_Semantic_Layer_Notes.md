# Enterprise Semantic Layer for AI-Powered CRM

## High-Level Architecture

``` text
Marketing User
    |
LLM / AI Assistant
    |
AI Request Processor
(Intent Parser, Entity Extractor, Business Concept Extractor)
    |
Semantic Resolution Engine
(Concept Resolver, Relationship Resolver, Rule Executor)
    |
Query Planning Engine
(Query Planner, SQL Generator, Query Validator)
    |
BigQuery
    |
Business Results
    |
Campaign Recommendation
```

## End-to-End Workflow

1.  User asks a business question.
2.  LLM understands the intent.
3.  AI Request Processor extracts intent, entities, and business
    concepts.
4.  Semantic Resolution Engine resolves business meaning.
5.  Query Planner creates a logical execution plan.
6.  SQL Generator creates SQL.
7.  BigQuery executes SQL.
8.  Results are returned for analytics or campaign generation.

## Semantic Layer

### Business Entities

Customer, Product, Campaign, Order, Device.

Built from: - Database schema - Domain model - SME workshops - ER
diagrams

Evaluation: - Entity Coverage - Entity Resolution Accuracy - Missing
Entity Rate

### Business Metrics

CLV, Revenue, Recommendation Score, Upgrade Propensity, Email CTR.

Evaluation: - Metric Accuracy - Metric Consistency - Duplicate Metric
Rate

### Business Rules

Examples: - Premium Customer = CLV \> 1000 - Upgrade Candidate = Upgrade
Propensity \> 0.75 - Campaign Eligible = No campaign in last 30 days

Evaluation: - Rule Accuracy - Rule Coverage - Rule Conflict Rate

### Relationships

Customer→Orders Customer→Campaign Customer→Recommendation

Evaluation: - Join Accuracy - Relationship Coverage

### Dimensions

Country, Age Group, Product Category, Device Family.

Evaluation: - Dimension Coverage - Hierarchy Accuracy

### Semantic Model Definition

Contains entities, metrics, rules, relationships and dimensions. Can be
represented in YAML, LookML, JSON or metadata tables. A YAML file is
only the model definition, not the semantic layer itself.

## Golden Customer 360

Sources: - Samsung Account - Samsung eStore - CRM - Web Analytics -
Feature Store - Recommendation Models

Customer360 contains: - Customer ID - Country - CLV - Recommendation
Score - Upgrade Propensity - Campaign History - Email CTR

### Evaluation Metrics

-   Completeness
-   Accuracy
-   Consistency
-   Uniqueness
-   Freshness
-   Validity
-   Referential Integrity
-   Identity Resolution Accuracy
-   Feature Availability
-   Business Readiness

## Semantic Resolution Engine

-   Concept Resolver
-   Relationship Resolver
-   Rule Executor

## Query Planning Engine

-   Query Planner
-   SQL Generator
-   Query Validator

## Response Evaluation

-   Intent Accuracy
-   Entity Accuracy
-   Concept Accuracy
-   Relationship Accuracy
-   Rule Accuracy
-   SQL Validation
-   Latency
-   Confidence Score

## Key Takeaways

-   Semantic Modeling defines business meaning.
-   Semantic Layer serves those definitions.
-   LLM understands language.
-   Semantic Engine resolves business meaning.
-   Query Planner builds logical plans.
-   SQL Generator generates SQL.
-   Golden Customer360 provides trusted data.
