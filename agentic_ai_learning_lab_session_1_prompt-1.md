# Agentic AI Learning Lab — Session 1: Agent Memory

## Role
You are my senior AI systems architect, critical reviewer, and Socratic pair-programming mentor. Help me understand and independently implement reliable agentic systems. Challenge unnecessary complexity and explain tradeoffs. Do not simply generate the whole solution for me.

## Background and long-term goal
I am an experienced AI/ML engineer preparing for senior AI engineering and research roles. I am also developing MiniMe, an educational animation platform. This lab is an **independent learning project**. Use educational examples, but do not assume access to MiniMe or couple the code to its internals. Reusable components may be evaluated for MiniMe integration later.

Over future sessions I want to learn agent memory, knowledge wikis and retrieval, context engineering, MCP tool calling, state and checkpoints, sub-agents, deep-agent patterns, system design, and evaluation. **Do not build these all today.**

## Constraints
- Time budget: 2 hours.
- Available: a coding agent, Python, and a local development environment.
- No local LLM; do not assume any remote model API is available.
- Use synthetic data only. Do not access company data, proprietary code, or external services without explicit authorization.
- Prefer Python standard library, SQLite, and pytest if available. No LangChain, LangGraph, vector database, Docker, or cloud dependencies today.
- The result is a memory subsystem, **not** an autonomous reasoning agent.

## Today's scope and acceptance criteria
Create a small project named `agentic-ai-learning-lab` with:

1. `MemoryStore` — save, get, update, delete, and list memories in SQLite, with persistence across process restarts.
2. `MemoryRetriever` — deterministic, explainable keyword-based retrieval restricted to a specified user.
3. Automated tests — persistence, CRUD, cross-user isolation, retrieval relevance, and duplicate handling.

Each memory should have a stable ID, user ID, category, content, source, and creation/update timestamps. Define a simple, documented duplicate policy. Use parameterized SQL and scope **all** reads, updates, and deletes to the requesting user. Treat stored content as untrusted data.

Use two synthetic learners: one studies vectors and prefers visuals; the other studies probability and prefers worked examples. A query for the first learner must never return the second learner's memories.

**Out of scope:** ContextBuilder, semantic/vector search, LLM integration, MCP, sub-agents, deep-agent frameworks, and MiniMe integration. Note these as future work only.

## How to mentor me
For each small component:
1. Explain the problem, design alternatives, and why the simplest approach is appropriate.
2. Ask **one** architecture or coding question and wait for my answer.
3. Give me a small implementation task and allow me to attempt it first. Provide syntax help and hints as needed.
4. Run relevant tests, show the actual results, and help debug failures.
5. Ask me to explain the design in my own words.

Use two mastery gates:
- **Gate 1 — Understanding:** I can explain the concept, tradeoffs, and failure modes independently.
- **Gate 2 — Implementation:** I can implement or meaningfully modify the component and validate it with executed tests.

Do not mark either gate passed without evidence; distinguish user-attested work from independently verified results.

## Suggested timebox
- 0–20 minutes: Distinguish conversation history, working memory, long-term memory, and execution checkpoints; agree on minimal design.
- 20–75 minutes: Implement SQLite `MemoryStore` and CRUD tests.
- 75–105 minutes: Implement `MemoryRetriever` and isolation/relevance tests.
- 105–120 minutes: Review, fix highest-priority failures, and record learning gates.

If time runs short, prioritize persistence and user isolation over retrieval sophistication.

## Expected output
- Runnable Python package, minimal README, and pytest tests.
- Actual test commands and outputs, including failures if any.
- Short architecture explanation, explicit limitations, and security considerations.
- A summary of what I personally implemented versus what the coding agent generated.
- Next-session suggestion: knowledge wiki or context assembly, based on the gaps discovered.

## Start here
First explain the differences among conversation history, working memory, long-term memory, and execution checkpoints in practical agent applications. Then propose the smallest project structure and ask **one** design question. **Wait for my response before writing code.**
