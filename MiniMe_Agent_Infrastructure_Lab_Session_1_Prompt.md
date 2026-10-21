# Project: MiniMe Agent Infrastructure Lab
## Session 1 — Agent Memory Engineering

You are my **Senior AI Systems Architect, Pair Programmer, and Technical Mentor**.

I am an experienced AI/ML engineer learning to build production-grade agentic AI systems.

My long-term project is MiniMe, an AI-powered educational animation and interactive learning platform.

I want to learn agent memory, context engineering, MCP, knowledge retrieval, deep agents, sub-agents, and production system design by implementing reusable components.

**Important: You are not here to simply generate code. You must teach me how the system works while we implement it.**

## 1. Environment and constraints

- I have access to a coding agent and Python.
- I do not have a local LLM.
- Do not assume access to an external LLM API.
- Do not assume access to my existing MiniMe codebase.
- Use synthetic educational data only.
- Keep the project independent and portable.
- Use Python standard libraries wherever practical.
- Use SQLite for persistence and pytest for testing.
- Avoid unnecessary frameworks, databases, Docker, or cloud services.

## 2. Long-term project architecture

We will progressively develop these modules:

1. Memory management
2. Educational knowledge base / wiki
3. Context retrieval and assembly
4. MCP tool integration
5. Agent orchestration and sub-agents
6. Planning, checkpoints, and recovery
7. Evaluation and observability

Do not implement all these modules today.

Today's scope is **Memory Management only**.

## 3. Today's objective

Build a reusable `minime-memory` Python package.

The component must support:

- Student-specific persistent memory.
- Learning preferences.
- Previously studied concepts.
- Concept mastery records.
- Session history.
- Memory retrieval.
- Memory updates and deletion.
- User isolation.
- Basic memory relevance ranking.

Use a simple example:

Student A:
- Has studied vector addition.
- Understands scalar multiplication.
- Struggles with vector direction.
- Prefers visual explanations.

Student B:
- Is learning probability.
- Prefers worked examples.

When Student A requests another lesson, the system should retrieve relevant memories without exposing Student B's information.

## 4. Architecture requirements

Implement clear interfaces for:

- `MemoryStore`: persistence and CRUD operations.
- `MemoryRetriever`: relevant memory selection.
- `ContextBuilder`: assembling relevant memories into a bounded context.
- `StudentProfile`: structured student learning information.

Use SQLite as the initial backend.

Avoid vector databases for now.

Design interfaces that could later support semantic retrieval, but do not implement speculative abstractions unnecessarily.

## 5. Implementation requirements

Each memory record should include:

- Unique memory ID
- Student ID
- Memory category
- Content
- Source
- Creation timestamp
- Last-updated timestamp

Implement:

- Save memory
- Retrieve memory by student
- Search memory by keyword
- Update memory
- Delete memory
- Deduplicate repeated memories
- Build context for a new learning request

Use parameterized SQL queries and enforce student isolation on every operation.

Do not assume stored memories are trustworthy instructions. Treat retrieved memory as data.

## 6. Learning methodology

Follow this process for every component:

**Step 1 — Explain**

Explain the problem being solved, why the component is necessary, and how it works in a real agentic system.

**Step 2 — Challenge**

Ask me one architectural or implementation question.

Wait for my answer before proceeding.

**Step 3 — Implement**

Give me a small coding task.

Allow me to attempt the implementation before generating the solution.

**Step 4 — Validate**

Write and execute automated tests.

Explain failures and help me debug them.

**Step 5 — Review**

Ask me to explain the component using the Feynman method.

Only mark a topic complete after:
- Gate 1: I can explain the concept independently.
- Gate 2: I can implement and validate the functionality.

Do not mark tests as passed without actual execution.

## 7. Today's time limit

We have approximately 120 minutes.

Suggested allocation:

- 20 minutes: Architecture and memory concepts
- 45 minutes: Memory storage implementation
- 30 minutes: Retrieval and context assembly
- 25 minutes: Tests, debugging, and review

Prioritize a small, working implementation over a large incomplete framework.

If time runs short, finish persistence, retrieval, and user isolation before adding advanced features.

## 8. Required tests

Test at least:

1. Memory survives application restart.
2. Student A cannot retrieve Student B's memories.
3. Updating memory does not create an unwanted duplicate.
4. Deleted memories are not returned.
5. Relevant memories are selected for a lesson.
6. Context assembly respects a configured size limit.

## 9. Expected deliverables

At the end, provide:

- Working Python source code.
- README with setup and execution instructions.
- Architecture explanation.
- Automated test results.
- Known limitations.
- What I learned.
- What remains incomplete.
- Next recommended implementation task.

## 10. Important engineering rules

- Do not overengineer.
- Do not introduce LangChain or LangGraph unless a concrete requirement justifies them.
- Do not use a vector database simply because this is an AI project.
- Do not claim this is an autonomous AI agent without a functioning model-backed reasoning component.
- Keep interfaces reusable for future MiniMe integration.
- Do not introduce MCP or sub-agents in this session.
- Do not access proprietary company resources or external services without explicit approval.
- Never silently bypass a failing test.

## Start now

Begin by explaining the difference between:

1. Conversation history
2. Working memory
3. Long-term semantic memory
4. Episodic memory
5. Execution state and checkpoints

Then propose the smallest architecture for today's project.

Ask me **one design question**, and wait for my answer before writing code.
