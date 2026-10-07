# Daily Execution Skill

## Purpose

Convert a high-level engineering task into a small, measurable, executable daily increment.

The goal is NOT to finish the entire project today.

The goal is to:

**Make one meaningful, measurable step forward today, learn from it, and use that feedback to decide the next step.**

Core principle:

> Don't build the building in one day. Complete one useful, testable part of the building today.

---

## 1. Inputs

The engineer should ideally provide:
- High-level task
- Available time
- Current state, if known
- Constraints, if known

Example:

```text
High-level task:
Integrate the vLLM inference service with our application.

Available time:
90 minutes.

Current state:
vLLM works independently.
```

If available time is not provided, ask for it when it materially changes the plan.

---

## 2. Understand the High-Level Goal

First determine: **What are we ultimately trying to achieve?**

Express it in one sentence. Do NOT immediately create tasks. First understand the destination.

---

## 3. Inspect the Current State

Determine where we are now. Separate:

### DONE
Things that already work.

### NOT DONE
Things that clearly don't work yet.

### UNKNOWN
Things that need validation.

Do not treat UNKNOWN as a problem until evidence confirms it.

---

## 4. Find Today's Smallest Useful Increment

Ask:

> What is the smallest useful thing we can complete today that moves the larger system forward?

A good daily increment should be:
- achievable within available time
- independently testable
- observable
- useful to the larger goal
- capable of producing feedback

Prefer vertical progress over broad unfinished work.

BAD: "Work on vLLM integration."

GOOD: "Send one request from the application to the inference service and receive one valid model response."

---

## 5. Scope Check

Before accepting today's task, estimate whether it fits the available time.

If estimated work > available time: **SHRINK THE TASK.**

Continue shrinking until estimated work fits the available time. When uncertain, prefer the smaller scope.

### Hard Rule

If the task cannot reasonably be completed during today's session, shrink the task. Do NOT automatically extend the working session.

---

## 6. Define Today's Outcome

Create exactly ONE primary outcome.

```text
TODAY'S OUTCOME

By the end of this session:

________________________________
```

The outcome must be verifiable.

Avoid vague outcomes such as:
- Learn vLLM
- Understand Kubernetes
- Explore API integration
- Work on performance

Replace them with measurable outcomes, such as:
"Measure baseline TTFT for 10 representative requests."

---

## 7. Define the Success Metric

Ask: **How will we KNOW today's task is finished?**

Prefer binary or numeric criteria.

Examples:
- HTTP status = 200
- Response contains generated_text
- 10/10 test requests succeed
- P95 latency recorded
- Unit test passes
- Container starts successfully
- GPU utilization measurement captured
- Architecture decision documented

There should be a clear moment when we can say: **DONE.**

---

## 8. Break the Increment Into Micro-Milestones

Create a maximum of **3–5 milestones**.

Each milestone should preferably take approximately **15–30 minutes**.

Every milestone must contain:

```text
TASK
What exactly should I do?

OUTPUT
What should exist afterward?

SUCCESS TEST
How do I know it worked?

TIMEBOX
How much time should I spend?
```

---

## 9. Execute One Milestone at a Time

Do NOT dump the entire implementation on the engineer unless requested.

Guide execution sequentially:

```text
CURRENT MILESTONE: 1

Goal:
...

Do:
...

Expected result:
...

Success test:
...
```

Wait for evidence when practical before progressing.

---

## 10. Evidence Before Progress

Do not mark a milestone complete merely because code was written.

Completion requires evidence.

```text
Code written               != Done
Code executed successfully  = Evidence

Deployment YAML created    != Done
Pod running                 = Evidence

Optimization added         != Done
Latency improved            = Evidence
```

Prefer:

**BUILD → RUN → OBSERVE → VERIFY**

---

## 11. Handle Failure as Feedback

If a milestone fails, do NOT immediately redesign everything.

Capture:

```text
EXPECTED
What should have happened?

ACTUAL
What happened?

EVIDENCE
Logs / Errors / Metrics / Responses

HYPOTHESIS
What might explain the difference?
```

Then choose the smallest diagnostic action. If necessary, invoke Root Cause Analysis.

Failure is information.

---

## 12. Prevent Scope Creep

New ideas will appear during implementation.

Do NOT automatically add them to today's work.

Create a **PARKING LOT**.

These are candidates for future increments. They are NOT today's tasks.

---

## 13. Learning Mode

When the purpose includes developing the engineer's skills, avoid immediately providing the complete answer.

Prefer:

```text
ENGINEER THINKS
      ↓
ENGINEER PROPOSES
      ↓
AI QUESTIONS / CRITIQUES
      ↓
ENGINEER IMPLEMENTS
      ↓
SYSTEM PROVIDES FEEDBACK
```

Use hints progressively:

**Hint 1 → direction → Hint 2 → concept → Hint 3 → pseudocode → Hint 4 → implementation**

This prevents productivity from replacing learning.

---

## 14. Timebox Protection

Periodically compare **TIME REMAINING vs WORK REMAINING**.

If there is not enough time, do NOT rush through remaining milestones. Finish the current useful milestone properly and move the rest to the next session.

Quality of feedback is more important than the number of tasks completed.

---

## 15. End-of-Day Validation

At the end of the session, compare **PLANNED vs ACTUAL**.

Ask:
1. What did we intend to achieve?
2. What actually worked?
3. What didn't work?
4. What evidence do we have?
5. What blocked progress?
6. What did we learn?
7. Was today's task sized correctly?

---

## 16. Score the Session

Use a lightweight scorecard:

```text
Outcome achieved:      YES / PARTIAL / NO
Milestones completed:  X / Y
Time estimate:         GOOD / TOO LARGE / TOO SMALL
Evidence collected:    YES / NO
New lesson:            YES / NO
```

A session is still valuable when the original outcome fails if it produced useful evidence.

---

## 17. Generate the Next Increment

Do NOT automatically create a giant tomorrow plan.

Ask:

> Given today's evidence, what is the smallest logical next increment?

Tomorrow should build on today's evidence.

---

## 18. Engineering Memory

Record important lessons when they are reusable.

```text
DATE:
PROJECT:
PROBLEM:
TODAY'S GOAL:
RESULT:
EVIDENCE:
WHAT FAILED:
ROOT CAUSE:
LESSON:
NEXT INCREMENT:
```

Avoid storing trivial implementation details. Store information that could improve future engineering decisions.

---

## Standard Daily Output

Whenever this skill is invoked, produce:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAILY EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIGH-LEVEL GOAL
...

CURRENT STATE
...

TODAY'S INCREMENT
...

WHY THIS INCREMENT
...

TIME BUDGET
...

SUCCESS CRITERIA
...

────────────────────────────

M1 — ...
Task:
Output:
Success test:
Timebox:

M2 — ...
Task:
Output:
Success test:
Timebox:

M3 — ...
Task:
Output:
Success test:
Timebox:

────────────────────────────

PARKING LOT
...

────────────────────────────

FINAL TEST

By the end of today:

[ ] ______________________

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## End-of-Day Output

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAILY REVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLANNED
...

ACHIEVED
...

RESULT
SUCCESS / PARTIAL / FAILED

EVIDENCE
...

BLOCKERS
...

LESSON
...

PARKING LOT
...

NEXT SMALLEST INCREMENT
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Core Rules

1. One meaningful outcome per session.
2. Maximum 3–5 micro-milestones.
3. Every milestone requires a success test.
4. Evidence beats assumption.
5. Unknown does not automatically mean broken.
6. New ideas go into the Parking Lot.
7. If the task doesn't fit the available time, shrink it.
8. Failure with evidence is useful progress.
9. Today's evidence determines tomorrow's task.
10. Never try to build the building in one day.

**Build one brick, verify the brick, learn from the brick, then choose the next brick.**
