# RetinaAgent --- Explanation Agent

## Overview

The **Explanation Agent** is responsible for converting the upstream
RetinaAgent results into grounded explanations for two audiences:

1.  **Clinicians**
2.  **Patients**

It also creates **structured claims** and preserves the evidence used to
generate those claims so that the downstream **Verifier Agent** can
perform traceability checking.

> **Research prototype:** The generated explanations are for research
> and evaluation and are not a substitute for clinical judgment.

------------------------------------------------------------------------

## Pipeline Position

``` text
Image Agent
     ↓
Clinical-Note Agent
     ↓
Grader Agent
     ↓
Consistency Checker
     ↓
Triage Agent
     ↓
Explanation Agent
     ↓
Verifier Agent
```

------------------------------------------------------------------------

## Objectives

The Explanation Agent:

-   receives the upstream triage result,
-   receives image/visual evidence,
-   receives structured clinical-note evidence,
-   receives retrieved guideline evidence when available,
-   generates a clinician-facing explanation,
-   generates a patient-facing explanation,
-   converts generated statements into structured claims,
-   stores evidence provenance,
-   passes the structured output to the Verifier Agent.

------------------------------------------------------------------------

## Evidence Model

The Explanation Agent maintains three main evidence sources:

``` text
L = image localization / visual evidence
e = clinical-note entities
R = retrieved guideline evidence
```

The structured evidence object follows the implementation pattern:

``` python
evidence = {
    "image": ...,
    "clinical_note": ...,
    "retrieved": ...
}
```

These sources are later used by the Verifier Agent.

------------------------------------------------------------------------

## Generated Outputs

The Explanation Agent produces fields such as:

``` python
{
    "agent": "ExplanationAgent",
    "clinician_explanation": "...",
    "patient_explanation": "...",
    "claims": [...],
    "evidence": {
        "image": ...,
        "clinical_note": ...,
        "retrieved": ...
    },
    "grounding_policy": ...
}
```

The exact serialized object depends on the current notebook version.

------------------------------------------------------------------------

## Structured Claims

A key implementation feature is the explicit `claims` field.

Conceptually:

``` python
[
    {
        "claim": "The retinal image shows ...",
        "output": "clinician"
    },
    {
        "claim": "The current follow-up category is ...",
        "output": "patient"
    }
]
```

The Verifier Agent consumes this structured list directly.

### Why structured claims matter

The Verifier should not attempt to recover claims by converting
arbitrary model objects or numeric arrays into strings.

The implemented pipeline therefore follows:

``` text
Explanation text
       ↓
Structured claims
       ↓
Verifier
```

rather than:

``` text
Explanation object
       ↓
string conversion
       ↓
guessing claims
```

------------------------------------------------------------------------

## Clinician Explanation

The clinician-facing explanation is intended to connect:

``` text
visual findings
      +
clinical context
      +
retrieved evidence
      +
triage result
```

The explanation should remain grounded in information actually available
to the system.

------------------------------------------------------------------------

## Patient Explanation

The patient-facing explanation is a simplified version intended to
communicate the result without unnecessary technical terminology.

It must still follow the same evidence-grounding requirement.

------------------------------------------------------------------------

## Grounding Principle

The agent should not invent missing clinical information.

For example, if HbA1c is unavailable:

``` text
HbA1c = missing
```

must remain missing rather than being replaced with a guessed value.

The same principle applies to symptoms, prior treatment, lens status,
and other unsupported clinical entities.

------------------------------------------------------------------------

## Blackboard

The Explanation Agent produces a blackboard artifact consumed by the
Verifier Agent.

Typical artifact:

``` text
explanation_blackboard.pt
```

In the current Kaggle workflow, the published notebook path may
resemble:

``` text
/kaggle/usr/lib/notebooks/premshaw23/
explanation_agent/explanation_blackboard.pt
```

The exact path can change between sessions.

------------------------------------------------------------------------

## Validation

The notebook validates that:

-   the Explanation Agent executes successfully,
-   clinician and patient explanations are produced,
-   structured claims are present,
-   evidence fields are available,
-   the blackboard can be consumed by the Verifier Agent.

A successful run reports:

``` text
Explanation Agent validation: PASSED
Structured claims: <N>
Blackboard ready for Verifier Agent.
```

------------------------------------------------------------------------

## Relationship to the Verifier

The Explanation Agent is intentionally separated from verification.

``` text
Explanation Agent
       │
       ├── clinician explanation
       ├── patient explanation
       ├── structured claims
       └── evidence
              │
              ▼
        Verifier Agent
```

The Explanation Agent generates the explanation.

The Verifier Agent decides whether each claim can be traced to the
available evidence.

------------------------------------------------------------------------

## Limitations

-   Retrieval evidence may be unavailable or empty in an MVP run.
-   Structured clinical metadata is not necessarily equivalent to full
    free-text referral notes.
-   Explanation generation does not establish clinical truth by itself.
-   Final verification checks traceability, not medical correctness.
