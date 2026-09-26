# RetinaAgent --- Verifier Agent

## Overview

The **Verifier Agent** is the final evidence-grounding stage of
RetinaAgent.

Its purpose is to check whether generated explanation claims can be
traced to the evidence available to the system:

``` text
L = image localization / visual evidence
e = clinical-note entities
R = retrieved guideline evidence
```

Supported claims are retained. Unsupported claims are dropped and logged
as hallucination events.

> **Research prototype:** Verification here means evidence traceability.
> It does not establish clinical truth or replace clinical validation.

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
┌──────────────────┐
│  Verifier Agent  │
└──────────────────┘
```

------------------------------------------------------------------------

## Core Principle

The Verifier follows:

``` text
Explanation Agent
       │
       ├── claims
       │
       └── evidence
             │
             ├── L: image
             ├── e: clinical note
             └── R: retrieved guidelines
                       │
                       ▼
                  Verifier
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Supported            Unsupported
             │                   │
            Keep            Drop + Log
```

This implements the paper's central anti-fabrication mechanism.

------------------------------------------------------------------------

## Input

The Verifier consumes the **structured `claims` field directly** from
the Explanation Agent.

Conceptually:

``` python
claims = explanation_output["claims"]
```

A claim has a structure similar to:

``` python
{
    "claim": "The current follow-up category is routine.",
    "output": "patient"
}
```

The Verifier does **not** reconstruct claims from arbitrary string
representations of model outputs.

This prevents artifacts such as:

``` text
[0.
0.
0.
...
```

from being incorrectly treated as natural-language claims.

------------------------------------------------------------------------

## Evidence Input

The Explanation Agent provides:

``` python
evidence = {
    "image": ...,
    "clinical_note": ...,
    "retrieved": ...
}
```

The Verifier converts these evidence objects into searchable
representations and evaluates each claim against the available sources.

------------------------------------------------------------------------

## Claim Verification

For every claim:

``` text
1. Normalize claim
2. Compare with L
3. Compare with e
4. Compare with R
5. Determine traceability
6. Keep or drop
7. Record verification result
```

The current MVP uses a **lightweight lexical overlap test**.

It is therefore important to distinguish:

``` text
Traceability ≠ Clinical truth
```

A production implementation could replace the lexical test with a
stronger semantic-entailment/NLI mechanism while preserving the same
provenance architecture.

------------------------------------------------------------------------

## Verification Status

The implementation uses three statuses.

### VERIFIED

All claims pass the configured traceability test.

``` text
Dropped = 0
```

### PARTIALLY_VERIFIED

At least one claim is supported and at least one claim is rejected.

``` text
Supported > 0
Dropped > 0
```

### UNVERIFIED

No claim is supported.

``` text
Supported = 0
Dropped > 0
```

------------------------------------------------------------------------

## Unsupported Claims

Unsupported claims are not silently ignored.

They are stored in:

``` python
dropped_claims
```

and:

``` python
hallucination_events
```

A hallucination event contains information similar to:

``` python
{
    "claim": "...",
    "output": "clinician",
    "reason": "Claim could not be traced to image localization L, clinical-note entities e, or retrieved evidence R.",
    "evidence_source": None,
    "overlap": 0.0
}
```

This provides an auditable record of why a generated statement was
rejected.

------------------------------------------------------------------------

## Output

The final verifier blackboard contains fields such as:

``` python
{
    "agent": "VerifierAgent",
    "verification_status": "...",
    "clinician_explanation": "...",
    "patient_explanation": "...",
    "claims_checked": 0,
    "supported_claims": 0,
    "dropped_claims": [...],
    "hallucination_events": [...],
    "evidence_traceability": {
        "image_localization": True,
        "clinical_note": True,
        "retrieved_guidelines": False
    },
    "claim_results": [...],
    "review_required": True
}
```

The exact values depend on the Explanation Agent output.

------------------------------------------------------------------------

## Blackboard

Typical output:

``` text
/kaggle/working/verifier_blackboard.pt
```

This artifact contains the final verification result.

------------------------------------------------------------------------

## Validation

The notebook validates:

-   Explanation Agent blackboard exists,
-   the blackboard can be loaded,
-   the nested Explanation Agent output is correctly unwrapped,
-   `claims` exists and is a list,
-   every claim is a valid text claim,
-   evidence is available as a structured object,
-   claim counts are internally consistent,
-   the final verifier blackboard can be saved.

A correct run reports:

``` text
Verifier Agent validation PASSED.
Status: <VERIFIED | PARTIALLY_VERIFIED | UNVERIFIED>
Claims checked: <N>
Supported: <N>
Dropped: <N>
Saved: /kaggle/working/verifier_blackboard.pt
```

------------------------------------------------------------------------

## Important Sanity Check

The number of claims should never unexpectedly become zero.

The verifier explicitly checks:

``` python
len(claims) > 0
```

after loading the Explanation Agent blackboard.

If the Explanation Agent reports:

``` text
Structured claims: 42
```

the Verifier should receive the same structured claim list.

------------------------------------------------------------------------

## Research Limitations

1.  The current lexical overlap method is an MVP traceability
    implementation.
2.  It is not a full semantic-entailment model.
3.  Passing verification does not prove medical correctness.
4.  Dropping a claim means it could not be traced under the configured
    verifier rule; it does not by itself prove that the underlying
    statement is medically false.
5.  Retrieved guideline evidence must actually be present to provide
    R-based grounding.
6.  The system is a research prototype and is not clinically validated.

------------------------------------------------------------------------

## Final Role in RetinaAgent

The Verifier provides the final safeguard:

``` text
Generate
   ↓
Structure claims
   ↓
Trace to L + e + R
   ↓
Keep supported claims
   ↓
Drop + log unsupported claims
```

This creates an auditable boundary between **generated explanation** and
**evidence-grounded explanation**.
