# RetinaAgent --- Triage Agent

## Overview

The **Triage Agent** is the fifth stage of the RetinaAgent pipeline. It
converts the upstream grading/consistency information and available
clinical cues into a structured follow-up category.

Its output is consumed by the **Explanation Agent**.

> **Research prototype:** The implemented triage policy is an MVP
> implementation and is not a clinically validated referral protocol.

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
```

------------------------------------------------------------------------

## Objectives

The Triage Agent is responsible for:

-   interpreting the upstream DR grade,
-   considering consistency/review information,
-   checking configured emergency cues,
-   assigning a follow-up urgency category,
-   incorporating available access/constraint information,
-   flagging cases that require additional review,
-   producing a structured blackboard output.

The Triage Agent does **not** perform the retinal image grading itself.

------------------------------------------------------------------------

## Inputs

The agent can use information such as:

### 1. DR Grade

The predicted diabetic-retinopathy grade from the upstream Grader Agent.

The prototype uses the five-class grading formulation:

``` text
0 → No DR
1 → Mild
2 → Moderate
3 → Severe
4 → Proliferative
```

### 2. Consistency Information

The Consistency Checker can flag unstable predictions.

The implemented review rule is based on test-time augmentation:

``` text
TTA grade spread > 1
        ↓
review_required = True
```

This allows the pipeline to avoid treating an unstable prediction as
completely reliable.

### 3. Emergency Cues

The prototype supports hard escalation cues including:

-   vitreous hemorrhage,
-   sudden vision loss,
-   rubeosis.

These cues can trigger urgent handling independently of the normal
grade-based pathway.

### 4. Access / Constraint Information

Where available, access constraints can be passed to the triage logic so
that the output remains structured around the information available to
the system.

------------------------------------------------------------------------

## Prototype Grade-to-Triage Policy

The current implementation uses:

    DR Grade Base Follow-up Tier
  ---------- ---------------------
           0 routine
           1 routine
           2 soon
           3 urgent
           4 urgent

``` python
BASE_TIER_BY_GRADE = {
    0: "routine",
    1: "routine",
    2: "soon",
    3: "urgent",
    4: "urgent",
}
```

**Important:** This table is an implementation parameter for the
research MVP. It should not be presented as a validated clinical
guideline.

------------------------------------------------------------------------

## Decision Flow

``` text
                 Upstream outputs
                       │
                       ▼
                Read DR grade
                       │
                       ▼
              Check emergency cues
                 │             │
               found         none
                 │             │
                 ▼             ▼
              urgent       grade-based
                              tier
                                │
                                ▼
                    Check consistency/review
                                │
                                ▼
                         Structured output
```

------------------------------------------------------------------------

## Example Output

Conceptually, the blackboard contains information similar to:

``` python
{
    "agent": "TriageAgent",
    "triage_category": "routine",
    "reason": "...",
    "review_required": False,
    ...
}
```

The exact serialized fields depend on the current notebook
implementation.

------------------------------------------------------------------------

## Blackboard

The output is stored for downstream agents through the project
blackboard mechanism.

Typical artifact:

``` text
triage_blackboard.pt
```

The exact Kaggle path can vary between notebook sessions.

------------------------------------------------------------------------

## Validation

The notebook validates that:

-   required upstream information is available,
-   the triage result has the expected structure,
-   the result can be serialized,
-   the downstream Explanation Agent can consume the blackboard.

------------------------------------------------------------------------

## Relationship to the Paper

The paper places the Triage Agent after the consistency gate and before
the Explanation Agent. It describes emergency cues and
grade/urgency/access information as inputs to triage.

The exact grade-to-tier mapping above is an implementation choice for
this MVP rather than a table specified by the paper.

------------------------------------------------------------------------

## Limitations

-   Not clinically validated.
-   Not intended for autonomous medical referral.
-   Emergency detection depends on the evidence supplied to the module.
-   The grade-to-tier policy is configurable MVP logic.
-   Clinical decisions require qualified healthcare professionals and
    appropriate clinical protocols.
