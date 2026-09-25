# RetinaAgent — Clinical-Note Agent

## Kaggle Notebook

The implementation is maintained in the Kaggle notebook:

**Kaggle:** https://www.kaggle.com/code/premshaw23/notebook884288cf97/edit

Use the notebook to run and inspect the Clinical-Note Agent implementation described in this README.

## 1. Purpose

The **Clinical-Note Agent** is the note/context component of the RetinaAgent pipeline.

According to the RetinaAgent paper, its job is to parse referral/clinical information into a structured entity representation so that the **Grader Agent can combine clinical context with the fundus-image representation**.

The paper describes the overall flow as:

```text
Fundus Image ──> Image Agent ───────────────┐
                                            │
Clinical / Referral Note ──> Note Agent ────┤
                                            ▼
                                      Grader Agent
                                            │
                                            ▼
                                      Triage Agent
                                            │
                                            ▼
                                   Explanation Agent
                                            │
                                            ▼
                                        Verifier
```

The paper also specifies a typed blackboard / LangGraph-style state so that agents communicate through shared state rather than directly calling one another.

---

## 2. What the Clinical-Note Agent takes as input

### Current implementation

The current MVP implementation takes **one BRSET record**:

```python
record
```

The record is loaded from:

```text
BRSET_16266.json
```

The function is:

```python
note_output = clinical_note_agent(record)
```

The current implementation uses information that is actually available in the BRSET record.

### Important limitation

BRSET is being used here as a source of **structured clinical/patient metadata** for the MVP. It is not being treated as a literal free-text referral-note dataset.

Therefore:

- available information can be returned;
- unavailable information stays `None` or `[]`;
- no clinical fact is inferred from unrelated fields;
- no value is fabricated.

This is consistent with the paper's requirement that absent clinical fields are defaulted/flagged rather than fabricated.

---

## 3. What the paper says the Note Agent should extract

The paper defines the referral-note entity representation using:

1. Visual acuity
2. Lens status
3. Prior laser / anti-VEGF treatment
4. HbA1c
5. Diabetes duration
6. Prior vitrectomy
7. Symptoms
8. Image-quality comment

The paper describes these as the structured clinical entity vector `e`.

### Current implementation contract

For the current MVP, the requested function output is:

```python
{
    "image_id": "...",

    "visual_acuity": {
        "right_eye": None,
        "left_eye": None
    },

    "lens_status": None,
    "prior_laser": None,
    "prior_anti_vegf": None,
    "hba1c": None,
    "diabetes_duration": None,
    "prior_vitrectomy": None,
    "symptoms": []
}
```

`image_id` is included specifically as the **case linkage key** for downstream integration.

> Note: the nested `right_eye` / `left_eye` representation is the implementation contract used in this MVP. The paper itself specifies visual acuity as an entity but does not prescribe this exact nested JSON/Python structure.

---

## 4. Output type

The Clinical-Note Agent does **not** need to write a JSON/CSV file during the agent pipeline.

It returns a normal **Python dictionary**:

```python
note_output = clinical_note_agent(record)
```

Example:

```python
{
    "image_id": "/All_data/BRSET_16266/1.0.0/fundus_photos/img00001.jpg",
    "visual_acuity": {
        "right_eye": None,
        "left_eye": None
    },
    "lens_status": None,
    "prior_laser": None,
    "prior_anti_vegf": None,
    "hba1c": None,
    "diabetes_duration": 12.0,
    "prior_vitrectomy": None,
    "symptoms": []
}
```

This object is what should be passed to the next stage.

---

## 5. Meaning of each output field

| Field | Meaning | Missing value |
|---|---|---|
| `image_id` | Identifier used to associate the clinical record with the image/case | Should not be missing |
| `visual_acuity.right_eye` | Right-eye visual acuity | `None` |
| `visual_acuity.left_eye` | Left-eye visual acuity | `None` |
| `lens_status` | Lens/cataract status | `None` |
| `prior_laser` | Previous laser treatment information | `None` |
| `prior_anti_vegf` | Previous anti-VEGF treatment information | `None` |
| `hba1c` | HbA1c value | `None` |
| `diabetes_duration` | Diabetes duration | `None` |
| `prior_vitrectomy` | Previous vitrectomy information | `None` |
| `symptoms` | Reported symptoms | `[]` |

### Anti-fabrication rule

Never do:

```python
hba1c = 7.2
```

if the source record does not contain HbA1c.

Never infer:

```python
prior_laser = True
```

from some unrelated field.

Instead:

```python
"hba1c": None
"prior_laser": None
```

The paper explicitly requires missing fields to be handled without fabrication.

---

## 6. Current function

The current implementation is:

```python
def clinical_note_agent(record):
    diabetes_duration = None

    if record.get("diabetes_time_y") not in [None, "", "nan"]:
        try:
            diabetes_duration = float(record["diabetes_time_y"])
        except (ValueError, TypeError):
            diabetes_duration = None

    return {
        "image_id": record["ImageID"],

        "visual_acuity": {
            "right_eye": None,
            "left_eye": None
        },

        "lens_status": None,
        "prior_laser": None,
        "prior_anti_vegf": None,
        "hba1c": None,
        "diabetes_duration": diabetes_duration,
        "prior_vitrectomy": None,
        "symptoms": []
    }
```

Usage:

```python
note_output = clinical_note_agent(record)
```

---

# 7. Connection to the Grader Agent

This is the most important downstream connection.

The paper defines the Grader as combining:

```text
Image representation z_v
        +
Clinical entity representation e
        ↓
Note encoder g_phi(e)
        ↓
Fusion
        ↓
Final grade + confidence
```

Mathematically, the paper describes:

```text
p = softmax(W_f [ z_v ⊕ g_phi(e) ])
```

where:

- `z_v` = image embedding
- `e` = structured clinical-note entities
- `g_phi(e)` = encoded note representation
- `⊕` = concatenation
- `p` = final five-class probability distribution

The final outputs are:

```text
predicted grade = argmax(p)
confidence       = max(p)
```

---

## 8. Important: the Note Agent does NOT directly return an embedding

The current Note Agent returns:

```python
note_output
```

which is a structured dictionary.

The **Grader stage** is responsible for converting the structured clinical information into a numerical representation:

```text
note_output
    ↓
note encoder
    ↓
g_phi(e)
    ↓
fusion with image embedding
```

Therefore, do not make the Note Agent itself return the final fused embedding.

---

## 9. Intended Python interface

The clean interface is:

```python
image_output = image_agent(image)

note_output = clinical_note_agent(record)

grader_output = grader_agent(
    image_output=image_output,
    note_output=note_output
)
```

The Grader then performs the note encoding and multimodal fusion.

Conceptually:

```python
def grader_agent(image_output, note_output):

    image_embedding = image_output["image_embedding"]

    note_embedding = encode_note(note_output)

    fused = fuse(
        image_embedding,
        note_embedding
    )

    grade = ...
    confidence = ...

    return {
        "image_id": note_output["image_id"],
        "grade": grade,
        "confidence": confidence
    }
```

The exact fusion dimensions and architecture still need to be defined/trained; they should not be invented merely to claim that the paper's full Grader has already been implemented.

---

# 10. `image_id` linkage

`image_id` is the key used to associate the two agent outputs.

Conceptually:

```text
Image Agent output
{
    "image_id": "img001",
    "image_embedding": ...
}
                 │
                 │ same image_id
                 ▼
Clinical Note Agent output
{
    "image_id": "img001",
    ...
}
                 │
                 ▼
             Grader
```

The important rule is:

```text
image_id(Image Agent) == image_id(Note Agent)
```

before treating the two outputs as the same case.

---

# 11. Critical dataset-linkage warning

The current MVP uses:

```text
Image Agent → APTOS
Note Agent  → BRSET
```

These are different datasets.

Therefore, **do not match records simply by row number**:

```python
aptos[0] + brset[0]     # WRONG
aptos[1] + brset[1]     # WRONG
```

That would not establish that the image and clinical record belong to the same case.

The `image_id` field provides the mechanism for matching, but it does not magically make two unrelated datasets share the same case IDs.

Before training a multimodal Grader, a legitimate same-case image/clinical linkage is required.

---

# 12. What the Grader should receive

The Grader interface should ultimately receive two pieces of information:

```text
1. Image Agent output
2. Clinical-Note Agent output
```

For example:

```python
image_output = {
    "image_id": "...",
    "image_embedding": ...,
    "predicted_grade": ...,
    "confidence": ...,
    "heatmap": ...
}
```

and:

```python
note_output = {
    "image_id": "...",
    "visual_acuity": {
        "right_eye": None,
        "left_eye": None
    },
    "lens_status": None,
    "prior_laser": None,
    "prior_anti_vegf": None,
    "hba1c": None,
    "diabetes_duration": None,
    "prior_vitrectomy": None,
    "symptoms": []
}
```

The Grader combines the image representation and encoded clinical representation.

---

# 13. Connection to LangGraph

The paper specifies a typed-blackboard architecture in which agents communicate through shared state rather than directly calling each other.

The intended graph is:

```text
image
  ↓
note
  ↓
grader
  ↓
triage
  ↓
explain
  ↓
verify
```

Conceptually:

```python
from langgraph.graph import StateGraph

g = StateGraph(dict)

g.add_node("image", image_agent)
g.add_node("note", clinical_note_agent)
g.add_node("grader", grader_agent)
g.add_node("triage", triage_agent)
g.add_node("explain", explanation_agent)
g.add_node("verify", verifier)

g.set_entry_point("image")

g.add_edge("image", "note")
g.add_edge("note", "grader")
g.add_edge("grader", "triage")
g.add_edge("triage", "explain")
g.add_edge("explain", "verify")

app = g.compile()
```

The paper's implementation sketch uses this same sequence.

For the current MVP, however, the agents can first be tested as normal Python functions. LangGraph can be added when the individual interfaces are stable.

---

# 14. What comes after the Grader

According to the paper:

```text
Clinical-Note Agent
        +
Image Agent
        ↓
Grader Agent
        ↓
Triage Agent
        ↓
Explanation Agent
        ↓
Verifier
```

### Grader

Produces:

```text
DR grade
confidence
```

The grading task uses five classes:

```text
0 = No DR
1 = Mild NPDR
2 = Moderate NPDR
3 = Severe NPDR
4 = PDR
```

### Triage

Uses the fused grade plus urgency/access information.

The paper specifies hard emergency cues including:

```text
vitreous hemorrhage
sudden vision loss
rubeosis
```

and otherwise maps the assessment to:

```text
routine
soon
urgent
emergency
```

### Explanation

Produces:

- clinician-facing explanation
- simplified patient-facing explanation

The explanation should be grounded in:

```text
L = image/lesion evidence
e = clinical-note entities
R = retrieved guideline evidence
```

### Verifier

Checks generated claims against:

```text
{L, e, R}
```

Unsupported claims are dropped/logged as hallucination events.

---

# 15. Current status

```text
Data loading                         DONE
BRSET clinical metadata loading      DONE
Clinical-note extraction             DONE
Missing-value handling               DONE
image_id preservation                DONE
Grader-ready function output         DONE

Same-case APTOS ↔ BRSET linkage      NOT YET ESTABLISHED
Grader implementation                NEXT
Triage                               AFTER GRADER
RAG / guideline retrieval            AFTER GRADER/Triage
Explanation                         AFTER RAG
Verifier                             LAST
LangGraph orchestration              AFTER interfaces stabilize
```

---

# 16. Final interface to remember

The most important contract is:

```python
note_output = clinical_note_agent(record)
```

returns:

```python
{
    "image_id": "...",
    "visual_acuity": {
        "right_eye": None,
        "left_eye": None
    },
    "lens_status": None,
    "prior_laser": None,
    "prior_anti_vegf": None,
    "hba1c": None,
    "diabetes_duration": None,
    "prior_vitrectomy": None,
    "symptoms": []
}
```

Then:

```python
grader_output = grader_agent(
    image_output=image_output,
    note_output=note_output
)
```

That is the **Note Agent → Grader Agent contract**.

---

## Source basis

This README follows the RetinaAgent paper's description of:

- the Clinical-Note Agent and its structured entities;
- missing-field handling;
- image/note multimodal fusion;
- the Grader's grade/confidence output;
- the blackboard/LangGraph agent sequence;
- downstream Triage, Explanation, and Verifier roles.

The paper specifically states that the Grader concatenates the image embedding with the encoded note entities through a fusion head, and that the agents communicate through a typed blackboard/StateGraph. It also specifies the sequence `image → note → grader → triage → explain → verify`.
