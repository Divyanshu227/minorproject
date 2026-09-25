# 👁️ RetinaAgent

RetinaAgent is a medical vision agent ecosystem for autonomous ophthalmic assessment, lesion segmentation, explainable localization, and multi-agent clinical decision support.

## 🌐 Kaggle Collection

All RetinaAgent notebooks, blackboards, and models are available in the official Kaggle collection:

**Collection Link:** [RetinaAgent Kaggle Collection (19217866)](https://www.kaggle.com/work/collections/19217866)

[![Kaggle Collection](https://img.shields.io/badge/Kaggle-RetinaAgent%20Collection-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/work/collections/19217866)

## 📂 Submodules

- **[ImageAgent](file:///c:/Users/divya/Desktop/minorproject/RetinaAgent/ImageAgent/README.md)**: Multi-task Swin-Transformer (Swin-T) vision agent for Diabetic Retinopathy 5-stage grading, IDRiD lesion segmentation (Microaneurysms, Haemorrhages, Hard/Soft Exudates), Grad-CAM explainability, connected component bounding box extraction, and Test-Time Augmentation (TTA) uncertainty guardrails.
- **[Clinical Note Agent](file:///c:/Users/divya/Desktop/minorproject/RetinaAgent/clinical_note_agent/RETINAGENT_NOTE_AGENT_README.md)**: Structured clinical entity extraction agent adhering to strict anti-fabrication guidelines for patient referral records.
- **[Grader Agent](file:///c:/Users/divya/Desktop/minorproject/RetinaAgent/GraderAgent/README.md)**: Central diagnostic synthesis and blackboard state hub combining fundus vision evidence and clinical context with uncertainty gating.
- **[Consistency Checker Agent](file:///c:/Users/divya/Desktop/minorproject/RetinaAgent/ConsitencyCheckerAgent/README.md)**: Clinical safety and verification agent auditing Test-Time Augmentation (TTA) consistency, confidence calibration, and referral completeness.

For complete architectural details, please refer to the respective agent documentation.
