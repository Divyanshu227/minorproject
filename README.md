# 🩺 Multi-Disease Prediction & Progression Modeling

An end-to-end machine learning and deep learning framework for early diagnosis, severity staging, and progression analysis across major chronic illnesses: **Diabetic Retinopathy**, **Alzheimer's Disease**, and **Multi-Condition Chronic Disease Progression**.

---

## 📌 Project Overview

Chronic illnesses such as diabetes and neurodegenerative disorders place a severe burden on healthcare systems globally. This project leverages computer vision, deep transfer learning, supervised classification, regression modeling, and unsupervised patient segmentation to support clinical decision-making:

1. **Diabetic Retinopathy Classification**: Deep transfer learning (PyTorch + EfficientNet-B0) to grade retinal fundus images across 5 clinical severity levels.
2. **Alzheimer's Disease Detection**: Brain MRI classification across cognitive impairment stages (*Non-Demented*, *Mild Cognitive Impairment*, and *Moderate Dementia*).
3. **Parkinson's Disease Detection**: Deep transfer learning with Attention-based Multiple Instance Learning (PyTorch + ResNet-18 MIL) on DaTscan SPECT images for subject-level diagnosis.
4. **Chronic Disease Progression & Clinical Modeling**:
   - **Disease Classification**: Multi-class Random Forest predicting underlying diagnosis (*Diabetes*, *Alzheimer's*, *Parkinson's*).
   - **Progression Regression**: Multiple Linear Regression (MLR) predicting continuous biomarker score evolution.
   - **Unsupervised Patient Segmentation**: K-Means clustering identifying distinct patient risk profiles from multi-dimensional clinical vitals.

---

## 📂 Repository Structure

```text
minorproject/
├── Alzeimer-prediction/
│   └── Alzeimer/
│       └── combined_images/
│           ├── MildDemented/           # Augmented brain MRI scans (Mild stage)
│           ├── ModerateDemented/       # Augmented brain MRI scans (Moderate stage)
│           └── NonDemented/            # Brain MRI scans (Healthy / Non-Demented)
│
├── Diabetes-prediction/
│   ├── Diabetes/
│   │   ├── colored_images/             # Fundus photography images across 5 severity classes
│   │   │   ├── No_DR/
│   │   │   ├── Mild/
│   │   │   ├── Moderate/
│   │   │   ├── Severe/
│   │   │   └── Proliferate_DR/
│   │   └── train.csv                   # Image annotations and class labels
│   ├── best_diabetic_retinopathy_model.pth # Trained PyTorch EfficientNet-B0 weights
│   ├── confusion_matrix.png            # Model confusion matrix plot
│   └── train.ipynb                     # PyTorch training, validation, and evaluation pipeline
│
├── parkinsons-disease-prediction/
│   ├── ntua-parkinson-dataset-master/  # Multi-subject DaTscan SPECT and MRI benchmark (78 subjects)
│   ├── best_parkinsons_dat_model.pth   # Saved PyTorch Attention-MIL weights
│   ├── confusion_matrix.png            # Subject-level confusion matrix plot
│   ├── training_metrics.png            # Loss and accuracy progression curves
│   ├── roc_curve.png                   # Receiver Operating Characteristic (ROC) curve
│   └── train.py                        # Subject-level MIL training, validation, and evaluation script
│
└── Chronic disease prediction models/
    ├── chronic_disease_progression.csv.xls # Clinical dataset (3,000 patient records, 26 features)
    ├── clustering_classification.ipynb     # K-Means clustering & Random Forest disease classifier
    └── mlr_chronic_disease.ipynb           # Multiple Linear Regression for biomarker prediction
```

---

## 🚀 Key Modules & Methodologies

### 1. 👁️ Diabetic Retinopathy Classification
- **Architecture**: Pretrained `EfficientNet-B0` with custom classification head fine-tuned for retinal grading.
- **Preprocessing & Augmentation**: Image normalization using ImageNet statistics, random horizontal/vertical flips, and color jittering.
- **Classes (5 stages)**:
  - `0` - No DR
  - `1` - Mild
  - `2` - Moderate
  - `3` - Severe
  - `4` - Proliferative DR
- **Key Metrics**:
  - **Test Accuracy**: `76.91%`
  - **Quadratic Weighted Kappa (QWK)**: `0.8476` *(demonstrates strong agreement in ordinal disease severity)*
  - **Macro F1 Score**: `0.6109`

### 2. 🧠 Alzheimer's Disease Stage Classification
- **Architecture**: Pretrained `EfficientNet-B0` with custom classification head and fine-tuned top feature layers.
- **Data Modality**: Structural Brain Magnetic Resonance Imaging (MRI).
- **Target Categories (4 stages)**: `MildDemented`, `ModerateDemented`, `NonDemented`, and `VeryMildDemented`.
- **Key Metrics**:
  - **Test Accuracy**: `58.83%`
  - **Macro F1 Score**: `0.5793`
  - **Cohen's Kappa Score**: `0.4511`
  - **Artifacts**: `best_alzheimer_model.pth`, `confusion_matrix.png`, `training_metrics.png`, `train.ipynb`.

### 3. 🔬 Parkinson's Disease Detection (DaTscan SPECT)
- **Architecture**: Pretrained `ResNet-18` feature backbone paired with **Gated Attention Multiple-Instance Learning (MIL)** for subject-level feature pooling.
- **Data Modality**: Dopamine Transporter Single-Photon Emission Computed Tomography (DaTscan SPECT) imaging.
- **Target Categories**: `Parkinson's Disease (PD)` vs. `Non-PD Controls` (including real-world clinic differential diagnoses: Essential Tremor, SWEDD, Dystonia).
- **Key Metrics**:
  - **Test Accuracy (Subject-Level)**: `70.00%`
  - **Test Sensitivity / Recall (PD)**: `100.00%`
  - **Binary F1 Score**: `0.8235`
  - **Artifacts**: `best_parkinsons_dat_model.pth`, `confusion_matrix.png`, `training_metrics.png`, `roc_curve.png`, `train.py`.

### 4. 📊 Chronic Disease Progression & Patient Profiling
- **Dataset**: 3,000 multi-factorial patient records tracking demographics, vital signs, sleep, activity, biomarker scores, cognitive indices, and medication adherence.
- **Supervised Classification (Random Forest)**:
  - Predicts diagnosis (`Diabetes`, `Alzheimer's`, `Parkinson's`).
  - **Accuracy**: `90.50%` | **Macro F1**: `0.90`
- **Multiple Linear Regression (MLR)**:
  - Models biomarker degradation over clinical and lifestyle variables.
  - **$R^2$ Score**: `0.7971`
  - **Root Mean Squared Error (RMSE)**: `5.9637`
- **K-Means Clustering**:
  - Unsupervised grouping based on physiological parameters (`Age`, `BiomarkerScore`, `MedicationDose`, `HeartRate`, `StressLevel`, `CognitiveScore`, `MoodScore`).

---

## 📈 Performance Summary

| Task / Sub-Problem | Model / Technique | Key Evaluation Metric | Result |
| :--- | :--- | :--- | :--- |
| **Diabetic Retinopathy Grading** | PyTorch EfficientNet-B0 | Quadratic Weighted Kappa (QWK) | **0.8476** |
| | | Test Accuracy | **76.91%** |
| | | Test Macro F1 | **0.6109** |
| **Alzheimer's Stage Classification** | PyTorch EfficientNet-B0 | Test Accuracy | **58.83%** |
| | | Test Macro F1 | **0.5793** |
| | | Cohen's Kappa | **0.4511** |
| **Parkinson's Disease Detection** | PyTorch ResNet-18 + Attention-MIL | Test Sensitivity (PD Recall) | **100.00%** |
| | | Test Accuracy (Subject-Level) | **70.00%** |
| | | Binary F1 Score | **0.8235** |
| **Chronic Disease Classification** | Random Forest Classifier | Classification Accuracy | **90.50%** |
| | | Weighted F1 Score | **0.91** |
| **Biomarker Progression** | Multiple Linear Regression | $R^2$ Variance Explained | **0.7971** |
| | | RMSE | **5.9637** |
| **Patient Phenotyping** | K-Means Clustering ($k=3$) | Feature Segmentation | Segmented Risk Cohorts |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for PyTorch deep learning training)

### 1. Clone the Repository
```bash
git clone https://github.com/Divyanshu227/minorproject.git
cd minorproject
```

### 2. Create and Activate Virtual Environment
```bash
# Using conda
conda create -n health-ai python=3.10 -y
conda activate health-ai

# Or using venv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn jupyter pillow
```

---

## 💻 Running the Notebooks

Launch Jupyter Lab / Notebook interface:
```bash
jupyter notebook
```

- **Diabetic Retinopathy**: Open and run `Diabetes-prediction/train.ipynb` to train/evaluate the EfficientNet-B0 model.
- **Clustering & Classification**: Open `Chronic disease prediction models/clustering_classification.ipynb` for K-Means and Random Forest analysis.
- **Progression Regression**: Open `Chronic disease prediction models/mlr_chronic_disease.ipynb` for biomarker score regression modeling.

---

## 🙏 Dataset Acknowledgements & Credits

We express our sincere gratitude to the authors and creators of the open-source datasets utilized in this project:

1. **Alzheimer's Multiclass Dataset (Equal and Augmented)**
   - **Author / Contributor**: **Aryan Singhal** ([@aryansinghal10](https://www.kaggle.com/aryansinghal10))
   - **Kaggle Dataset**: [Alzheimer's Multiclass Dataset: Equal and Augmented](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented)
   - **Description**: Equalized and augmented MRI scan collection enabling balanced multi-class training for Alzheimer's disease diagnosis.

2. **Diabetic Retinopathy 224x224 (2019 Data)**
   - **Author / Contributor**: **Sovit Rath** ([@sovitrath](https://www.kaggle.com/sovitrath))
   - **Kaggle Dataset**: [Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data/)
   - **Description**: High-quality resized $224 \times 224$ retinal fundus photography dataset derived from the APTOS 2019 Blindness Detection challenge.

3. **Parkinson's Disease fMRI Images**
   - **Author / Contributor**: **Salman Eunus** ([@salmaneunus](https://www.kaggle.com/salmaneunus))
   - **Kaggle Dataset**: [Parkinson's Disease fMRI Images](https://www.kaggle.com/datasets/salmaneunus/parkinsons-disease-fmri-images)
   - **Description**: Functional Magnetic Resonance Imaging (fMRI) brain image dataset designed for automated Parkinson's disease detection and classification.

---

## ⚖️ Disclaimer

This repository is developed for academic and research purposes as part of a minor project. The predictive models and outputs provided are not intended to replace professional medical diagnosis, advice, or treatment.
