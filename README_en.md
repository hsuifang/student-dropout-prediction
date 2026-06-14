# Student Dropout and Academic Success Prediction System

[👉 中文版 Readme.md](./README.md)

**A Secure Deep Learning System for Predicting Students' Dropout and Academic Success**

This project predicts students' academic outcomes using enrollment, academic, and background data.  
The modeling stage focuses on the most critical **binary classification task (Dropout vs Graduate, with Enrolled samples removed)**, and also covers explainability, security, fairness, and an actionable inference interface.

> **Live Demo**: <https://huggingface.co/spaces/hsuifang/student-dropout-prediction>
>
> Dataset: [UCI – Predict students' dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## 📋 Data Card (Completed by Member A, 113AB8049)

### Data Card

* **Dataset name**:
  Predict Students' Dropout and Academic Success

* **Dataset source**:
  UCI Machine Learning Repository (Dataset ID 697)

* **Original dataset size**:
  * **Instances**: 4,424 students
  * **Features**: 36 raw input variables
  * **Target classes**: 3 classes (Dropout, Enrolled, Graduate)

* **Processed dataset size**:
  * **Features**: 14 final model features (11 selected original features and 3 engineered features)
  * **Target classes**: 2 classes (binary classification: Dropout / Graduate)
  * *Note: For detailed implementation of data cleaning, feature selection, and feature engineering, see [01_data_preprocessing.md](./reports/01_data_preprocessing.md).*

---

* **Feature domains**:
  The original 36 features (reduced to 14 after optimization) mainly cover:
  * Student demographics (e.g., gender, age at enrollment, nationality)
  * Academic background (e.g., admission grade, previous qualification grade)
  * Family and socioeconomic background
  * Academic performance (first- and second-semester course information)
  * Macroeconomic indicators (GDP, inflation rate, unemployment rate)

* **Target label mapping**:
  * **Original labels (3-class)**: `Dropout`, `Enrolled`, `Graduate`
  * **Processed labels (binary)**: `1` represents Dropout, `0` represents Graduate (`Enrolled` samples removed)

---

### Data Preprocessing & Feature Engineering Overview

To improve model accuracy while considering fairness for sensitive attributes, we applied the following optimizations to the raw data **(full technical details and code are provided in 01_data_preprocessing.md)**:

* **Data filtering**:
  Removed `Enrolled` samples, which are not directly aligned with the main prediction goal, and focused on confirmed Graduation and Dropout outcomes.
* **Target encoding**:
  Converted the original text-based `Target` column from a 3-class label into a binary label (`Dropout` mapped to `1`, `Graduate` mapped to `0`).
* **Feature engineering**:
  Based on domain knowledge and correlation screening, we selected 11 key features from the original 36 and engineered 3 high-impact derived features that capture students' learning dynamics (*first-semester pass rate*, *second-semester pass rate*, and *grade change across semesters*), resulting in 14 core model features.
* **Feature scaling**:
  Applied mean-variance normalization (Z-score standardization) to numerical features to help neural networks and linear models converge. We also **strictly separated training and test statistics to prevent data leakage**.
* **Validation setup**:
  Used a strict 80% training / 20% independent test split, and adopted **5-fold stratified cross-validation** during training to preserve the real-world dropout ratio (39.15%) while ensuring generalization.

---

* **Sensitive attributes**:
  * `Tuition fees up to date`

* **Privacy risks**:
  `Tuition fees up to date` reflects a student's financial condition and is sensitive personal information. Although the dataset is de-identified (no names or student IDs), combining this variable with other demographic or academic data may still increase the risk of inferring an individual's economic status. Appropriate access control and protection measures should therefore be applied during data use and sharing.

* **Bias & fairness risks**:
  `Tuition fees up to date` may indirectly reflect socioeconomic status. If the model relies too heavily on this feature, economically disadvantaged students may be systematically predicted as having a higher dropout risk. To mitigate this issue, we introduced a **fairness repair algorithm** in this project. The model's predictions should be used only as a reference for providing additional support and interventions, not as a tool to limit educational opportunities.

* **Intended use**:
  To identify students at risk of dropping out as early as possible, so schools can provide academic guidance, financial aid, and other targeted support.

* **Prohibited use**:
  This model must not be used as the sole basis for:
  * Student dismissal or expulsion decisions
  * Admission decisions
  * Scholarship allocation or disciplinary action
  * Any automated decision affecting student rights without human review

* **Dataset limitations**:
  1. The data comes from a single higher education institution, so the results may not generalize to other schools or countries.
  2. The data reflects a specific educational system and socioeconomic environment, and may not represent other student populations.
  3. Important latent factors affecting dropout, such as mental health, learning motivation, and family support, are not included in the dataset.
  4. The macroeconomic variables are broad indicators and cannot precisely reflect each student's personal financial status.
  5. The inclusion of second-semester academic performance features may reduce the model's timeliness in a truly early-warning setting.

---

## 📋 Model Card (Completed by Member B, 113AB8046)

- **Model name**: Fair-Predict Student Retention Model
- **Model version**: `1.0.0` *(synchronized with `MODEL_VERSION` in `src/schema.py`)*
- **Model architecture**: Deep feedforward neural network (Multi-Layer Perceptron, MLP)
  * **Layer Stack**: `Input (14 features) → Dense(64) + BatchNorm + ReLU + Dropout(0.4) → Dense(32) + BatchNorm + ReLU + Dropout(0.3) → Dense(16) + ReLU → Linear(1)`
  * **Loss Function**: `Focal Loss` (for label imbalance) + `MMD MinDiff Loss` (fairness constraint)
  * **Optimization**: Automated Bayesian hyperparameter tuning with `Optuna` + `5-Fold` cross-validation
- **Training data**:
  Uses de-identified student academic and socioeconomic feature data (`train_scaled.csv`). It contains **14 core features**: 11 original features selected through SHAP analysis, plus 3 manually engineered features from our team (first-semester pass rate, second-semester pass rate, and grade change rate). No SMOTE oversampling was applied to avoid distorting the data distribution.

- **Evaluation results**:
  All models in this project were trained under a strict **5-fold cross-validation** setup, and final predictions on the independent gold test set were integrated via soft voting. The final quantitative results are shown below:

| Model Stage | Accuracy | Macro F1 | Dropout Recall | ROC-AUC (Integrated Test Set) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** (5-Fold Logistic Regression) | 0.94 | 0.93 | 0.90 | 0.9685 |
| **MLP** (5-Fold) | 0.94 | 0.92 | 0.90 | **0.9730** |
| **MLP + MinDiff** (Optuna-tuned) | 0.92 | 0.91 | 0.90 | 0.9485 |

> 💡 **5-Fold internal validation stability**
> * **MLP + MinDiff mean validation AUC**: $0.9268 \pm 0.0146$
> * **MLP + MinDiff mean validation Gap**: $22.67\% \pm 18.32\%$

- **Fairness results**:
  This project uses `Tuition fees up to date` as the **protected sensitive attribute** to assess whether the model introduces serious systematic discrimination against economically disadvantaged students.

  * **Baseline (FPR Gap)**: **62.54%** 🚨 *(Sensitive group FPR: 66.67% / Reference group FPR: 4.13%)*
  * **Pure MLP (FPR Gap)**: **78.52%** 🚨 *(Sensitive group FPR: 83.33% / Reference group FPR: 4.82%)*
  * **MLP + MinDiff (FPR Gap)**: 👑 **8.18%** ✨ *(Sensitive group FPR: 16.67% / Reference group FPR: 8.49%)*

  **Bias mitigation insight:**  
  By maximizing the custom composite metric $\text{Score} = \text{Mean AUC} - \text{Mean FPR Gap}$ with Optuna, the final model successfully **eliminated 86.7% of the historical financial-status bias**, while sacrificing only a very small amount (2.39%) of peak predictive AUC.

- **Intended use**:
  Designed for automated academic review at the end of the first year in colleges and universities. It aims to identify students with high dropout risk so that academic offices, counseling centers, and advisors can use it as an early intervention signal for **care interviews, psychological counseling, and campus resource allocation**.

- **Out-of-scope use**:
  * ❌ The model's predicted labels or probabilities must **never** be used directly for scholarship evaluation, financial aid allocation, or penalties in outstanding student selection.
  * ❌ No campus administrative unit may use this automated model to directly trigger forced withdrawal, grade retention, or punitive administrative action without human review.

- **Model limitations**:
  * **Time lag**: Core features rely heavily on end-of-semester grades and credits, so the model may react too late for sudden dropout events caused by financial emergencies or loss of interest mid-semester.
  * **Binary simplification bias**: The tuition payment status is currently simplified into a binary representation (below median vs above median), which cannot precisely capture dynamic and continuous changes in family finances.

- **Ethical risks**:
  If a model without MinDiff correction (such as the pure MLP) were deployed directly, it would create severe **financial shortcut learning** bias. The system could rely solely on whether a student comes from a financially disadvantaged background or has overdue tuition, which is unrelated to academic ability, and blindly misclassify the student as certain to drop out with a probability as high as 66.67%. This would create structural discrimination in the allocation of campus administrative resources and inflict secondary stigmatizing harm on specific socioeconomic groups.

- **Security risks**: See the Security section below.
- **Human oversight**:
  **Responsible AI core principle (Human-in-the-loop):** The model's predictions and risk probabilities should be treated only as supporting signals for frontline academic counselors. Final intervention decisions, practical response measures, and administrative judgments must remain fully subject to human review and professional advisor assessment.

- **Deployment status**:
  * 🟢 **Status**: `Deployed`
  * **Online demo**: Hugging Face Spaces — <https://huggingface.co/spaces/hsuifang/student-dropout-prediction>
  * The production **5-fold MinDiff ensemble** (`model.pt` + `preprocessor.joblib`) is included in the repo and deployed via Docker (CPU torch, ~1.76GB); no changes are required in the UI or inference layer.
  * 5-fold cross-validation and debiasing evaluation passed (integrated FPR Gap < 10%).

---

## 🔒 Security (Member C, 113AB8050)

Key controls: **Input Validation**, sensitive attribute **de-identification**, shared **preprocessor** across training and inference (to prevent training-serving skew), **Human Review** warning, and de-identified **inference logs**.

> Full risk register (8 risks × description × control × implementation × status) and TODOs are available in [`reports/05_security.md`](reports/05_security.md).

### System Warning (Shown in the UI)

```text
This prediction is intended for early intervention support only.
It must not be used as the sole basis for academic,
disciplinary, or enrollment decisions.
```

## Quick Start · How to Use

### Step 1 — Setup
```bash
python3 -m venv .venv && source .venv/bin/activate   # Virtual environment recommended
pip install -r requirements.txt
```

### Step 2 — Prepare a Model — **Choose One**
Both options produce `models/model.pt` and `models/preprocessor.joblib`, both follow the inference contract, and both can be loaded directly by `load_checkpoint` with **no UI changes required**.

```bash
# A) Placeholder model: no real data needed, fastest way to bring up the UI / demo flow
python -m scripts.train_placeholder

# B) Production model: requires the real data.csv (with Target column; not included in the repo)
python -m scripts.export_model --data notebooks/data.csv
```

| | A. `train_placeholder` | B. `export_model` |
| --- | --- | --- |
| Data | Synthetic / UCI (no `data.csv` required) | Real `data.csv` |
| Model | Placeholder "fake model", predictions are not meaningful | Production model |
| Method | Standard BCE | Focal + MinDiff, **5-fold soft-voting ensemble** (= the one reported in the project report, consistent with Member B's notebook) |
| Use case | Run the UI without data | Formal demo / deployment |

> 💡 The `export_model` pipeline combines the 5 member models through soft voting and serializes them into a single `models/model.pt`;  
> the inference layer (`src/inference`) uses `EnsembleMLP` to average probabilities automatically, so neither the UI nor the inference code needs to change.

### Step 3 — Run the App
```bash
streamlit run app/streamlit_app.py     # Opens http://localhost:8501
```
Operation flow: enter student information in the left panel → click **Run Assessment** → the right panel immediately shows  
**risk level → suggested action → main risk factors → class probabilities**, along with usage limitations and a human-review reminder.

<p align="center">
  <img src="results/main.png" alt="Dropout Risk Prediction interface" width="527">
</p>

### Step 4 — Optional: Generate Global Explanation Plot
```bash
python -m scripts.plot_global_importance       # → results/global_importance.png
```

> ⚠️ The `models/model.pt` generated by option A is a **placeholder** and is only for UI integration and demo purposes.  
> For real performance, use option B (or replace it with the trained weights from Member B). No changes are needed in the UI or inference layer.

---

## Docker Deployment

Deploy the interface to a server using the **production model (not the placeholder)**:

```bash
# 1) Generate the production model on a machine that has the real data.csv
python -m scripts.export_model --data notebooks/data.csv
#    → models/model.pt (5-fold ensemble) + models/preprocessor.joblib

# 2) Build the image
#    If models/ does not contain a real model, the image build falls back to a placeholder automatically
docker build -t dropout-app .

# 3) Run / deploy to the server
docker run -d -p 8501:8501 --name dropout dropout-app
#    → http://<server-address>:8501
```

- The image does **not** include raw or processed student data (`data.csv`, `*_scaled.csv` are excluded by `.dockerignore`).
  Only the trained `model.pt` and `preprocessor.joblib` are packaged, so runtime behavior is equivalent to running the real model locally.
- To persist inference logs to the host machine, mount `-v $(pwd)/results:/app/results`.
- If the model changes, rerun step 1 → `docker build` → `docker run`.

---

## Inference Flow

The interface (`app/`) does not perform machine learning directly. It only collects inputs, orchestrates the flow, and displays outputs. Prediction and explanation are delegated to `src/`:

```text
Form input (11 fields) → ① validate_input → ② predict → ③ explain_record → interpret into plain language → display + log
                         (security)         (inference)   (explain)
```

| Module | Responsibility |
| --- | --- |
| `inference.py` | **How high is the risk?** — preprocessing (11 → 14) + model (`DropoutMLP` / `EnsembleMLP`) → probability / risk level |
| `explain.py` | **Why?** — contribution of each feature to increasing or decreasing dropout risk (SHAP / gradient fallback) |
| `interpret.py` / `security.py` | Convert numerical outputs into human-readable explanations / input validation + de-identified logs |

> 💡 The interface depends only on the two functions `predict()` and `explain_record()`. Even after replacing a single model with a **5-fold ensemble**, `inference / explain / app` required no code changes.

---

## Project Structure

```text
student-dropout-prediction/
├── README.md                # Includes Data Card / Model Card / Security
├── requirements.txt
├── src/                     # Shared modules
│   ├── schema.py            # Single source of truth: feature order, labels, sensitive attributes, version
│   ├── model.py             # MLP definition + checkpoint I/O format
│   ├── preprocessing.py     # Scaler build/load (prevents training-serving skew)
│   ├── inference.py         # Load model + prediction + risk level
│   ├── explain.py           # SHAP / gradient fallback (Explainability)
│   ├── interpret.py         # Convert model output into teacher-readable risk factors / action suggestions
│   └── security.py          # Input Validation / de-identification / Inference Log
├── scripts/
│   ├── train_placeholder.py # Placeholder model (synthetic data)
│   ├── export_model.py      # Production model (real data, 5-fold MinDiff ensemble)
│   └── plot_global_importance.py # Global explanation plot -> results/global_importance.png
├── app/
│   └── streamlit_app.py     # Inference interface (main deliverable by Member C)
├── models/                  # model.pt / preprocessor.joblib (generated artifacts)
├── results/                 # Inference records, evaluation results
├── notebooks/               # Exploratory analysis
├── reports/                 # Detailed reports for each stage (preprocessing / evaluation / fairness / explainability / security)
└── docs/                    # Assignment requirements and reference documents
```

---

## Model Contract (For Member B to Replace the Production Model)

As long as the following two conditions are met, `models/model.pt` can be replaced directly without changing the frontend:

1. **Feature order and labels** must come from `src/schema.py` (`FEATURE_ORDER`, `LABELS`).
2. **Checkpoint format** must use the save functions in `src/model.py`, and the corresponding `models/preprocessor.joblib` must be exported together:
   - Single model via `save_checkpoint(...)`:
     `state_dict / input_dim / dropout_1 / dropout_2 / num_classes / model_version / label_names`
   - 5-fold ensemble via `save_ensemble_checkpoint(...)`:
     `ensemble (member state_dicts) / num_members / input_dim / dropout_1 / dropout_2 / num_classes / model_version / label_names`
   - `load_checkpoint(...)` automatically detects both formats, so the frontend and inference layer require no changes.

---

## Team Responsibilities

| Module | Owner |
| --- | --- |
| Data preprocessing, Data Card, fairness data | 113AB8049 |
| Baseline, MLP, MinDiff, evaluation, Model Card | 113AB8046 |
| Explainability, Security, Inference interface, Deployment | 113AB8050 |

---

## Detailed Reports

Below is the summary of the key sections (Data Card / Model Card / Security). For full methods and results, see [`reports/`](reports/README.md):

| Report | Content |
| --- | --- |
| [01 Preprocessing](reports/01_data_preprocessing.md) | Data analysis, encoding, splitting, Data Leakage |
| [02 Model & Evaluation](reports/02_model_evaluation.md) | Baseline / MLP / MinDiff, metric comparison |
| [03 Fairness](reports/03_fairness.md) | Group metrics, before/after MinDiff comparison |
| [04 Explainability](reports/04_explainability.md) | SHAP global / local explanations |
| [05 Security](reports/05_security.md) | Full risk register and control details |
