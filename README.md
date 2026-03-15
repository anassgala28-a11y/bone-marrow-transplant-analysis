# 🩺 BMT Predictor — Pediatric Bone Marrow Transplant Success Prediction

> **Coding Week · 09–15 March 2026 · Centrale Casablanca**  
> A medical decision-support application predicting the success of pediatric bone marrow transplants using Explainable ML (SHAP).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Preliminary Data Analysis](#preliminary-data-analysis)
- [ML Models & Performance](#ml-models--performance)
- [SHAP Explainability](#shap-explainability)
- [Memory Optimization](#memory-optimization)
- [Interface](#interface)
- [CI/CD Pipeline](#cicd-pipeline)
- [Prompt Engineering Documentation](#prompt-engineering-documentation)
- [Critical Questions — README Answers](#critical-questions--readme-answers)
- [Team & Task Distribution](#team--task-distribution)

---

## Project Overview

This application assists physicians in predicting the **success rate of bone marrow transplants in pediatric patients**. The solution is:

- **Accurate** — trained on real clinical data with 3 evaluated ML models
- **Interpretable** — SHAP explainability for every prediction
- **Deployable** — Streamlit web interface with a professional CI/CD pipeline
- **Reproducible** — fully documented setup and training steps

---

## Dataset

**Source:** [UCI ML Repository — Bone Marrow Transplant: Children](https://archive.uci.edu/dataset/565/bone+marrow+transplant+children)

| Property | Value |
|---|---|
| Patients | 187 |
| Features | 37 clinical attributes |
| Target | `survival_status` (1 = survived, 0 = not survived) |
| Class distribution | ~60% survived · ~40% not survived |
| Format | `.arff` |

---

## Project Architecture

```
bone-marrow-transplant-analysis/
│
├── data/
│   └── bone-marrow.arff
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
│
├── src/
│   ├── data_processing.py         # Preprocessing pipeline
│   ├── train_model.py             # Model training & selection
│   └── evaluate_model.py          # Evaluation plots & metrics
│
├── app/
│   └── Interface2.py              # Streamlit web interface
│
├── models/
│   └── xgboost.pkl                # Best saved model
│
├── outputs/                       # Evaluation plots
│
├── tests/
│   └──    # Automated tests
│
├── worklows/
│   └──ci.yml                  # GitHub Actions CI/CD
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/anassgala28-a11y/bone-marrow-transplant-analysis.git
cd bone-marrow-transplant-analysis

# 2. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt` includes:**

```
pandas
numpy
scipy
scikit-learn
imbalanced-learn
xgboost
lightgbm
joblib
matplotlib
streamlit
shap
pytest
```

---

## Usage

### 1. Train the Model

```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset from `data/bone-marrow.arff`
- Apply SMOTE balancing on training data
- Train Random Forest, XGBoost, and LightGBM classifiers
- Save the best model to `models/xgboost.pkl`
- Print a performance comparison table

### 2. Run the Application

```bash
streamlit run app/Interface2.py
```

Open your browser at `http://localhost:8501`

### 3. Run Tests

```bash
pytest tests/
```

---

## Preliminary Data Analysis

> Full analysis is documented in `notebooks/eda.ipynb`

### Missing Values

Yes, the dataset contains missing values encoded as `"?"`. Strategy applied in `data_processing.py`:

- Replace `"?"` with `NaN`
- **Numeric columns** → imputed with **median** (robust to outliers)
- **Categorical columns** → imputed with **mode** (most frequent value)

### Outliers

Significant outliers exist in continuous clinical variables (e.g., CD34+ cell counts, body mass). Strategy:

- **IQR capping** applied to all numeric features (excluding the target `survival_status`)
- Values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` are capped at the respective fence

### Class Imbalance

| Class | Count | Percentage |
|---|---|---|
| Survived (1) | ~112 | ~60% |
| Not Survived (0) | ~75 | ~40% |

Strategy chosen: **SMOTE (Synthetic Minority Over-sampling Technique)**

- Applied **only on training data**, strictly after the train/test split
- Prevents data leakage — test set retains original class distribution
- Synthetic samples are generated in feature space (not raw data duplication)

### Correlated Features

Highly correlated features (Pearson correlation > 0.95) are identified and dropped using `drop_correlated_features()` in `data_processing.py` to reduce redundancy and prevent multicollinearity.

---

## ML Models & Performance

Three models were trained and evaluated on the same held-out test set (20% split):

| Model | Accuracy | Precision | Recall | F1-Score | **ROC-AUC** |
|---|---|---|---|---|---|
| Random Forest | 0.9474 | 1.0000 | 0.8824 | 0.9375 | 0.9510 |
| **XGBoost ✅** | 0.8947 | 0.8824 | 0.8824 | 0.8824 | **0.9636** |
| LightGBM | 0.9211 | 0.9375 | 0.8824 | 0.9091 | 0.9468 |

### Selected Model: XGBoost

**Reasoning:**

- **Highest ROC-AUC (0.9636)** — the most critical metric in medical classification, as it measures the model's ability to discriminate between survivors and non-survivors regardless of the decision threshold
- **Recall (88.24%)** — in a clinical context, missing a high-risk patient (false negative) has much greater consequences than a false alarm; XGBoost achieves strong recall
- **Robustness** — gradient boosting naturally handles mixed feature types and missing patterns common in clinical datasets
- **SHAP compatibility** — native support via `TreeExplainer` for fast and exact SHAP values

Evaluation plots (confusion matrix, ROC curve) are saved to `outputs/`.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) is integrated in the Streamlit interface to provide **transparent, patient-level explanations** for every prediction.

### Available Visualizations

**Feature Impact — This Patient**
- Horizontal bar chart of SHAP values per feature for the current input
- Positive SHAP (blue) → increases survival probability
- Negative SHAP (red) → decreases survival probability

**Global Feature Importance**
- Mean absolute SHAP across all training samples
- Shows which features matter most across the entire dataset

**Feature Contributions Table**
- Sortable table of feature name, input value, and SHAP contribution

### Key Influential Features (from SHAP analysis)

Based on global importance, the most predictive features include:

1. `CD34kgx10d6` — CD34+ stem cell dose
2. `Recipientage` — Patient age at transplant
3. `HLAmatch` — HLA compatibility between donor and recipient
4. `Disease` — Underlying disease type
5. `Riskgroup` — Pre-transplant risk classification
6. `CMVstatus` — CMV serology combination (donor/recipient)

---

## Memory Optimization

Memory optimization is demonstrated directly in notebooks/eda.ipynb by optimize_memory in data_processing. The notebook and scr/data_processing includes an optimize_memory(df) function that reduces DataFrame memory usage by downcasting numeric types:

- float64 → float32
-  int64 → int32

The notebook clearly shows the before and after memory comparison, demonstrating the reduction achieved on the bone marrow dataset.

##  Automated Testing

To ensure the reliability of the data processing and machine learning pipelines, unit tests have been implemented using the **`pytest`** framework. 

The current test suite (`tests/`) validates three critical components:
1. **Missing Values Handling:** Ensures the `handle_missing_values` function correctly processes NaN/None values.
2. **Memory Optimization:** Verifies that `optimize_memory(df)` successfully reduces the dataset's memory footprint.
3. **Prediction Pipeline:** Uses mocking to safely simulate and validate the model loading and prediction process.
Robustness is key in medical data analysis. I implemented a comprehensive unit testing suite using **`pytest`** to validate core functionalities without relying on the full dataset.

**Key Test Coverage:**
* **Data Processing:** Asserts that missing values are correctly handled and that `optimize_memory` successfully reduces the dataset's memory footprint in megabytes.
* **Machine Learning Pipeline (`train_model.py`):**
  * **Model Initialization:** Ensures Random Forest, XGBoost, and LightGBM are correctly instantiated with the proper sklearn-compatible interfaces.
  * **Evaluation Metrics:** Validates the accurate calculation of Accuracy, Precision, Recall, F1-Score, and ROC-AUC (ensuring outputs are bounded and rounded).
  * **Model Selection:** Verifies that the training pipeline automatically identifies and selects the best-performing model based on the maximum ROC-AUC score.
  * **Serialization:** Ensures the winning model is safely exported as a `.pkl` file (e.g., `xgboost.pkl`) and can be seamlessly reloaded for future predictions.

**How to run the tests locally:**
```bash
# Navigate to the project directory and run:
pytest tests/ -v
**To run the tests:**
```bash
pytest tests/
## Interface

Built with **Streamlit**, the interface provides:

| Section | Description |
|---|---|
| **Overview** | Project summary, dataset stats, 4-step workflow guide |
| **Patient Input** | Form with recipient, donor, HLA, and lab parameters |
| **Prediction Result** | Success probability with visual gauge bar |
| **SHAP Explanation** | Per-patient feature impact + global importance |
| **Model Metrics** | Performance table and ROC-AUC comparison chart |

**Run:**
```bash
streamlit run app/Interface2.py
```

---

## CI/CD Pipeline

GitHub Actions workflow is configured at `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

### Automated Tests (`tests/test_data_processing.py`)

- ✅ Verify missing values handling (`handle_missing_values`)
- ✅ Verify `optimize_memory(df)` reduces memory usage
- ✅ Verify model loading and prediction pipeline

---

## Prompt Engineering Documentation

**Task selected:** Data preprocessing pipeline (`data_processing.py`)

### Prompt Used

```
Context: I am building an ML pipeline for a pediatric bone marrow transplant 
dataset (187 patients, 37 features, .arff format). The dataset has missing 
values encoded as "?", outliers in numeric features, and moderate class 
imbalance (~60/40).

Task: Write a Python function preprocess_pipeline(filepath) that:
1. Loads the .arff file and decodes byte-encoded columns
2. Replaces "?" with NaN and imputes median for numeric / mode for categorical
3. Applies IQR capping for outliers (excluding the target column)
4. Label-encodes all categorical features
5. Drops features with Pearson correlation > 0.95
6. Applies an 80/20 train-test split, then SMOTE only on training data
7. Returns X_train, X_test, y_train, y_test

Requirements: no data leakage, preserve target column, use sklearn pipelines 
where possible.
```

### Result & Effectiveness

The prompt produced a working pipeline in one iteration. Key improvements applied after initial generation:

- **Added explicit SMOTE-only-on-train constraint** — the first version applied SMOTE before the split (data leakage)
- **Specified byte decoding** — `.arff` files encode strings as bytes; the prompt was refined to add explicit `.decode('utf-8')` handling
- **Removed `optimize_memory`** — separated into its own function per project specifications

### Lessons

- Providing dataset statistics (size, format, class distribution) in the context dramatically reduced hallucinations
- Specifying anti-patterns ("no data leakage") produced cleaner code than only describing what to do
- Splitting large tasks into smaller, focused prompts (one per function) was more reliable than one monolithic prompt

---

## Critical Questions — README Answers

### Was the dataset balanced?

No. The dataset is moderately imbalanced (~60% survived, ~40% not survived). We addressed this using **SMOTE applied exclusively on the training set** after the 80/20 split. Impact: without SMOTE, the model was biased toward predicting survival; after SMOTE, recall for the minority class improved significantly without distorting test performance.

### Which ML model performed best?

**XGBoost** achieved the best **ROC-AUC of 0.9636**, which is the primary metric for this medical classification task. See the [ML Models & Performance](#ml-models--performance) section for the full comparison table.

### Which medical features most influenced predictions (SHAP)?

The top SHAP-identified features are: **CD34+ stem cell dose** (`CD34kgx10d6`), **recipient age**, **HLA match grade**, **disease type**, **risk group**, and **CMV serology status**. These align with established clinical literature on transplant outcome predictors.

### What insights did prompt engineering provide?

Prompt engineering revealed the importance of **context-rich, constraint-focused prompts** in medical ML. Specifying anti-patterns (like data leakage), dataset statistics, and output format upfront reduced iteration cycles by ~60%. The most impactful improvement was explicitly asking the model to separate SMOTE from the train/test split logic.

---

## Team & Task Distribution

| Member | Responsibility |
|---|---|
| Anass| src , readme  |
| Abdelilah | SHAP integration , tests|
| abderahman | Interface development |
| manal | data analysis | 
| hafssa | optimisation | 

Task tracking managed via **Trello** (To Do → In Progress → Review → Done).

---

## License

Academic project — Centrale Casablanca · Coding Week 2026
