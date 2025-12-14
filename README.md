# 🚨 Insurance Claim Fraud Detection using Machine Learning

## 📌 Project Overview

Insurance fraud leads to significant financial losses every year.
This project aims to **detect fraudulent insurance claims early in the claim lifecycle** using **machine learning models** and **data-driven insights**.

We build and evaluate **Logistic Regression** and **Random Forest** models to classify claims as **Fraudulent** or **Legitimate**, optimise decision thresholds, and provide actionable business insights.

---

## 🎯 Business Problem

**Global Insure**, a leading insurance provider, processes thousands of claims annually.
Manual fraud detection is:

* Time-consuming
* Inefficient
* Often reactive (fraud detected after payout)

### Objective

Build a predictive model to:

* Identify **fraudulent claims early**
* Reduce **financial losses**
* Optimise **investigation effort**

---

## 🧠 Key Questions Addressed

* How can historical data reveal fraud patterns?
* Which features are most predictive of fraud?
* Can we estimate fraud probability for incoming claims?
* How can model insights improve the fraud detection process?

---

## 📂 Dataset Description

* **Rows:** 1000 insurance claims
* **Columns:** 40 features (customer profile, policy details, incident details, claim amounts)

### Target Variable

* `fraud_reported`

  * `Y` → Fraudulent
  * `N` → Legitimate

---

## 🛠️ Project Workflow

```
├── Data Preparation
├── Data Cleaning & Preprocessing
├── Exploratory Data Analysis (EDA)
├── Feature Engineering
├── Handling Class Imbalance (RandomOverSampler)
├── Model Building
│   ├── Logistic Regression (Statsmodels)
│   └── Random Forest (Sklearn)
├── Feature Selection (RFECV)
├── Model Evaluation
├── Optimal Cutoff Selection
├── Business Insights & Conclusions
```

---

## 🔍 Exploratory Data Analysis (EDA)

* Univariate & bivariate analysis
* Fraud likelihood across:

  * Incident severity
  * Claim amounts
  * Vehicle types
  * Customer hobbies & occupations
* Correlation analysis for multicollinearity
* Class imbalance analysis

---

## 🧪 Feature Engineering

Key engineered features:

* **Incident Day of Week**
* **Claim-to-Deductible Ratio**
* Category consolidation for sparse categorical variables
* One-hot encoding of categorical features
* Feature scaling using `StandardScaler`

---

## ⚖️ Handling Class Imbalance

Applied **RandomOverSampler** to balance fraud and non-fraud classes:

| Class      | Before | After |
| ---------- | ------ | ----- |
| Legitimate | 526    | 526   |
| Fraudulent | 173    | 526   |

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression

* Feature selection using **RFECV**
* Multicollinearity check using **VIF**
* Statistical significance via **p-values**
* Optimal cutoff selection using **Youden’s Index**

### 2️⃣ Random Forest

* Feature importance analysis
* Cross-validation to check overfitting
* Hyperparameter tuning using **GridSearchCV**

---

## 📊 Model Performance (Training)

### Logistic Regression (Optimal Cutoff = 0.40)

| Metric               | Score    |
| -------------------- | -------- |
| Accuracy             | **0.89** |
| Sensitivity (Recall) | **0.93** |
| Specificity          | **0.84** |
| Precision            | **0.85** |
| F1 Score             | **0.89** |
| ROC-AUC              | **0.92** |

---

### Random Forest (Final Model)

| Metric    | Score                               |
| --------- | ----------------------------------- |
| Accuracy  | **~0.83 (Validation)**              |
| Recall    | **~0.84**                           |
| Precision | **Higher than Logistic Regression** |
| F1 Score  | **Best overall**                    |

---

## 📈 Evaluation Visualizations

* ROC Curve
* Precision-Recall Curve
* Sensitivity-Specificity tradeoff
* Cutoff optimization plots
* Confusion matrices

---

## 🧠 Key Insights

* **Incident severity** is the strongest fraud indicator
* Certain **hobbies and vehicle models** have higher fraud likelihood
* High **claim-to-deductible ratios** strongly indicate fraud
* Random Forest captures non-linear fraud patterns better
* Logistic Regression provides superior **explainability**

---

## ✅ Final Recommendation

| Use Case                    | Recommended Model   |
| --------------------------- | ------------------- |
| Regulatory & Explainability | Logistic Regression |
| Production Fraud Detection  | **Random Forest**   |

Deploying this system enables:

* Early fraud detection
* Reduced investigation cost
* Faster claim processing
* Data-driven fraud prevention

---

## 🧰 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Statsmodels**
* **Seaborn, Matplotlib**
* **Imbalanced-learn**

---

## 🚀 Future Improvements

* Cost-sensitive learning
* SHAP for model explainability
* Real-time scoring pipeline
* Ensemble models
* Deployment via REST API

---

## 👤 Author

**Rahul Das**
MSc in Machine Learning & AI
 Liverpool John Moores University
