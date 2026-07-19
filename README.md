# Loan Approval Prediction Using Gaussian Naive Bayes

## Project Overview

This project implements a **Gaussian Naive Bayes (GNB)** model to predict whether a loan application will be **Approved** or **Rejected** based on applicant financial and demographic information.

The project demonstrates an end-to-end machine learning workflow, including data preprocessing, categorical encoding, handling class imbalance using SMOTE, feature scaling, model training, and evaluation.

---

## Dataset

**Dataset:** Loan Approval Prediction Dataset (Kaggle)

- Samples: 381
- Features: 12
- Target: Loan_Status

Target Classes:

- 0 → Loan Rejected
- 1 → Loan Approved

The dataset contains applicant information such as:

- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

---

## Project Workflow

- Data Loading and Exploration
- Missing Value Handling
- Duplicate Record Checking
- Exploratory Data Analysis (EDA)
- Categorical Feature Encoding using LabelEncoder
- Train-Test Split
- Handling Class Imbalance using SMOTE
- Feature Scaling using StandardScaler
- Training Gaussian Naive Bayes Classifier
- Model Evaluation
- Prediction on New Loan Applications

---

## Handling Class Imbalance

The target variable was imbalanced:

| Loan Status | Count |
|-------------|------:|
| Approved (1) | 271 |
| Rejected (0) | 110 |

To improve minority class prediction, **SMOTE (Synthetic Minority Oversampling Technique)** was applied only on the training data.

Before SMOTE:

- Approved: 217
- Rejected: 87

After SMOTE:

- Approved: 217
- Rejected: 217

---

## Model Implementation

**Algorithm Used:**

Gaussian Naive Bayes

Gaussian Naive Bayes was selected because the dataset contains continuous numerical features such as:

- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term

The algorithm assumes numerical features follow a Gaussian distribution and calculates the probability of each class using Bayes' theorem.

---

## Model Performance

**Classification Report**

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Rejected (0) | 0.67 | 0.55 | 0.60 |
| Approved (1) | 0.83 | 0.89 | 0.86 |


**Overall Accuracy:**

79%


**Weighted F1-score:**

79%

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Gaussian Naive Bayes

---

## Key Concepts Covered

- Bayes Theorem
- Conditional Probability
- Prior and Posterior Probability
- Gaussian Distribution Assumption
- Naive Bayes Classification
- Label Encoding
- Feature Scaling
- Class Imbalance Handling
- SMOTE
- Classification Metrics
- Confusion Matrix
- ROC-AUC Evaluation

---

## Conclusion

The Gaussian Naive Bayes model successfully learned patterns from applicant financial and demographic information to predict loan approval outcomes.

SMOTE improved the model's ability to identify rejected loan applications, making the classifier more balanced compared to training on the original imbalanced dataset.

This project demonstrates the practical application of Naive Bayes for binary classification problems in the banking and financial analytics domain.
