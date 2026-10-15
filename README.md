# DLS_Credit_Scoring

# Feature Engineering & Data Preprocessing Pipeline

## 📌 Overview
This repository contains a comprehensive guide and code implementation for feature engineering techniques applied to a credit scoring dataset (binary classification task: predicting if a client will choose our bank). The notebook demonstrates robust strategies for handling missing values, categorical encoding, and feature generation using Python, Pandas, and Scikit-Learn.

## 🚀 Key Topics Covered

### 1. Handling Missing Values
- **Analysis:** Identifying gaps in both training and test sets to prevent data leakage.
- **Strategies:**
  - **Statistical Imputation:** Using `SimpleImputer` with median (robust to outliers) or constant values based on domain knowledge.
  - **KNN Imputation:** Using `KNNImputer` to fill missing numerical values based on the similarity of other features (Euclidean distance).
  - **Best Practice:** All imputers are `.fit()` on training data and `.transform()` on test data to simulate production environments.

### 2. Categorical Feature Encoding
- **Label Encoding vs. One-Hot Encoding:** Discussion on why simple integer mapping creates false hierarchies in nominal data.
- **One-Hot Encoding (OHE):**
  - Implementation using `sklearn.preprocessing.OneHotEncoder`.
  - Handling binary features with `drop='first'` to avoid multicollinearity.
  - Robustness against unseen categories in test data (mapped to all zeros).
- **High Cardinality Features:** Strategies for features like "Living Region" (grouping rare classes, avoiding sparse matrices).

### 3. Feature Generation & Selection
- **Domain-Specific Features:** Creating meaningful ratios (e.g., monthly payment load, income relative to regional average).
- **Polynomial Features:** Generating interaction terms using `PolynomialFeatures`.
- **Feature Selection:** Using weak models (Linear/Logistic Regression) or tree-based importance (`SelectFromModel`) to filter out noise before training complex algorithms.

## 🛠️ Tech Stack
- **Python**
- **Pandas & NumPy:** Data manipulation
- **Scikit-Learn:** `SimpleImputer`, `KNNImputer`, `OneHotEncoder`, `PolynomialFeatures`, `SelectFromModel`
- **Matplotlib/Seaborn:** Visualization (optional)

## 📊 Dataset Description
The dataset contains client information including:
- **Demographics:** Gender, Age, Marital Status, Education.
- **Financials:** Monthly Income, Credit Amount, Credit Term, Existing Credits.
- **Target Variable:** `OpenAccFlag` (1 if the client chose our bank, 0 otherwise).

## 💡 Key Takeaways for ML Engineers
1. **Never fit on Test Data:** Always calculate statistics (mean, median, encoders) on the training set only.
2. **Understand Your Features:** Don't blindly apply OHE to high-cardinality columns; consider grouping or target encoding.
3. **Quality over Quantity:** Generating thousands of polynomial features without selection leads to overfitting. Use feature importance metrics to prune the feature space.

## 📝 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn jupyter

Will be continued...
