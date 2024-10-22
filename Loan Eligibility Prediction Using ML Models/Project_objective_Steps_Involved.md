# Loan Approval Prediction

## Problem Statement
The company seeks to automate (in real time) the loan qualifying procedure based on information provided by customers while filling out an online application form. The goal is to develop machine learning models that can help predict loan approval, accelerating the decision-making process for determining whether an applicant is eligible for a loan or not.

## Business Objective
The objective of this project is to automate the loan approval process by developing a real-time machine learning model. The model will analyze customer-provided information during the loan application process and predict loan eligibility. This automation will allow the company to make faster, more accurate, and data-driven decisions, improving operational efficiency and scalability.

## Technology Stack
- **Database:** Structured datasets
- **Libraries:**
  - Pandas
  - NumPy
  - Seaborn
  - Matplotlib
  - Scikit-Learn

## Project Flow

### 1. Import Libraries
- Import all necessary libraries for data manipulation, visualization, and model building.

### 2. Load Data
- Load the dataset and conduct basic data understanding, including data types, missing values, and summary statistics.

### 3. Exploratory Data Analysis (EDA)
- **Categorical Analysis:** Analyze categorical variables to understand their distribution.
- **Numerical Analysis:** Perform analysis on numerical variables to check their spread, skewness, and distribution.
- **Categorical to Categorical Analysis:** Explore relationships between categorical variables.

### 4. Data Preprocessing
- **Data Imputation:** Handle missing values.
  - Imputation for numerical features.
  - Imputation for categorical features.
- **Outlier Treatment:** Identify and treat outliers.
- **Encoding:** Convert categorical variables into numerical formats.
  - Label Encoding.
  - One-Hot Encoding.

### 5. Feature Selection
- **Variance Inflation Factor (VIF):** Analyze multicollinearity and select important features.

### 6. Check for Class Imbalance
- Use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.

### 7. Data Scaling
- Apply scaling techniques to normalize the data.
  - Min-Max Scaler.
  - Standard Scaler.

### 8. Splitting Dataset
- Split the dataset into training and testing sets.

### 9. Model Training
- Train various machine learning models to predict loan approval.

### 10. Model Evaluation
- Evaluate model performance using various metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

### 11. Hyperparameter Tuning
- Optimize models using:
  - **Grid Search CV**
  - **Randomized Search CV**

### 12. Finalizing the Best Model
- Select the best performing model based on evaluation metrics and hyperparameter tuning.

### 13. Export Model
- Save the final model using serialization techniques such as **joblib** or **pickle**.

### 14. Model Deployment
- Deploy the model to production using web frameworks (e.g., Streamlit, Flask, etc.) for real-time predictions.

