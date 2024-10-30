# Project Title

## Business Problem
- Briefly describe the business problem your project aims to address.

## Project Objective
- Summarize the main objective of the project.

## Project Deliverables
- Final Model with Streamlit UI

## Technology Stack
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, optuna

## Project Workflow

### 1. Load Libraries
- Load all necessary libraries and dependencies for the project.

### 2. Import Data
- Merge data to get a holistic view.
- Perform a basic understanding of data.
- Split data into training and test sets to avoid data leakage.

### 3. Data Cleaning
- Check for null values.
- Remove duplicate entries.
- Clean text values (extra spaces, case formatting).
- Identify unique values for categorical features.
- Data imputation for missing values.
- Outlier detection:
  - Interquartile Range (IQR)
  - Standard deviation
  - Box plot analysis

### 4. Exploratory Data Analysis (EDA)
- **Distribution analysis**: Histograms to understand distributions.
- **Categorical analysis**: Bar plots for categorical features.
- **Numerical analysis**: Bar plots for numeric features.
- **Category to Category analysis**: KDE and histogram plots.
- **Numeric to Numeric analysis**: Scatter plots.

### 5. Feature Engineering
- **Adding Features**: Based on business logic.
- **Feature Selection**:
  - Correlation analysis.
  - VIF for continuous values.
  - Weight of Evidence / Information values for categorical values.
- **Encoding**:
  - One Hot Encoding for categorical features.

### 6. Model Training
- Experiment with various algorithms using default and custom parameters.

### 7. Model Fine Tuning
- **Hyperparameter Tuning**:
  - Randomized Search CV
  - Grid Search CV
- **Class Imbalance Handling**:
  - Oversampling (SMOTE, SMOTE Tomek)
  - Undersampling (Random Under Sampling)
- **Optimization**: Use Optuna for parameter tuning.

### 8. Model Evaluation
- Evaluate model performance using suitable metrics.

### 9. Build UI Framework Using Streamlit
- Create an interactive UI for the model using Streamlit.

### 10. Model Deployment
- Deploy the model to a production environment.

---

