# Car Price Prediction Project

## Problem Statement
A XYZ automobile company is planning to enter the Indian market by setting up their manufacturing unit locally. To understand the pricing dynamics of cars in the Indian market, they have contracted an automobile consulting company. The goals of this project are:

- To identify the significant factors affecting car prices in the Indian market.
- To evaluate how well these factors predict the price of a car.

---

## Project Workflow

### a. Load Required Libraries
Import all necessary libraries for data manipulation, visualization, and modeling.

### b. Understand Basic Elements of Data
- **Shape:** Analyze the shape of the dataset.
- **Info():** Understand the structure and types of data.
- **isnull().sum():** Check for missing values.
- **Describe():** Generate summary statistics for numerical data.
- **isduplicated()/duplicated().sum():** Check for duplicate records.
- **drop_duplicates(keep='first'):** Remove duplicate rows.
- **select_dtypes(include='object') / select_dtypes(exclude='object'):** Separate categorical and numerical features.
- **value_counts():** Analyze frequency distribution of categorical variables.
- **unique() / nunique():** Get unique values and counts for each feature.

### c. Perform Exploratory Data Analysis (EDA)
#### i. Univariate Analysis
- Use plots like **histplot**, **boxplot**, **kdeplot**, **countplot** to analyze:
  - **Numerical Features**
  - **Categorical Features**

#### ii. Multivariate Analysis
- Analyze relationships between variables using:
  - **pairplot**
  - **scatterplot**
  - **heatmap**
  - **corr()**
  - **crosstab()**

#### iii. Outlier Treatment
- Use **IQR (Interquartile Range)** to identify and handle outliers.

### d. Data Split
- Perform **train-test split** to prepare data for modeling.

### e. Feature Engineering
#### i. Numerical Binning
- Segment continuous variables into discrete bins.

#### ii. Feature Addition/Deletion
- Add or remove features based on data insights.
  - **Note:** Document any significant changes.

#### iii. Encoding
- **One-Hot Encoding (OHE):** For low cardinality categorical variables.
- **Target Encoding:** For high cardinality categorical variables.
- **Label Encoding:** For ordinal variables.

#### iv. Scaling
- **MinMaxScaler:** For non-Gaussian distributions (scales values to [0,1]).
- **StandardScaler:** For Gaussian distributions (scales to mean = 0, std = 1).

#### v. Feature Selection
- **RFE (Recursive Feature Elimination):** For selecting the most important features.
- **VIF (Variance Inflation Factor):** To check for multicollinearity.

### f. Model Training & Evaluation
#### i. Train & Test Models
- Train and test various machine learning models.

#### ii. Model Evaluation
- Compare performance across different models.

#### iii. Hyperparameter Tuning
- Optimize model parameters for better performance.

#### iv. Best Model
- Select and finalize the best-performing model.

#### v. Export Model
- Save the trained model for deployment.

