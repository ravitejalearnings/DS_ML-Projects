# ML Project Flow for Regression Problems

## Import Libraries

## Load Data
- `shape`
- `info()`
- `describe()`
- `isna().sum()`
- `isduplicated().sum()`
- `columns`
- 'value_counts()'
- 'unique()'
- 'nunique()'

## EDA (Exploratory Data Analysis)
### Univariate Analysis
- Analyze individual features, understand distribution, and central tendency.

### Bivariate Analysis
- Explore relationships between two features:
  - **Categorical to Categorical:** Cross-tabulation, heatmaps, stacked bar plots.
  - **Categorical to Numerical:** Boxplots.
  - **Numerical to Numerical:** Scatterplots, lineplots.

### Multivariate Analysis
- **Correlation:** Identify multicollinearity (highly correlated features can be dropped).
- **Missing Value Analysis:** Visualize using heatmaps.
- **Outlier Detection:** Use IQR.

### Feature Engineering (FE)
- Create or remove features.
- Apply transformations (e.g., log, square root).

### Target Variable Analysis
- Analyze the distribution of the target variable.

## Split Data
- `Train_test_split()`

## Imputation
- Use `SimpleImputer()`:
  - **Categorical Features:** Impute with mode.
  - **Numerical Features:** Impute with median.

## Encoding
- **High Cardinality Features:** Target encoding without data leakage.
- **Low Cardinality Nominal Features:** One-hot encoding (OHE) without data leakage.
- **Ordinal Features:** Ordinal encoding.

## Scaling
- **MinMaxScaler:** For distance-based models like K-NN.
- **StandardScaler:** For gradient-based models like linear regression.

## Model Fit & Training
- `Model.fit(X_train, y_train)`

## K-Fold Cross-Validation
- Check model consistency:
  - `np.mean(kfold_score)`
  - `np.var(kfold_score)`

## Model Evaluation Metrics
- Root Mean Squared Error (RMSE).
- Mean Squared Error (MSE).
- Mean Absolute Error (MAE).
- R² Score.
- Adjusted R² Score.

## Hyperparameter Tuning
- `GridSearchCV`
- `RandomizedSearchCV`

## Regularization Models
- Lasso.
- Ridge.

## Ensemble Models
- XGBoost.
- Random Forest Regressor.

## Error Analysis
- Analyze residuals.

## Finalize Best Model

## Export Model
- Save model using `joblib` or `pickle`.

