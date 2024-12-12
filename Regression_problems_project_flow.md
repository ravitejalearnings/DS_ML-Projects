ML Project flow for Regression Problems
Import Libaries
Load data
Shape, info, describe, isnll().sum(), isna().sum(), isduplicated().sum(), columns,
EDA:
- univariant analysis: analyze individual features, understand distribution, Central tendency
- bivariant analysis: explore relationship between 2 features
- categorical to categorical: cross-tab, heatmaps, stacked bars
- categorical to numerical: boxplots,
- numerical to numerical: scatterplots, linepolts
- Correlation: identifying multicollinearity (highly correlated features can be dropped)
- Missing value analysis: heatmaps
- Outlier detection: IQR
- FE: create or remove features, feature transformation(log, sqrt)
- Target: variable analysis: distribution of target variable
Split data:
	Train_test_split()
Imputation:
	SimpleImputer()
	Categorical features with mode()
	Numerical features with median()
Encoding:
	High cardinality features: Target encoding without dataleakage
	Low cardinality nominal features: OHE encoding without dataleakage
	Ordinal features: 
Scaling:
	MinMaxScaler(): for distance based models like K-NN
	StandardScaler(): gradient-based models like linear regression
Model fit & training:
	Model.fit(X_train, y_train)
Kfold CV: to check mode consistency, 
	np.(kfold_score).mean()
	np.(kfold_score).var()
Model evaluation metrics:
	RMSE
	MSE
	R2_score
	MAE
	Adjusted R2
Hyperparameter tunning:
	Gridsearchcv
	Randomizedsearchcv	
Regularization models:
	Lasso
	Ridge
Ensemble models:
	XGB 
	Random Forest regressor
Error Analysis:
	Analyze residuals 
Finalize Best Model:
Export model: joblib, pickle file
