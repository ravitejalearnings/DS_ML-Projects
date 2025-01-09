## Simple code for solving regression models in one go:

    def function_name(model_name, X_train, X_test, y_train, y_test):
    models = {
        'model_name': model_1,
        'model_name': model_2,
        'model_name': model3,
        'model_name': model_4
    }

        results = {}
    
        for name, model in models:
            model.fit(X_train, y_train)
    
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            trian_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
    
            results[name] = {
                'model_name': print(f"{name}"),
                'train_rmse': print(f"train_rmse: {train_rmse}"),
                'test_rmse': print(f"test_rmse: {test_rmse}"),
                'train_r2': print(f"train_r2: {train_r2}"),
                'test_r2': print(f"test_r2: {test_r2} \n")
            }


## Simple code for solving classification models in one go:

    def functiona_name(model_name, X_train, X_test, y_train, y_test): #input values
    
        models = {
            'model_name': model1,
            'model_name': model2,
            'model_name': model3
        } # list of models trained & evalulated in dict format
    
        results = {} #results stored in dict format
    
        for name, model in models: #loop runs for each model
            model.fit(X_train, y_train) #training each model
    
            y_pred = model.predict(X_test) #prediction
    
            report = classification_report(y_test, y_pred) #classification report for each model
            results[name] = {
                                'classification_report': 
                                    print(f"{name}: {report} \n")
            } #output with model_name & classification report


## Process/steps or project workflow
        load libraries
        load df
        understanding of data at high level use function below:
            shape
            info()
            head()/tail()
            describe()
            unique()/nunique()
            isnull().sum()
            isna().sum()
            isduplicated().sum()
            drop.duplicates(keep='first')
            value_counts()
            understanding of numerical features vs categorical features
        EDA:
            understanding of data
                    any transformations are required within features
                    quality checks within features especially categorical features
                    check for outliers using IQR(popular technique)
            univariant analysis (single feature)
                box plot: mean, median,range, percentile in single go
                distplot: distribution & density
                histplot: bins & distribution
                pie plot: contribution/propotions (%)
            bivariant (2 features)
                scatterplot: how 1 feature influence another features
                boxplot: categorical feature to numerical feature
            multivariant (> 2 features)
                correlation matrix
                heatmaps
        Split of data:
                independent variables (X)
                dependent / target variables (y)
                train test split 
                X_train, X_test, y_train, y_test
        important notes:
                unseen data: X_test, y_test
                training data: X_train, y_train
                always no.of cols of X_train & X_test dataset should remain same
                when outliers are removed from X_train, same index should remain for y_train as well 
                        can be achieved using index.intersection
        feature engineering:
                any new feature added or removed, same to be applied to X_test dataset as well to maintain consistency

        feature selection technique's:
                note: 
                    before applying technique, features needs to be scaled using minmaxscaler() / standardscaler()
                    1st feature reduction technique to be applied then check for multicollinearity
                    order of execution:
                        feature selection > feature selection technique > VIF 
                    
                RFE: feature reduction technique
                VIF: to check multicollinearity
                    applied only to numerical features
                
                encoding:
                    ordinal features: direct mapping using "map" function
                    nominal features with low cardinality: OHE
                    high cardinality: target encoding
        model fit train & evaluate:
                implement Kfold technique for regression or classification problems
                use direct function for regression & classification problems
        hyperparameter tunning:
                use grid_search_CV or randomized search CV
        Best model:
                find out best model, model_coefficents, parameters
        Error Analysis:
                y_test, y_pred

## Understanding of tp,tn,fp,fn
    TP (true postive) : models tries to predict postive class (1) and actual outcome is  positive (1)
    TN (true negative) : models tries to predict negative class (0) and actual outcome is  negative (0)
    FP (false positive) : models tries to predict postive class (1) and actual outcome is  negative (0)
    FN (false negative) : models tries to predict negative class (0) and actual outcome is  positive (1)


## IQR (inter quartile range)
    def get_iqr(df, col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*1qr
        return (lower_bound, upper_bound)

## VIF (variance inflation factor): technique to detect multicollinearity
- Applicable only to numerical features.
- before applying VIF, features needs to be scaled

        def get_vif(data):
            vif_df = pd.DataFrame()
            vif_df['features'] = data.columns
            vif_df['values'] = [variance_inflation_factor(data.values, i )for i in range(data.shape[1])]
            return vif_df

## feature selection technique's
    Filter methods :    based on statistical measures to score features. They are independent of machine learning algorithms
            Correlation-based selection: Remove features with high correlation
            Chi-Square Test: Measures the dependency between categorical features and the target variable
            Mutual Information: Evaluates the dependency between variables
            Variance Threshold: Removes features with low variance.
            ANOVA (Analysis of Variance): Measures the variance between groups to select features.
            Information Gain: Measures the reduction in entropy after splitting on a feature.
            F-Test: Tests if a feature is linearly correlated with the target.
            
    Wrapper methods :    use a predictive model to evaluate feature combinations and iteratively select the best subset.
            Recursive Feature Elimination (RFE): Selects features by recursively removing the least significant features.
            
    Embeded methods :    combine feature selection with model training. They are part of the learning algorithm itself
            LASSO (L1 Regularization): Shrinks coefficients of less important features to zero.
            Ridge Regression (L2 Regularization): Penalizes large coefficients but does not eliminate features.
            Elastic Net: Combines L1 and L2 regularization for feature selection.
            Tree-based Methods: Models like Random Forest, XGBoost, and LightGBM provide feature importance scores.
            Regularized Logistic Regression: Selects features during the training process.
            Gradient Boosting Feature Importance: Uses the importance scores derived from boosting models

## Encoding:
    label encoding: applied to ordinal variables like salary = (low,medium,high)
            salary_mapping = {'low':1, 'medium':2, 'high':3}
            
    Target encoding: applied to high cardinality features like models, brands, company etc
            cols_to_te = ['col1',''col2','col3']
            encoder = TargetEncoder(smoothing=0.3)
            encoder.fit(X_train[cols_to_te], y_train)
            X_train[cols_to_te] = encoder.transform(X_train[cols_to_te])
            X_test[cols_to_te] = encoder.transform(X_test[cols_to_te])
            
    One Hot encoding:
            cols_to_encode = ['col1',''col2','col3']
            df[cols_to_encode] = pd.get_dummies(data= df, columns= cols_to_encode, drop_first=True, dtype='int8') 
        
    
                
            
        

    
        
