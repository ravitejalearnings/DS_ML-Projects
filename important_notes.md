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
