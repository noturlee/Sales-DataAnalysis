import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('/Users/leighchejaikarran/Downloads/SalePredictionModel/advertising.csv.xls')

sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8)
plt.show()

X_rf = df[['TV', 'Radio', 'Newspaper']]  
y_rf = df['Sales']  

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_model,
                              param_grid=param_grid_rf,
                              cv=5,
                              scoring='neg_mean_squared_error',
                              n_jobs=-1)

grid_search_rf.fit(X_train_rf, y_train_rf)

print("Best Parameters (Random Forest):", grid_search_rf.best_params_)
print("Best Negative MSE Score (Random Forest):", grid_search_rf.best_score_)

best_rf_model = grid_search_rf.best_estimator_

y_pred_rf = best_rf_model.predict(X_test_rf)

mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

print("\nEvaluation Metrics on Test Set (Random Forest):")
print("Mean Squared Error (MSE):", mse_rf)
print("Mean Absolute Error (MAE):", mae_rf)
print("R-squared (R2):", r2_rf)

X_lr = df[['TV']]  
y_lr = df['Sales']  

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

lr_model = LinearRegression()

lr_model.fit(X_train_lr, y_train_lr)

print("\nLinear Regression Model Coefficients:", lr_model.coef_)
print("Linear Regression Model Intercept:", lr_model.intercept_)

y_pred_lr = lr_model.predict(X_test_lr)

mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
r2_lr = r2_score(y_test_lr, y_pred_lr)

print("\nEvaluation Metrics on Test Set (Linear Regression):")
print("Mean Squared Error (MSE):", mse_lr)
print("R-squared (R2):", r2_lr)

new_data_lr = [[100]]  
predicted_sales_lr = lr_model.predict(new_data_lr)
print('\nPredicted Sales using Linear Regression:', predicted_sales_lr)

new_data_rf = [[100, 20, 30]]  
predicted_sales_rf = best_rf_model.predict(new_data_rf)
print('Predicted Sales using Random Forest:', predicted_sales_rf)
