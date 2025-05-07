import pandas as pd
import ast
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

train_data = pd.read_csv("./output_data.csv")

train_data['Data'] = train_data['Data'].apply(ast.literal_eval)

X = pd.DataFrame(train_data['Data'].tolist())
y = train_data['Distance']
y_normalized=y

X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
tau_train, p_value_train = kendalltau(y_train, y_pred_train)
tau_test, p_value_test = kendalltau(y_test, y_pred_test)

print("Training Real：", y_train)
print("Training Pred：", y_pred_train)
print("Training MSE：", mse_train)
print("Training Kendall's Tau:", tau_train)

print("Testing Real：", y_test)
print("Testing Pred：", y_pred_test)
print("Testing MSE：", mse_test)
print("Testing Kendall's Tau:", tau_test)

joblib.dump(poly, 'hand_feature_model.pkl')
joblib.dump(model, 'hand_regression_model.pkl')
