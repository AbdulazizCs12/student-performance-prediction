import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# Load dataset
data = pd.read_csv("student_data.csv")

# Features used for both models
features = ["G1", "G2", "studytime", "absences"]


# ==========================================
# 1. LINEAR REGRESSION
# Predict the student's final grade (G3)
# ==========================================

X = data[features]
y = data["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("===== Linear Regression =====")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nModel Coefficients:")

for feature, coefficient in zip(X.columns, linear_model.coef_):
    print(feature, ":", coefficient)

print("Intercept:", linear_model.intercept_)


# ==========================================
# 2. LOGISTIC REGRESSION
# Predict whether a student will Pass or Fail
# ==========================================

# Pass = 1 if G3 >= 10
# Fail = 0 if G3 < 10
data["Pass"] = (data["G3"] >= 10).astype(int)

X_class = data[features]
y_class = data["Pass"]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)

logistic_model = LogisticRegression()

logistic_model.fit(X_train_class, y_train_class)

y_pred_class = logistic_model.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred_class)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_class))

print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))