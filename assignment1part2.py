import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score

# Load dataset
dataset = pd.read_csv("house_price_dataset.csv")

# Independent Variables (Multiple Features)
X = dataset[['Square_Feet', 'Num_Bedrooms', 'House_Age']].values
Y = dataset['House_Price'].values

# Split dataset into Training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

###  Sklearn Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred_sklearn = model.predict(X_test)

# Performance Metrics
rmse_sklearn = np.sqrt(mean_squared_error(Y_test, Y_pred_sklearn))
r2_sklearn = r2_score(Y_test, Y_pred_sklearn)

print(" Sklearn Multiple Linear Regression Results:")
print("Coefficients (Slopes):", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error (RMSE):", rmse_sklearn)
print("Model Accuracy (R² Score):", r2_sklearn)


def gradient_descent(X, Y, learning_rate=0.01, epochs=1000):
    m = np.zeros(X.shape[1])  # Initialize slopes for multiple variables
    c = 0  # Initialize intercept
    n = len(Y)  
    cost_history = []  # Store cost function values

    for _ in range(epochs):
        Y_pred = np.dot(X, m) + c
        error = Y - Y_pred

        # Compute gradients
        dm = (-2/n) * np.dot(X.T, error)
        dc = (-2/n) * np.sum(error)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Compute cost function (Mean Squared Error)
        cost = np.mean(error**2)
        cost_history.append(cost)

    print(f"Total Epochs Run: {epochs}")  # Print the number of epochs

    return m, c, cost_history


# Normalize data before applying Gradient Descent
X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Train using Gradient Descent
m_gd, c_gd, cost_history = gradient_descent(X_train_norm, Y_train)

# Predictions using Gradient Descent
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y_pred_gd = np.dot(X_norm, m_gd) + c_gd

# Performance Metrics for Gradient Descent
rmse_gd = np.sqrt(mean_squared_error(Y, Y_pred_gd))
r2_gd = r2_score(Y, Y_pred_gd)

print("\n Gradient Descent Multiple Linear Regression Results:")
print("Gradient Descent Coefficients (Slopes):", m_gd)
print("Gradient Descent Intercept:", c_gd)
print("Root Mean Squared Error (RMSE):", rmse_gd)
print("Model Accuracy (R² Score):", r2_gd)


#  Precision, Recall & F1-Score Calculation
def calculate_metrics(y_actual, y_pred, threshold=0.5):
    """ Convert Regression Output to Classification Labels """
    y_pred_labels = (y_pred >= threshold * max(y_actual)).astype(int)
    y_actual_labels = (y_actual >= threshold * max(y_actual)).astype(int)

    precision = precision_score(y_actual_labels, y_pred_labels)
    recall = recall_score(y_actual_labels, y_pred_labels)
    f1 = f1_score(y_actual_labels, y_pred_labels)

    return precision, recall, f1

# Calculate metrics for both models
precision_sklearn, recall_sklearn, f1_sklearn = calculate_metrics(Y_test, Y_pred_sklearn)
precision_gd, recall_gd, f1_gd = calculate_metrics(Y, Y_pred_gd)

print("\n Classification Metrics (Thresholded)")
print(" Sklearn Model: Precision:", precision_sklearn, "Recall:", recall_sklearn, "F1-Score:", f1_sklearn)
print(" Gradient Descent Model: Precision:", precision_gd, "Recall:", recall_gd, "F1-Score:", f1_gd)


### Visualization - Actual vs Predicted
plt.figure(figsize=(16, 6))

#  Sklearn Model Visualization
plt.subplot(2, 3, 1)
plt.scatter(Y_train, model.predict(X_train), color="blue", label="Training Data Predictions")
plt.scatter(Y_test, Y_pred_sklearn, color="red", label="Testing Data Predictions")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='black', linewidth=2)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Sklearn Multiple Linear Regression")
plt.legend()

#  Gradient Descent Model Visualization
plt.subplot(2, 3, 2)
plt.scatter(Y_train, np.dot(X_train_norm, m_gd) + c_gd, color="blue", label="Training Data Predictions")
plt.scatter(Y_test, np.dot((X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0), m_gd) + c_gd, 
            color="red", label="Testing Data Predictions")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='black', linestyle="dashed", linewidth=2)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Gradient Descent Multiple Linear Regression")
plt.legend()

plt.subplot(2, 3, 3)  
plt.plot(range(len(cost_history)), cost_history, color="purple", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Cost Function (MSE)")
plt.title("Gradient Descent Cost Function")

# Residual Errors
residuals_sklearn = Y_test - Y_pred_sklearn

# Residuals Plot for Sklearn Model
plt.subplot(2, 3, 4)
plt.scatter(Y_test, residuals_sklearn, color="red", alpha=0.6)
plt.axhline(y=0, color='black', linestyle='dashed')
plt.xlabel("Actual House Price")
plt.ylabel("Residual Error")
plt.title("Sklearn Model: Residual Errors")
residuals_gd = Y_test - np.dot((X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0), m_gd) - c_gd

# Residuals Plot for Gradient Descent Model
plt.subplot(2, 3, 5)
plt.scatter(Y_test, residuals_gd, color="purple", alpha=0.6)
plt.axhline(y=0, color='black', linestyle='dashed')
plt.xlabel("Actual House Price")
plt.ylabel("Residual Error")
plt.title("Gradient Descent Model: Residual Errors")

plt.tight_layout()
plt.show()
