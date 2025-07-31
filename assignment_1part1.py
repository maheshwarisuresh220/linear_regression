import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
import pandas as pd

# Load dataset
dataset = pd.read_csv("house_price_dataset.csv")

X = dataset.Square_Feet.values.reshape(-1,1)
Y = dataset.House_Price

# First split: 60% train, 40% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
# Second split: 20% test, 20% validation
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Train model using Sklearn
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred_sklearn = model.predict(X_test)

# Print model parameters
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# RMSE
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_sklearn))
print("Root Mean Squared Error (RMSE):", rmse)

# R² Score
r2 = r2_score(Y_test, Y_pred_sklearn)
print("Model Accuracy (R² Score):", r2)

# Gradient Descent Implementation
def gradient_descent(X, Y, learning_rate=0.00000001, epochs=1000):
    m, c = 0, 0  # Initialize slope and intercept
    n = len(Y)  # Number of data points
    cost_history = []  # Track cost function

    for epoch in range(epochs):
        Y_pred = m * X + c
        error = Y - Y_pred.flatten()

        # Compute gradients
        dm = (-2/n) * np.sum(X.flatten() * error)
        dc = (-2/n) * np.sum(error)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Compute cost function (Mean Squared Error)
        cost = np.mean(error**2)
        cost_history.append(cost)

        # Print cost at intervals
        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    print(f"Total Epochs Run: {epochs}")

    return m, c, cost_history

# Train using gradient descent
epochs = 1000
m_gd, c_gd, cost_history = gradient_descent(X_train, Y_train, epochs=epochs)

print("Gradient Descent Coefficient (Slope):", m_gd)
print("Gradient Descent Intercept:", c_gd)

# Gradient Descent Predictions
Y_pred_gd = (m_gd * X_test) + c_gd

# Convert regression output to binary classification for precision, recall, and F1-score
threshold = np.median(Y)  # Median price as threshold
Y_test_class = (Y_test >= threshold).astype(int)
Y_pred_class = (Y_pred_sklearn >= threshold).astype(int)

# Precision, Recall, F1-Score
precision = precision_score(Y_test_class, Y_pred_class)
recall = recall_score(Y_test_class, Y_pred_class)
f1 = f1_score(Y_test_class, Y_pred_class)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Sklearn Model Visualization
plt.figure(figsize=(16, 6))

plt.subplot(2, 3, 1)  # First subplot for Sklearn Model
plt.scatter(X_train, Y_train, color="blue", label="Training Data")  
plt.scatter(X_test, Y_test, color="red", label="Testing Data")  
plt.plot(X, model.predict(X), color="black", linewidth=2, label="Sklearn Regression")
plt.xlabel("Square Feet")
plt.ylabel("House Price")
plt.title("Sklearn Linear Regression")
plt.legend()

# Gradient Descent Model Visualization
plt.subplot(2, 3, 2)  # Second subplot for Gradient Descent Model
plt.scatter(X_train, Y_train, color="blue", label="Training Data")  
plt.scatter(X_test, Y_test, color="red", label="Testing Data")  
plt.plot(X, m_gd * X + c_gd, color="green", linestyle="dashed", linewidth=2, label="Gradient Descent Regression")
plt.xlabel("Square Feet")
plt.ylabel("House Price")
plt.title("Gradient Descent Linear Regression")
plt.legend()

# Cost Function Visualization
plt.subplot(2, 3, 3)  # Third subplot for Cost Function
plt.plot(range(epochs), cost_history, color="purple", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Cost Function")
plt.title("Gradient Descent Cost Function")

# Residual Error Plot for Sklearn Model
plt.subplot(2, 3, 4)
residuals_sklearn = Y_test - Y_pred_sklearn
plt.scatter(Y_test, residuals_sklearn, color="orange", alpha=0.6)
plt.axhline(y=0, color='black', linestyle="dashed")
plt.xlabel("Actual House Price")
plt.ylabel("Residual Error")
plt.title("Residual Error: Sklearn Model")

# Residual Error Plot for Gradient Descent Model
plt.subplot(2, 3, 5)
residuals_gd = Y_test - Y_pred_gd.flatten()
plt.scatter(Y_test, residuals_gd, color="green", alpha=0.6)
plt.axhline(y=0, color='black', linestyle="dashed")
plt.xlabel("Actual House Price")
plt.ylabel("Residual Error")
plt.title("Residual Error: Gradient Descent Model")

# Test model with random Square Feet values
random_sqft = np.array([[1500], [2500], [3500]])
predicted_prices = model.predict(random_sqft)

print("\n--- Random Data Test ---")
for sqft, price in zip(random_sqft.flatten(), predicted_prices):
    print(f"Square Feet: {sqft}, Predicted Price: {price}")


plt.tight_layout()
plt.show()
