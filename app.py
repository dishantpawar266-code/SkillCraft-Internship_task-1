import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)
print(data.head())

# Features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target
y = data["SalePrice"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# Coefficients
print("\n===== MODEL COEFFICIENTS =====")
print("Square Footage :", model.coef_[0])
print("Bedrooms       :", model.coef_[1])
print("Bathrooms      :", model.coef_[2])
print("Intercept       :", model.intercept_)

# New house prediction
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})

prediction = model.predict(new_house)

print("\n===== NEW HOUSE PREDICTION =====")
print("Square Feet : 2000")
print("Bedrooms    : 3")
print("Bathrooms   : 2")
print("Predicted Price : $", round(prediction[0], 2))

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()