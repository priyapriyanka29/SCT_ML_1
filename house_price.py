# ================================
# House Price Prediction (Final)
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ================================
# 1. Load Dataset
# ================================

data = pd.read_csv("train.csv")

print("Columns in Dataset:\n")
print(data.columns)

# ================================
# 2. Select Important Features
# ================================

data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'SalePrice']]

# ================================
# 3. Data Cleaning
# ================================

data = data.dropna()

print("\nCleaned Data:\n", data.head())

# ================================
# 4. Correlation Heatmap
# ================================

plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# ================================
# 5. Prepare Data
# ================================

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars']]
y = data['SalePrice']

# ================================
# 6. Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ================================
# 7. Train Model
# ================================

model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# 8. Predictions
# ================================

predictions = model.predict(X_test)

print("\nSample Predictions:", predictions[:5])
print("Actual Values:", list(y_test[:5]))

# ================================
# 9. Evaluation
# ================================

score = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, predictions)

print("\nModel Accuracy (R² Score):", score)
print("Mean Absolute Error:", mae)

# ================================
# 10. Visualization (Improved)
# ================================

plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# ================================
# 11. User Input Prediction
# ================================

print("\n--- Predict New House Price ---")

area = float(input("Enter area (GrLivArea): "))
bed = int(input("Enter bedrooms: "))
bath = int(input("Enter bathrooms: "))
garage = int(input("Enter garage capacity: "))

result = model.predict([[area, bed, bath, garage]])

print("Predicted House Price: ₹", round(result[0], 2))