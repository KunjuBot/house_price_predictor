import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Loading dataset...")
# Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

print("Preparing data...")
# Features (X) and Target (y)
X = df.drop('MedHouseVal', axis=1) # MedHouseVal is the median house value in 100,000s
y = df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
# Initialize and train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

print("Saving model...")
# Save the trained model to a file so we can use it in our web app
joblib.dump(model, 'house_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
print("Model saved successfully as 'house_model.pkl'")
