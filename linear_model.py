# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# The CSV file 'temperature_data.csv' is expected to be in the same directory as this script.
try:
    data = pd.read_csv('temperature_data.csv')
except FileNotFoundError:
    print("Error: 'temperature_data.csv' not found. Make sure the file is in the current directory.")
    exit()

# Define features (X) and target (y)
# 'Temperature_Celsius' is the feature we'll use to predict 'Ice_Cream_Sales'.
X = data[['Temperature_Celsius']]
y = data['Ice_Cream_Sales']

# Split the data into training and testing sets
# We use 80% of the data for training and 20% for testing.
# random_state is set for reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
# The model learns the relationship between temperature and ice cream sales from the training data.
model.fit(X_train, y_train)

# At this point, the model is trained.

# Make predictions on the test data
# The model uses the learned relationship to predict ice cream sales for the test set temperatures.
y_pred = model.predict(X_test)

# Evaluate the model
# Mean Squared Error (MSE) measures the average squared difference between actual and predicted values.
# Lower MSE indicates a better fit.
mse = mean_squared_error(y_test, y_pred)

# R-squared (R²) is the proportion of the variance in the dependent variable that is predictable from the independent variable.
# It ranges from 0 to 1, with 1 indicating a perfect fit.
r2 = r2_score(y_test, y_pred)

print("Linear Regression model trained successfully.")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Demonstrate the model with a sample temperature
# This section shows how to use the trained model to predict sales for a new temperature value.
sample_temperature = [[25]]  # Temperature in Celsius, needs to be a 2D array for the model
predicted_sales_for_sample = model.predict(sample_temperature)

print(f"\n--- Model Demonstration ---")
print(f"Sample Temperature (Celsius): {sample_temperature[0][0]}")
print(f"Predicted Ice Cream Sales: {predicted_sales_for_sample[0]:.2f}")
