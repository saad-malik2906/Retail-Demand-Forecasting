import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
from math import sqrt
import os

# --- Step 1: Load and preprocess dataset ---

# Change this path to your CSV dataset path
DATA_PATH = 'retail_sales.csv'  

df = pd.read_csv(DATA_PATH)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Drop columns not useful for forecasting
df_clean = df.drop(columns=['Transaction ID', 'Customer ID'])

# One-hot encode categorical variables 'Gender' and 'Product Category'
df_encoded = pd.get_dummies(df_clean, columns=['Gender', 'Product Category'])

# Group by Date and sum all numeric columns to get daily aggregates
daily_sales = df_encoded.groupby('Date').sum()

# --- Step 2: Feature Engineering ---

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1, 2, 3], rolling_windows=[3, 7]):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Create lag features
        for lag in self.lags:
            X_transformed[f'lag_{lag}'] = X_transformed['Total Amount'].shift(lag)

        # Create rolling mean features
        for window in self.rolling_windows:
            X_transformed[f'rolling_mean_{window}'] = X_transformed['Total Amount'].rolling(window=window).mean()

        # Drop rows with NaN values created by shifts and rolling calculations
        X_transformed = X_transformed.dropna()

        # Drop target column from features
        return X_transformed.drop(columns=['Total Amount'])

# --- Step 3: Prepare data for modeling ---

# Apply feature engineering
feature_engineer = FeatureEngineer()
X = feature_engineer.fit_transform(daily_sales)

# Target variable aligned to features (drop initial rows to align with lags)
y = daily_sales.loc[X.index, 'Total Amount']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Step 4: Build pipeline and train model ---

pipeline = Pipeline([
    ('model', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

# --- Step 5: Evaluate model ---

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Forest MAE: {mae:.2f}")
print(f"Random Forest RMSE: {rmse:.2f}")

# --- Step 6: Save the model ---

MODEL_PATH = 'random_forest_retail_demand.pkl'
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- Step 7: Plot actual vs predicted sales ---

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual Sales')
plt.plot(y_test.index, y_pred, label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.title('Actual vs Predicted Daily Sales')
plt.legend()

# Save plot to file in current directory
plot_path = 'actual_vs_predicted_sales.png'
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

plt.show()
