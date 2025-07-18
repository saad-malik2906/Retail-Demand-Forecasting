# Retail Sales Forecasting – Time Series with Random Forest

This project builds a **daily sales forecasting system** using Random Forest Regression. It applies time series feature engineering (lags and rolling means) to historical retail sales data and predicts future sales trends. It includes training, evaluation, model saving, and visualization of predictions.

---

## Project Summary

### ✅ What I Did:

1. **Data Preprocessing**
   - Loaded dataset: `retail_sales.csv`.
   - Converted `Date` column to datetime format.
   - Dropped unnecessary columns: `Transaction ID`, `Customer ID`.
   - One-hot encoded categorical variables: `Gender`, `Product Category`.
   - Aggregated the data to daily sales using `groupby('Date').sum()`.

2. **Feature Engineering**
   - Built a custom transformer `FeatureEngineer` to create:
     - **Lag features** (`lag_1`, `lag_2`, `lag_3`) of previous days' sales.
     - **Rolling mean features** (`rolling_mean_3`, `rolling_mean_7`) to capture trends.
   - Dropped any rows with NaN values caused by shifting and rolling operations.

3. **Modeling**
   - Used `RandomForestRegressor` as the prediction model.
   - Wrapped it in a `Pipeline` (no preprocessing needed after feature engineering).
   - Split the data into 80% train and 20% test (time-based split, no shuffle).

4. **Training and Evaluation**
   - Trained the model on engineered features.
   - Evaluated performance using:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
   - Achieved scores printed in console.

5. **Model Saving**
   - Saved the trained pipeline as `random_forest_retail_demand.pkl` using `joblib`.

6. **Prediction Visualization**
   - Plotted actual vs predicted daily sales using Matplotlib.
   - Saved the plot as `actual_vs_predicted_sales.png`.

---

## Future Improvements

- Use more advanced time series models (e.g., XGBoost, LightGBM, Prophet, ARIMA).
- Include additional date-based features like day of week, holidays, month, etc.
- Implement cross-validation with time series split instead of a simple hold-out.
- Deploy the model to a dashboard or app for real-time sales forecasting.