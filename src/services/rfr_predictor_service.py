from flask import Flask, request, jsonify
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.response import PredictResp
from src.utils.http_status_code import HTTPStatusCode
from src.utils.date_format import FormatTimestampToDay, ConvertMonthtoLatin
from src.utils.data_format import RemoveOutlier

from src.config.logger_config import logger
from src.config.config import Config

class RandomForest:
    def predict(month, lag):
        try:
            testFilePath = Config().test_file.TEST_FILE_PATH
            date_format = "%d/%m/%Y"
            df = pd.read_csv(testFilePath, delimiter=';', decimal=',', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
            df.dropna(inplace=True)

            df['date'] = pd.to_datetime(df['date'], format=date_format)

            df = RemoveOutlier(df, df['outgoing'])

            # Create lag features as list for 'outgoing'
            for lag_i in range(1, lag + 1):
                df[f'lag_{lag_i}'] = df['outgoing'].shift(lag_i)

            # Drop rows with NaN values resulting from lagging
            df.dropna(inplace=True)

            # Convert 'date' column to timestamp
            df['date'] = df['date'].apply(lambda x: x.timestamp())

            # Define features (including lagged values) and target
            X = df.drop(columns=['outgoing'])
            y = df['outgoing']  # Target variable

            # Ensure all column names are strings
            X.columns = X.columns.astype(str)

            # Split the data into training and testing sets (80:20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Standardize the dataset
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize the RandomForestRegressor with minimal tuning
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)

            # Predict for evaluation
            y_pred = rf.predict(X_test_scaled)

            # RMSE MAE MAPE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f'RMSE: {rmse}')

            # MAE
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f'MAE: {mae}')

            # MAPE
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            logger.info(f'mape: {mape}')


            # Prediction for 1 month
            future_dates = pd.date_range(start="2024-01-01", end="2024-01-31")
            future_date = pd.DataFrame({'date': future_dates,
                                        'population': 84772,
                                        'GRDP per capita': 35178,
                                        'price': 10000})

            # Transform 'date' column to timestamp
            future_date['date'] = future_date['date'].apply(lambda x: x.timestamp())

            # Initialize the lags with the last known outgoing values
            last_known_lags = df['outgoing'].iloc[-lag:].tolist()

            predictions = []
            for i, row in future_date.iterrows():
                # Update the row with the current lag values
                for lag_i in range(1, lag + 1):
                    row[f'lag_{lag_i}'] = last_known_lags[-lag_i]

                # Convert the row to a DataFrame with the same column names
                row_df = pd.DataFrame([row])

                # Ensure all column names are strings
                row_df.columns = row_df.columns.astype(str)

                # Ensure the columns are in the same order as the training data
                row_df = row_df[X.columns]

                # Fill NaN values with the mean of the respective columns (or any other appropriate method)
                row_df.fillna(row_df.mean(), inplace=True)

                # Transform the features using the scaler
                features_scaled = scaler.transform(row_df)

                # Predict the outgoing value
                prediction = rf.predict(features_scaled)

                # Store the prediction
                respons = {
                    "date": FormatTimestampToDay(row['date']),
                    "prediction": round(prediction[0], 3)
                }
                predictions.append(respons)

                # Update last_known_lags with the latest predictions
                last_known_lags = [round(prediction[0], 3)] + last_known_lags[:-1]  # Update the lag list

            return PredictResp(HTTPStatusCode.OK, ConvertMonthtoLatin(month), predictions, 'Successfully predict data 30 days ahead')

        except ValueError as ve:
            return JSON(HTTPStatusCode.BAD_REQUEST, f"failed predict: {str(ve)}")

        except Exception as e:
            return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}") 