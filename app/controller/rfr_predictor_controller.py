from flask import Flask, request, jsonify
import warnings

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app import app, response
from app.utils.date import formatTimestampToDay, convertMonthtoLatin

warnings.filterwarnings('ignore')

def rfr_prediction(month, lag):
    # Load dataset
    date_format = "%d/%m/%Y"
    df = pd.read_csv('./app/dataset/Biosolar.csv', delimiter=';', decimal=',', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
    df.dropna(inplace=True)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format=date_format)

    # # Remove outliers
    # data = df['outgoing'] 
    # Q1 = data.quantile(0.25)
    # Q3 = data.quantile(0.75)
    # IQR = Q3 - Q1
    # lower_limit = Q1 - 1.5 * IQR
    # upper_limit = Q3 + 1.5 * IQR
    # df['outgoing'] = data[(data >= lower_limit) & (data <= upper_limit)]

    # Drop rows with NaN values in 'outgoing'
    df.dropna(subset=['outgoing'], inplace=True)

    # Create lag features for 'outgoing'
    lags = lag # Number of lag features to create
    for lag in range(1, lags + 1):
        df[f'lag{lag}'] = df['outgoing'].shift(lag)

    # Drop rows with NaN values resulting from lagging
    df.dropna(inplace=True)

    # Preparation for training
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    X = df.drop(columns=['outgoing'])  # Features values, excluding 'outgoing'
    y = df['outgoing']  # Target variable

    # Show the first few rows of the feature set to confirm lag features are included
    print(X.head())

    # Split the data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train_scaled, y_train)

    # Predict for evaluate
    y_pred = rf_regressor.predict(X_test_scaled)

    data = json.dumps(y_pred.tolist())
    # print(data)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print("MAPE:", mape)

    predictions = []

    return response.successPredict(data, predictions, 'Successfully predicted outgoing for the next 30 days')

    # # Prediction for 1 month
    # future_dates = pd.date_range(start="2023-01-01", end="2023-01-31")
    # future_date = pd.DataFrame({'date': future_dates,
    #                             'population': 84772,
    #                             'GRDP per capita': 35178,
    #                             'price': 10000})

    # # Transform 'date' column to timestamp
    # future_date['date'] = future_date['date'].apply(lambda x: x.timestamp())

    # # Predict outgoing for January 2023
    # predictions = []
    # for i, row in future_date.iterrows():
    #     features = row.values.reshape(1, -1)
    #     features_scaled = scaler.transform(features)
    #     prediction = rf_regressor.predict(features_scaled)
    #     respons = {
    #         "tanggal": formatTimestampToDay(row['date']),
    #         "prediksi": round(prediction[0], 1)
    #     }
    #     predictions.append(respons)

    # # TODO: Passing Akurasi
    # return response.successPredict(convertMonthtoLatin(month), predictions, 'Successfully predicted outgoing for the next 30 days')
