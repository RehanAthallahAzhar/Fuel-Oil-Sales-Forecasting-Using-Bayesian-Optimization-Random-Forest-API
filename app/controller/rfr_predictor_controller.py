from flask import Flask, request, jsonify
import warnings

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

from app import app, response
from app.utils.date import formatTimestampToDay, convertMonthtoLatin

warnings.filterwarnings('ignore')

def rfr_prediction(month):
    date_format = "%d/%m/%Y"
    df = pd.read_csv('./app/dataset/dataset.csv', delimiter=';', decimal='.', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
    df.dropna(inplace=True)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format=date_format)

    # Preparation for training
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    X = df.drop(columns=['outgoing'])  # Features values, excluding 'outgoing'
    y = df['outgoing']  # Target variable

    # Remove outliers
    data = df['outgoing'] 
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df['outgoing'] = data[(data >= lower_limit) & (data <= upper_limit)]

    # Split the data into training and testing sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train_scaled, y_train)

    # Predict Dataset
    y_pred = rf_regressor.predict(X_test_scaled)

    data = json.dumps(y_pred.tolist())

    # Prediction for 1 montth
    future_dates = pd.date_range(start="2023-01-01", end="2023-01-31")
    future_date = pd.DataFrame({'date': future_dates,
                                    'population': 84772,
                                    'GRDP per capita': 35178,
                                    'price': 10000})

    # Transform 'date' column to timestamp
    future_date['date'] = future_date['date'].apply(lambda x: x.timestamp())

    # Predict outgoing for January 2023
    predictions = []
    for i, row in future_date.iterrows():
        features = row.values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = rf_regressor.predict(features_scaled)
        respons = {
            "tanggal" : formatTimestampToDay(row['date']),
            "prediksi" : round(prediction[0],1)
        }
        predictions.append(respons)

    return response.successPredict(convertMonthtoLatin(month),predictions, 'Successfully predicted outgoing for the next 30 days')


    # # Membuat DataFrame untuk tanggal 1 sampai 31 Januari 2023
    # start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
    # end_date = datetime.strptime("2023-01-31", "%Y-%m-%d")
    # dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    # january_2023_data = pd.DataFrame({'date': dates})

    # print(january_2023_data)
    
    # # Menambahkan fitur variabel lainnya
    # january_2023_data['population'] = 84772
    # january_2023_data['GRDP per capita'] = 35178
    # january_2023_data['price'] = 10000
    
    # # Memperoleh outgoing untuk hari sebelumnya untuk setiap tanggal
    # outgoing_for_previous_day = []  # List untuk menyimpan outgoing hari sebelumnya
    # for date in dates:
    #     previous_day = date - timedelta(days=1)
    #     # Convert date to proper datetime format if needed
    #     previous_day_data = df[df['date'] == previous_day]
    #     if not previous_day_data.empty:
    #         outgoing_for_previous_day.append(previous_day_data.iloc[0]['outgoing'])
    #     else:
    #         outgoing_for_previous_day.append(0)  # Jika data untuk hari sebelumnya tidak tersedia, dianggap sebagai 0
            
    # january_2023_data['outgoing'] = outgoing_for_previous_day
        
    # # Konversi tanggal ke timestamp jika diperlukan
    # january_2023_data['date'] = january_2023_data['date'].apply(lambda x: x.timestamp())
    
    # # Skala fitur
    # # january_2023_features = january_2023_data.drop(columns=['date'])
    # january_2023_features_scaled = scaler.transform(january_2023_data)
    
    # # Prediksi outgoing untuk setiap tanggal
    # predictions = rf_regressor.predict(january_2023_data)
    
    # # Mengembalikan hasil prediksi
    # return response.success(predictions.tolist(), 'Successfully predicted outgoing for the next 30 days')

    # # # Memanggil fungsi untuk memprediksi outgoing untuk Januari 2023
    # # predicted_outgoing_january_2023 = predict_outgoing_for_january_2023(rf_regressor, scaler)
    # # print(predicted_outgoing_january_2023.tolist())

    # return response.success(data, 'Successfully predicted outgoing for the next 30 days')
