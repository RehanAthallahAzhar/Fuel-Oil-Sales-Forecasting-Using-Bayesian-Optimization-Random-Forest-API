import pandas as pd
from app import app, response

def read_dataset(year):
    date_format = "%d/%m/%Y"
    df = pd.read_csv('./app/dataset/dataset.csv', delimiter=';', decimal=',', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
    df.dropna(inplace=True)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format=date_format)

    # Filter data based on the given year
    df = df[df['date'].dt.year == year]

    # Group by month
    df['month'] = df['date'].dt.month
    grouped_data = df.groupby('month')['outgoing'].sum()

    # Convert to JSON
    json_data = grouped_data.to_json(orient='index')

    return response.successGetData(grouped_data.tolist(), "Success Get Data")

# # Contoh penggunaan
# year_data = read_dataset(2024)
# print(year_data)