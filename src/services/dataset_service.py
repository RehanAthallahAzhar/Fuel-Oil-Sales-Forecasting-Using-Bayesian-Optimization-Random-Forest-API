import pandas as pd
from src.response import DataResp, JSON
from src.utils.http_status_code import HTTPStatusCode

from src.config.config import Config

class Dataset:
    def read(year):
        try:
            testFilePath = Config().test_file.TEST_FILE_PATH
            date_format = "%d/%m/%Y"
            df = pd.read_csv(testFilePath, delimiter=';', decimal=',', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format=date_format))
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

            return DataResp(HTTPStatusCode.OK, grouped_data.tolist(), "Success read Data")

        except ValueError as ve:
            return JSON(HTTPStatusCode.BAD_REQUEST, f"failed read: {str(ve)}")

        except Exception as e:
            return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}") 

