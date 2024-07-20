# Fuel Oil Sales Forecasting Using Bayesian Optimization Random Forest

## Description
The increase in fuel consumption in the transportation industry is driven by population growth and an increase in per capita GDP. Gas stations order fuel from Dempo based on sales transactions and the amount of fuel available in the tanks. One of the issues faced by gas station X in Lumajang District is the difficulty in predicting the amount of fuel needed, which impacts the tank inventory. The gas station often experiences shortages because the demand exceeds the forecast. This study aims to analyze the random forest algorithm with and without outlier removal and compare it to a model optimized using Bayesian Optimization (BO) with and without outliers in predicting fuel consumption. The study uses data from January 2022 to December 2023. The objective is to provide a basic reference for determining the amount of fuel to order using the Economic Order Quantity (EOQ) method. Variables considered include sales data, prices, population, and per capita GDP in Lumajang.

## Running on Local
- clone app
```bash
git clone https://github.com/RehanAthallahAzhar/Syirkah-Amanah-API.git
```
- install packages in virtual environment
```bash
pip install -r requirements.txt
```
- Run flask
```bash
flask run
```

## Architecture
Embraces the MVC concept:
- Controller/Handler Layer This is the link between the client and the server of the application. Mainly to facilitate data and coordinate the actions to be taken.
- Service Layer (Business logic) The purpose of this layer is to implement specific operations and flow of application requirements such as business flow implementation, data processing, interaction with later data, etc.
- Data Access Layer is the component that interacts with the database such as retrieving data, manipulating data, connecting databases, mapping data, and performing transaction flow.(ONGOING)

## API Specification

### Machine Learning Route
### GET api/ml/rfr-predict
Predict fuel consumption using random forest regression

- **URL Query** : month(int) (will be removed), lag(int)
- **Data Params** : None
- **Headers** : Content-type: application/json
- **Success Response Code**: 200

JSON response:
```JavaScript
{
    data: [
        {
            "date": 1,
            "prediction": 21378.24
        },
        {
            "date": 2,
            "prediction": 17263.24
        },
        {
            "date": 3,
            "prediction": 3521.24
        },
    ],
    "message": "Successfully predict data",
    "month": "January"
}
```

### GET api/ml/bo-rfr-predict
Predict fuel consumption using bayesian optimization random forest

- **URL Query** : month(int)(will be removed), lag(int), and cvparam(int)
- **Data Params** : None
- **Headers** : Content-type: application/json
- **Success Response Code**: 200
- **Response** :
    ```JavaScript
    {
        data: [
            {
                "date": 1,
                "prediction": 21378.24
            },
            {
                "date": 2,
                "prediction": 17263.24
            },
            {
                "date": 3,
                "prediction": 3521.24
            },
        ],
        "message": "Successfully predict data",
        "month": "January"
    }
    ```

### GET api/ml/read-data-by-year
read data by year from csv

- **URL Query** : year(int)
- **Data Params** : None
- **Headers** : Content-type: application/json
- **Success Response Code**: 200
- **Response** :
    ```JavaScript
    {
        data: [
            124321,
            412442,
            541351,
            124125,
            513513,
            531512
        ],
        "message": "Success read Data"
    }
    ```

### GET api/ml/upload
upload excel file

- **URL Query** : None
- **Data Params** : None
- **Headers** : Content-type: application/json
- **Success Response Code**: 200
- **Response** :
    ```JavaScript
    {
        data: [
            {
                "date": 1,
                "prediction": 21378.24
            },
            {
                "date": 2,
                "prediction": 17263.24
            },
            {
                "date": 3,
                "prediction": 3521.24
            },
        ]
    }
    ```


## References:
- folder structure : https://ashleyalexjacob.medium.com/flask-api-folder-guide-2023-6fd56fe38c00