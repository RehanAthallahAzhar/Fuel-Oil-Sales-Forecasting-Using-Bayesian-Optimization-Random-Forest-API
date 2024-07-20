from flask import Blueprint, request

from src.services.bo_rfr_predict_service import BayesianRandomForest
from src.services.dataset_service import Dataset
from src.services.rfr_predictor_service import RandomForest
from src.services.file_upload_service import Upload

from src.utils.http_status_code import HTTPStatusCode
from src.response import JSON


ml = Blueprint('ml', __name__)

@ml.route('/')
def index():
    return 'Hello World for Machine Learning Page!'

@ml.route('/rfr-predict') # month and lag
def rfr_predict():
    try:
        month = request.args.get('month', type=int)
        lag = request.args.get('lag', type=int)

        missing_params = [
            name for param, name in ((month, "month"), (lag, "lag"))
            if param is None
        ]
        
        if missing_params:
            return JSON(HTTPStatusCode.BAD_REQUEST, f"Missing required query parameter(s): {', '.join(missing_params)}")

        return RandomForest.predict(month, lag)
    
    except ValueError:
        return JSON(HTTPStatusCode.BAD_REQUEST, "Invalid input type")

    except Exception as e:
        return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}")

@ml.route('/bo-rfr-predict') # month, lag, cvParam
def bo_rfr_predict():
    try:
        month = request.args.get('month', type=int)
        lag = request.args.get('lag', type=int)
        cvParam = request.args.get('cvparam', type=int)

        missing_params = [
            name for param, name in ((month, "month"), (lag, "lag"), (cvParam, "cvparam"))
            if param is None
        ]
        
        if missing_params:
            return JSON(HTTPStatusCode.BAD_REQUEST, f"Missing required query parameter(s): {', '.join(missing_params)}")

        return BayesianRandomForest.predict(month, lag, cvParam)

    except ValueError:
        return JSON(HTTPStatusCode.BAD_REQUEST, "Invalid input type")

    except Exception as e:
        return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}")

@ml.route('/read-data-by-year') # year
def read_csv():
    try:
        year = request.args.get('year', type=int)
        
        if year is None:
            return JSON(HTTPStatusCode.BAD_REQUEST, "Missing year as a parameter")

        return Dataset.read(year)

    except ValueError:
        return JSON(HTTPStatusCode.BAD_REQUEST, "Invalid input type")

    except Exception as e:
        return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}")


@ml.route('/upload', methods=['POST'])
def upload_fuelConsumption():
    try:
        return Upload.excel()

    except ValueError:
        return JSON(HTTPStatusCode.BAD_REQUEST, "Invalid input type")

    except Exception as e:
        return JSON(HTTPStatusCode.INTERNAL_SERVER_ERROR, f"Internal Server Error: {str(e)}")