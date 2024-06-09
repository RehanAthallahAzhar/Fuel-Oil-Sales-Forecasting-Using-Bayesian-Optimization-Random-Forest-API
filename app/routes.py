from app import app
from app.controller import rfr_predictor_controller, bo_rfr_predict_controller
from app.controller import read_dataset

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/rfr-prediction/<int:month>/<int:lag>')
def rfr_prediction(month, lag):
    return rfr_predictor_controller.rfr_prediction(month, lag);

@app.route('/bo-rfr-prediction/<int:month>/<int:lag>/<int:cvParam>')
def bo_rfr_prediction(month, lag,cvParam):
    # bo_rfr_prediction(month=1, lag=56, cvParam= 5)
    return bo_rfr_predict_controller.bo_rfr_prediction(month, lag, cvParam);

@app.route('/read-data-by-year/<int:year>')
def read_csv(year):
    return read_dataset.read_dataset(year)