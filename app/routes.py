from app import app
from app.controller import rfr_predictor_controller
from app.controller import read_dataset

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/rfr-prediction/<int:month>/<int:lag>')
def rfr_prediction(month, lag):
    return rfr_predictor_controller.rfr_prediction(month, lag);

@app.route('/read-data-by-year/<int:year>')
def read_csv(year):
    return read_dataset.read_dataset(year)