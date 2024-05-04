from app import app
from app.controller import rfr_predictor_controller as RFRController

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/rfr-prediction/<int:month>')
def rfr_prediction(month):
    return RFRController.rfr_prediction(month);