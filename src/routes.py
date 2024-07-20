from flask import Blueprint
from src.controllers.user_controller import users
from src.controllers.ml_controller import ml

api = Blueprint("v1", __name__)

# Register the blueprints with the api blueprint
api.register_blueprint(users, url_prefix='/users')
api.register_blueprint(ml, url_prefix='/ml')