import os
import logging
from dotenv import load_dotenv

from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS

from src.config.config import Config
from src.routes import api


# loading environment variables
load_dotenv()

# declaring flask application
app = Flask(__name__)

# app.secret_key = os.urandom(24)

# calling the dev configuration
config = Config().dev_config

# making our application to use dev env
app.env = config.ENV

# load the secret key defined in the .env file
app.secret_key = os.environ.get("SECRET_KEY")
bcrypt = Bcrypt(app)

# cors application
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger for the module
logger = logging.getLogger(__name__)


# # Path for our local sql lite database
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SQLALCHEMY_DATABASE_URI_DEV")

# # To specify to track modifications of objects and emit signals
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS")

# # sql alchemy instance
# db = SQLAlchemy(app)

# # Flask Migrate instance to handle migrations
# migrate = Migrate(app, db)

# # import models to let the migrate tool know
# from src.models.user_model import User

# routes configuration
app.register_blueprint(api, url_prefix='/api')

