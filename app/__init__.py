from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua rute

from app import routes