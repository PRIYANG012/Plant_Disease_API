from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource

import logging
from rich.logging import RichHandler


# Initialising and configuring flask object
app = Flask(__name__)
app.config['SECRET_KEY'] = b'\xb0\xf4\xe8\\U\x8d\xba\xb4B2h\x88\xf9\x08\xb1J'

CORS(app=app)

api = Api(app=app)

# Logging module
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler()]
)

logger = logging.getLogger("rich")