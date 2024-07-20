from src.config.dev_config import DevConfig
from src.config.production_config import ProductionConfig
from src.config.test_file import TestFile
from src.config.logger_config import setup_logger

class Config:
    def __init__(self):
        self.dev_config = DevConfig()
        self.production_config = ProductionConfig()
        self.test_file = TestFile()
        self.setup_logger = setup_logger()