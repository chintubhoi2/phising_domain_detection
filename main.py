import os
import sys
from src.exception import PhisingException
from src.logger import logging
from src.configuration.mongo_db_connection import MongoDBClient
from src.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)