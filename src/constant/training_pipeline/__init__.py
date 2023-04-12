import os


SAVED_MODEL_DIR = os.path.join("saved_model")
MODEL_FILE_NAME = "model.pkl"

PIPELINE_NAME:str = "phising"
ARTIFACT_DIR:str = "artifact"
FILE_NAME:str = "phising.csv"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
TARGET_COLOUMN:str = "phishing"

PREPROCESING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

""" 
Data ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME:str = "phising"
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:str = 0.2


""" 
Data validation related constants
"""
DATA_VALIDATION_DIR_NAME:str = "data_valiadtion"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"


""" 
Data transformation related constants
"""
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str ="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_obj"


"""
Model Training related constants
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDERFITTING_THRESHOLD: float = 0.05


""" 
Model Evaluation realted constant
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME = "report.yaml"