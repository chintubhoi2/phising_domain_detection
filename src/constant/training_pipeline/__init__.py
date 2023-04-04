import os


PIPELINE_NAME:str = "phising"
ARTIFACT_DIR:str = "artifact"
FILE_NAME:str = "phising.csv"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

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