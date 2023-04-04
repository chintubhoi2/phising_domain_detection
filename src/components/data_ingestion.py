from src.exception import PhisingException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.data_access.phising_data import PhisingData
from src.utills.main_utils import read_yaml_file
from src.constant.training_pipeline import SCHEMA_FILE_PATH


import os,sys
from pandas import DataFrame

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):

        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise PhisingException(e,sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from mongodb to feature store")
            phising_data = PhisingData()
            dataframe = phising_data.export_collection_as_dataframe(
                collection_name = self.data_ingestion_config.collection_name
            )
            fetaure_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(fetaure_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(fetaure_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise PhisingException(e,sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame)->None:
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train test split")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok= True)

            logging.info("exporting train test files")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)

            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train test files")

        except Exception as e:
            raise PhisingException(e,sys)
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            dataframe = dataframe.drop(self._schema_config['drop_columns'],axis=1)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise PhisingException(e,sys)




