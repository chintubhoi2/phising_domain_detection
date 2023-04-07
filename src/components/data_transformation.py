import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.constant.training_pipeline import TARGET_COLOUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact)

from src.exception import PhisingException
from src.logger import logging
from src.utills.main_utils import save_numpy_array,save_object




class DataTransformation:
    
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                    data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise PhisingException(e,sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise PhisingException(e,sys)
        
    @classmethod
    def get_transformer_object(cls)->object:
        try:
            robust_scaler = RobustScaler()
            return robust_scaler
        except Exception as e:
            raise PhisingException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            preprocessor = self.get_transformer_object()

            #train dataframe
            input_feature_train_df = train_df.drop(TARGET_COLOUMN,axis=1)
            target_feature_train_df = train_df[TARGET_COLOUMN]

            #test dataframe
            input_feature_test_df = test_df.drop(TARGET_COLOUMN,axis=1)
            target_feature_test_df = test_df[TARGET_COLOUMN]

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train_df = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test_df = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_df,np.array(target_feature_test_df)]
            
            #save numpy array
            save_numpy_array(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_obj_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data Transformation Artifact : {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise PhisingException(e,sys)