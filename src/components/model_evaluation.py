from src.exception import PhisingException
from src.logger import logging
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from src.ml.metric.classification_metric import get_classification_score
from src.ml.model.estimator import PhishingModel
from src.utills.main_utils import save_object,load_object,load_numpy_array,save_numpy_array,write_yaml_file
from src.ml.model.estimator import ModelResolver
from src.constant.training_pipeline import TARGET_COLOUMN
import sys,os
import pandas as pd

class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise PhisingException(e,sys)
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df,test_df])
            y_true = df[TARGET_COLOUMN]
            df.drop(TARGET_COLOUMN,axis=1,inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True


            if not model_resolver.is_model_exist():
                model_eval_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=train_model_file_path,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact)
                logging.info(f"Model evaluation artifact : {model_eval_artifact}")
                return model_eval_artifact
        
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(latest_model_path)
            train_model = load_object(train_model_file_path)
            best_model_path = None

            y_train_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(y_true,y_train_pred)
            latest_metric = get_classification_score(y_true,y_latest_pred)
            best_metric =None
            improved_acc = trained_metric.f1_score - latest_metric.f1_score

            if self.model_eval_config.model_evaluation_threshold < improved_acc:
                is_model_accepted = True
                best_model_path = train_model_file_path
                best_metric = trained_metric
            else:
                is_model_accepted = False
                best_model_path = latest_model_path
                best_metric = latest_metric

            model_eval_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_acc,
                trained_model_path=train_model_file_path,
                best_model_path=best_model_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=best_metric
            )

            model_eval_report = model_eval_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path,model_eval_report)
            logging.info(f"model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact
        
        except Exception as e:
            raise PhisingException(e,sys)
        
        