from src.exception import PhisingException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.ml.metric.classification_metric import get_classification_score
from src.ml.model.estimator import PhishingModel
from src.utills.main_utils import save_object,load_object,load_numpy_array
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import os,sys
from xgboost import XGBClassifier


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise PhisingException(e,sys)
        
    def perform_hyper_param_tuning(self,x_train,y_train):
        try:
            logging.info("performing hyperparameter tuning")
            base_learners = [20,40,60,80,100,120]
            max_depth = [1,5,10,100,500,1000]
            param_grid = {'n_estimators':base_learners,'max_depth':max_depth}
            model = XGBClassifier()
            hsv = HalvingGridSearchCV(model,param_grid,verbose=10,n_jobs=3,min_resources="exhaust",factor=3)
            hsv.fit(x_train,y_train)
            logging.info(f"best params : {hsv.best_params_} best score : {hsv.best_score_}")
            return hsv.best_params_
        except Exception as e:
            raise PhisingException(e,sys)

    def train_model(self,x_train,y_train,n_estimators=120,max_depth=10):
        try:
            logging.info("starting model training")
            xgb_clf = XGBClassifier(max_depth=max_depth,n_estimators=n_estimators)
            xgb_clf.fit(x_train,y_train)
            return xgb_clf
        except Exception as e:
            raise PhisingException(e,sys)

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            ) 

            #best_params = self.perform_hyper_param_tuning(x_train,y_train)
            n_estimators = 80#best_params.get('n_estimators')
            max_depth = 1000#best_params.get('max_depth')

            model = self.train_model(x_train,y_train,n_estimators,max_depth)
            y_train_pred = model.predict(x_train)
            train_metric = get_classification_score(y_train,y_train_pred)

            if train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good")
            
            y_test_pred = model.predict(x_test)
            test_metric = get_classification_score(y_test,y_test_pred)

            diff = abs(train_metric.f1_score-test_metric.f1_score)
            if diff > self.model_trainer_config.overfit_underfit_threshold:
                raise Exception("model is not good try expermenting on different params")
            
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_obj_file_path)
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir,exist_ok=True)
            phising_model = PhishingModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path,phising_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            #logging.info(f"model training completed artifat: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise PhisingException(e,sys)