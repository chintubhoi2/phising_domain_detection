from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception import PhisingException
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:

    try:
        accuracy = accuracy_score(y_true,y_pred)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)

        metric = ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall,
            accuracy_score=accuracy
        )
        return metric
    except Exception as e:
        raise PhisingException(e,sys)