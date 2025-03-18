import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from urllib.parse import urlparse
from src.Creditcardfaultdetection.utils.utils import load_obj
from mlflow.models import infer_signature


class ModelEvaluation:
    def __init__(self):
        pass
    
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        conf_matrix = confusion_matrix(actual, pred)
        return accuracy, conf_matrix

    def plot_confusion_matrix(self, conf_matrix):
        """Saves confusion matrix as an image."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

    def initiate_model_evaluation(self, test_arr):
        try:
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_obj(model_path)
            
            # Ensure this URL is correct for your remote MLflow instance
            mlflow.set_registry_uri('https://dagshub.com/BhaveshNikam09/Credit_card_default_prediction.mlflow')
            
            tracking_uri = mlflow.get_tracking_uri()
            tracking_url_type_store = urlparse(tracking_uri).scheme
            
            print(f"Tracking URI: {tracking_uri}")
            print(f"Tracking store type: {tracking_url_type_store}")
            
            # Ensure that MLflow is using the correct tracking URI
            if tracking_url_type_store == "file":
                print("Using local MLflow server (file-based store).")
            else:
                print("Using remote MLflow server.")

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)

                # Infer the model signature
                signature = infer_signature(X_test, predicted_qualities)

                # Compute evaluation metrics
                accuracy, conf_matrix = self.eval_metrics(y_test, predicted_qualities)
                
                # Log accuracy metric
                tn, fp, fn, tp = conf_matrix.ravel()  # Extract values from confusion matrix

                mlflow.log_metric("Accuracy", accuracy)  
                mlflow.log_metric("True Positives", tp)
                mlflow.log_metric("False Positives", fp)
                mlflow.log_metric("False Negatives", fn)
                mlflow.log_metric("True Negatives", tn)
                mlflow.autolog()  # Enables automatic logging, including some system metrics


                
                # Plot and log confusion matrix
                self.plot_confusion_matrix(conf_matrix)
                mlflow.log_artifact("confusion_matrix.png")

                # Log the model in MLflow with signature
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model", signature=signature)
                    print("Model logged to remote registry.")
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)
                    print("Model logged locally.")

        except Exception as e:
            print(f"Error: {e}")
            raise e
