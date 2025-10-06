import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ------------------------
# 1. Train Model
# ------------------------
def train_model(model, X_train, y_train):
    """Fit a model on training data."""
    model.fit(X_train, y_train)
    return model


# ------------------------
# 2. Evaluate Model
# ------------------------
def evaluate_model(model, X_test, y_test, average="binary"):
    """
    Evaluate a trained model on test data.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average=average),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return metrics


# ------------------------
# 3. Save Model
# ------------------------
def save_model(model, path):
    """Save a model (or pipeline) to disk."""
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


# ------------------------
# 4. Load Model
# ------------------------
def load_model(path):
    """Load a model (or pipeline) from disk."""
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model


# ------------------------
# 5. MLflow Logging (Optional)
# ------------------------
def log_with_mlflow(model, X_test, y_test, run_name="experiment"):
    """
    Log model, parameters, and metrics to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        # Log parameters (if available from estimator)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        })

        # Log confusion matrix as artifact (optional)
        # Could extend with matplotlib figure export
        mlflow.sklearn.log_model(model, artifact_path="model")

        return metrics