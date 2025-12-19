import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bank_dataset_preprocessing.csv")

RUN_ID = os.environ.get("MLFLOW_RUN_ID")
if RUN_ID is None:
    raise RuntimeError("MLFLOW_RUN_ID tidak ditemukan. Script ini harus dijalankan via `mlflow run`.")

mlflow.set_experiment("Bank Churn - Baseline Model")

with mlflow.start_run(run_id=RUN_ID):

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 2000)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    model_path = os.path.join(BASE_DIR, "logreg_model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    cm_path = os.path.join(BASE_DIR, "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write(str(cm))
    mlflow.log_artifact(cm_path)

    mlflow.sklearn.log_model(model, artifact_path="model")