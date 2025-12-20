import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI tidak ditemukan! Pastikan sudah diset di GitHub Secrets.")
if not tracking_username:
    raise ValueError("MLFLOW_TRACKING_USERNAME tidak ditemukan! Pastikan token DagsHub sudah diset di Secrets.")

mlflow.set_tracking_uri(tracking_uri)

os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password or ""  # Kosong jika None

mlflow.set_experiment("Bank Churn - CI Workflow")

print(f"MLflow tracking aktif: {tracking_uri}")
print("Experiment: Bank Churn - CI Workflow")
print("Training dimulai...\n")

DATA_PATH = "bank_dataset_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="Logistic Regression - Baseline CI"):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("stratify", True)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    model_path = "logreg_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    cm_path = "confusion_matrix.txt"
    with open(cm_path, "w") as f:
        f.write(str(cm))
    mlflow.log_artifact(cm_path)

    summary_path = "dataset_summary.csv"
    df.describe().to_csv(summary_path)
    mlflow.log_artifact(summary_path)

    coef_path = "feature_coefficients.txt"
    with open(coef_path, "w") as f:
        f.write("Feature Coefficients (Logistic Regression):\n\n")
        for feature, coef in zip(X.columns, model.coef_[0]):
            f.write(f"{feature}: {coef:.6f}\n")
    mlflow.log_artifact(coef_path)

    mlflow.sklearn.log_model(model, "model")

    print("Training & logging selesai!")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   Run ID   : {mlflow.active_run().info.run_id}")
    print(f"   Semua artifact telah di-log ke DagsHub.")