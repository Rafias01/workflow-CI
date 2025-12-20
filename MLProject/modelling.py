import os
import pandas as pd
import mlflow
import shutil
import sklearn
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Bank Churn - CI Workflow")

os.makedirs("mlruns", exist_ok=True)

print("MLflow tracking lokal aktif: mlruns/")
print("Experiment: Bank Churn - CI Workflow")
print("Training dimulai...\n")

DATA_PATH = "bank_dataset_preprocessing.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

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

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(model, "logreg_model.pkl")
    mlflow.log_artifact("logreg_model.pkl")

    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    mlflow.log_artifact("confusion_matrix.txt")

    df.describe().to_csv("dataset_summary.csv")
    mlflow.log_artifact("dataset_summary.csv")

    with open("feature_coefficients.txt", "w") as f:
        f.write("Feature Coefficients:\n\n")
        for feat, coef in zip(X.columns, model.coef_[0]):
            f.write(f"{feat}: {coef:.6f}\n")
    mlflow.log_artifact("feature_coefficients.txt")

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    import sklearn
    mlmodel_content = f"""
artifact_path: model
flavors:
  sklearn:
    pickled_model: model.pkl
    sklearn_version: {sklearn.__version__}
    serialization_format: cloudpickle
run_id: null
utc_time_created: null
"""
    with open(os.path.join(model_dir, "MLmodel"), "w") as f:
        f.write(mlmodel_content.strip())

    mlflow.log_artifacts(model_dir, artifact_path="model")

    current_experiment = mlflow.get_experiment_by_name("Bank Churn - CI Workflow")
    exp_id = current_experiment.experiment_id
    run_id = mlflow.active_run().info.run_id
    artifact_path = f"mlruns/{exp_id}/{run_id}/artifacts/model"
    os.makedirs(artifact_path, exist_ok=True)

    import shutil
    for file_name in os.listdir(model_dir):
        shutil.copy(os.path.join(model_dir, file_name), os.path.join(artifact_path, file_name))

    print("Training & logging selesai!")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")