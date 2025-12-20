import os
import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Bank Churn - CI Workflow")

os.makedirs("./mlruns", exist_ok=True)

print("MLflow tracking lokal aktif: ./mlruns")
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

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_score", f1)

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

    # Save model
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    mlmodel_content = f"""artifact_path: model
flavors:
  sklearn:
    pickled_model: model.pkl
    sklearn_version: {sklearn.__version__}
    serialization_format: cloudpickle
  python_function:
    loader_module: mlflow.sklearn
    python_version: 3.9
    env: conda.yaml
run_id: null
utc_time_created: null
"""

    with open(os.path.join(model_dir, "MLmodel"), "w") as f:
        f.write(mlmodel_content.strip())

    conda_src = "conda.yaml"
    conda_dst = os.path.join(model_dir, "conda.yaml")
    if os.path.exists(conda_src):
        shutil.copy(conda_src, conda_dst)
        print("conda.yaml berhasil disalin ke folder model untuk Docker build")
    else:
        raise FileNotFoundError("File conda.yaml tidak ditemukan! Harus ada di folder MLProject untuk Docker build.")

    mlflow.log_artifacts(model_dir, artifact_path="model")

    print("Training & logging selesai!")
    print(f"   Test Accuracy : {acc:.4f}")
    print(f"   Test F1-Score : {f1:.4f}")