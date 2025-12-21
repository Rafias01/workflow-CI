import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib  

mlflow.set_experiment("Bank Churn - CI Workflow")

print("MLflow tracking lokal aktif (default: ./mlruns)")
print("Experiment: Bank Churn - CI Workflow")
print("Training dimulai..\n")

DATA_PATH = "bank_dataset_preprocessing.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("test_accuracy", acc)
mlflow.log_metric("test_f1_score", f1)

mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("max_iter", 1000)
mlflow.log_param("random_state", 42)

print("Logging model dengan cara resmi MLflow...")
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

pkl_path = "logreg_model.pkl"
joblib.dump(model, pkl_path)
mlflow.log_artifact(pkl_path, artifact_path="model")  
print(f"logreg_model.pkl berhasil dibuat dan di-log ke MLflow")

print("\nMODEL BERHASIL DI-LOG KE MLFLOW!")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1-Score : {f1:.4f}")