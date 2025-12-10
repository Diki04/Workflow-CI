import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns

# === Parameter dari CLI (default bila tidak ada input) ===
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")

# === Load dataset ===
data_path = "../MLProject/gym_preprocessing/gym_cleaned.csv"
try:
    print(f"📥 Memuat dataset tunggal dari: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset dimuat! Bentuk data awal: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: File {data_path} tidak ditemukan.")
    sys.exit(1)

# === Preprocessing Data ===
print("🧹 Melakukan preprocessing data...")

# Hapus duplikasi
before = len(df)
df = df.drop_duplicates()
print(f"   🔁 Hapus duplikasi: {before - len(df)} baris dihapus")

# Tangani nilai kosong
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"   ⚠️ Menemukan {missing} nilai kosong — mengganti dengan median/nilai modus")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
else:
    print("   ✅ Tidak ada nilai kosong")

# Encoding untuk kolom kategorikal (object)
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"   🔡 Encoding kolom kategorikal: {list(categorical_cols)}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    print("   ✅ Tidak ada kolom kategorikal")

print(f"✅ Preprocessing selesai. Bentuk data: {df.shape}")

# === Pisahkan fitur dan target ===
target_col = "Calories_Level"  # *** TARGET KLASIFIKASI ***
if target_col not in df.columns:
    print(f"❌ Kolom target '{target_col}' tidak ditemukan di dataset.")
    sys.exit(1)

X = df.drop(columns=[target_col])
y = df[target_col]

# === Split train dan test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Data train: {len(X_train)} | Data test: {len(X_test)}")

input_example = X_train.iloc[0:5]

# === Training Model ===
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# === Logging Parameter & Metrik ===
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# === METRIC KLASIFIKASI ===
acc = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average="macro", zero_division=0)
recall = recall_score(y_test, predicted, average="macro", zero_division=0)
f1 = f1_score(y_test, predicted, average="macro", zero_division=0)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

print(f"🏁 Run selesai. Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# === Logging Model ===
print("💾 Logging model ke MLflow...")
mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    input_example=input_example
)

# === Plot Confusion Matrix ===
print("📈 Membuat confusion matrix plot...")
cm = confusion_matrix(y_test, predicted)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plot_path = "confusion_matrix.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)

# === Simpan metrik ke JSON ===
print("🧮 Menyimpan metrik ke JSON...")
metrics = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}
json_path = "metrics.json"
with open(json_path, "w") as f:
    json.dump(metrics, f)
mlflow.log_artifact(json_path)

print("✅ Script selesai tanpa error.")
