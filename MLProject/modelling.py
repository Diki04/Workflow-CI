import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# --- (1) Baca parameter dari 'mlflow run' ---
# Parameter ini didefinisikan di file 'MLProject'
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# --- (2) Kode ini berjalan DI DALAM 'mlflow run' (Tidak perlu 'with mlflow.start_run()') ---
print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")

# Load Data (sesuai case Anda)
data_path = "gym_preprocessing/"
try:
    print("Loading preprocessed data...")
    X_train = np.load(data_path + "X_train.npy")
    X_test = np.load(data_path + "X_test.npy")
    y_train_df = pd.read_csv(data_path + "y_train.csv")
    y_test_df = pd.read_csv(data_path + "y_test.csv")
    y_train = y_train_df.values.ravel()
    y_test = y_test_df.values.ravel()
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data files not found in {data_path}")
    sys.exit(1)

# Ambil contoh input
input_example = X_train[0:5]

# Training Model (Regresi)
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# --- (3) Log Parameter dan Metrik ---
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# Metrik Regresi
r2 = r2_score(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
mlflow.log_metric("r2_score", r2)
mlflow.log_metric("mse", mse)

print(f"Run complete. R2 Score: {r2:.4f}, MSE: {mse:.4f}")

# --- (4) Log Model (Sangat Penting) ---
# 'name' harus "model" agar build-docker bisa menemukannya
print("Logging model...")
mlflow.sklearn.log_model(
    sk_model=model,
    # artifact_path="model", # <-- DIHAPUS, sesuai pesan error
    name="model", # Ini akan otomatis membuat folder 'model' di artefak
    input_example=input_example
)

# --- (5) PERBAIKAN: Simpan Artefak di Root (Bukan di sub-folder 'artifacts') ---
# Hapus 'os.makedirs("artifacts", ...)'

# Buat Plot Regresi (True vs Predicted)
print("Creating regression plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', lw=2)
plt.xlabel("True Values (Calories Burned)")
plt.ylabel("Predicted Values (Calories Burned)")
plt.title("True vs. Predicted Values")
# Simpan plot di root
plot_path = "regression_plot.png" 
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)

# Simpan Metrik ke JSON
print("Saving metrics to JSON...")
metrics = {
    "r2_score": r2,
    "mse": mse
}
# Simpan JSON di root
json_path = "metrics.json" 
with open(json_path, "w") as f:
    json.dump(metrics, f)
mlflow.log_artifact(json_path)

print("Script finished successfully.")