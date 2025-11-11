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

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")

# Load dataset 
data_path = "../MLProject/gym_preprocessing/gym_preprocessing.csv"
try:
    print(f"📥 Memuat dataset tunggal dari: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset dimuat! Bentuk data: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: File {data_path} tidak ditemukan.")
    sys.exit(1)

# Pisahkan fitur dan target
target_col = "Calories_Burned"
if target_col not in df.columns:
    print(f"❌ Kolom target '{target_col}' tidak ditemukan di dataset.")
    sys.exit(1)

X = df.drop(columns=[target_col])
y = df[target_col]

# Split dataset jadi train dan test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Data train: {len(X_train)} | Data test: {len(X_test)}")

input_example = X_train.iloc[0:5]

# Training Model (Regresi) 
model = RandomForestRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# Log Parameter dan Metrik ke MLflow
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

r2 = r2_score(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
mae = np.mean(np.abs(y_test - predicted))
rmse = np.sqrt(mse)

mlflow.log_metric("r2_score", r2)
mlflow.log_metric("mse", mse)
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)

print(f"🏁 Run selesai. R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Log Model
print("💾 Logging model ke MLflow...")
mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    input_example=input_example
)

# Buat Plot Hasil Regresi 
print("📈 Membuat plot hasil prediksi...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--k',
    lw=2
)
plt.xlabel("True Values (Calories Burned)")
plt.ylabel("Predicted Values (Calories Burned)")
plt.title("True vs. Predicted Values")

plot_path = "regression_plot.png"
plt.savefig(plot_path)
mlflow.log_artifact(plot_path)

# Simpan metrik ke JSON 
print("🧮 Menyimpan metrik ke JSON...")
metrics = {
    "r2_score": r2,
    "mse": mse,
    "mae": mae,
    "rmse": rmse
}
json_path = "metrics.json"
with open(json_path, "w") as f:
    json.dump(metrics, f)
mlflow.log_artifact(json_path)

print("✅ Script selesai tanpa error.")
