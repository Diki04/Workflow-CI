import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import sys
import warnings
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # --- 1. Load Data (Sesuai Case Anda) ---
    data_path = "gym_preprocessing/"
    try:
        X_train = np.load(data_path + "X_train.npy")
        X_test = np.load(data_path + "X_test.npy")
        y_train = pd.read_csv(data_path + "y_train.csv").values.ravel()
        y_test = pd.read_csv(data_path + "y_test.csv").values.ravel()
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_path}")
        sys.exit(1)

    # Input example untuk 'mlflow.sklearn.log_model'
    input_example = X_train[0:5]

    # --- 2. Ambil Parameter dari CLI (didefinisikan di MLProject) ---
    # Jika 'mlflow run' dijalankan tanpa arg, ini akan jadi default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with mlflow.start_run():
        # Set tracking URI agar 'mlruns' ada di dalam folder MLProject
        mlflow.set_tracking_uri("file:./mlruns")

        # --- 3. Training Model REGRESI ---
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)

        # --- 4. Log Model ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Ini harus "model"
            input_example=input_example
        )

        # --- 5. Log Metrik REGRESI ---
        r2 = r2_score(y_test, predicted)
        mse = mean_squared_error(y_test, predicted)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

        # Log parameter yang digunakan
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", str(max_depth)) # Log sebagai string

        # --- 6. Buat dan Simpan Artefak Visualisasi ---
        # Ganti Confusion Matrix dengan Plot Regresi
        
        # Buat folder 'artifacts' (meniru 'cm' dari contoh)
        os.makedirs("artifacts", exist_ok=True)
        
        # Buat Scatter Plot (True vs Predicted)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, predicted, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Nilai Asli (True Values)")
        ax.set_ylabel("Nilai Prediksi (Predicted Values)")
        ax.set_title(f"Plot Regresi (R2: {r2:.3f})")
        
        plot_path = os.path.join("artifacts", "regression_plot.png")
        plt.savefig(plot_path)
        
        # Log .png ini ke MLflow
        mlflow.log_artifact(plot_path)

        # Simpan metrik ke JSON (meniru contoh)
        metrics = {
            "r2_score": r2,
            "mse": mse
        }
        json_path = os.path.join("artifacts", "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f)
        
        # Log .json ini ke MLflow
        mlflow.log_artifact(json_path)

        print(f"Run complete. R2 Score: {r2:.4f}, MSE: {mse:.4f}")