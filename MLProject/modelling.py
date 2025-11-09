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
from sklearn.metrics import r2_score, mean_squared_error

# --- KITA HAPUS 'with mlflow.start_run():' ---
# 'mlflow run' sudah menyediakan konteks run untuk kita.

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # --- 1. Load Data ---
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
        print(f"Error: Tidak dapat menemukan file data di '{data_path}'")
        sys.exit(1)

    # --- 2. Ambil Parameter dari Argumen CLI ---
    # Ini akan mengambil '100' dan '10' dari 'mlflow run ...'
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")

    # --- 3. Training Model ---
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # --- 4. Log Model (Perbaikan Kunci) ---
    print("Logging model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        # 'name' adalah pengganti 'artifact_path' yang modern
        # Ini WAJIB "model" agar 'mlflow build-docker' bisa menemukannya
        name="model",
        input_example=X_train[:5]
    )

    # --- 5. Log Metrik ---
    print("Logging metrics...")
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # --- 6. Log Parameter ---
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # --- 7. Log Artefak (Plot & JSON) ---
    print("Saving and logging artifacts...")
    
    # Buat folder 'artifacts' jika belum ada
    os.makedirs("artifacts", exist_ok=True)

    # Buat dan simpan plot regresi
    fig, ax = plt.subplots()
    ax.scatter(y_test, predicted, edgecolors=(0, 0, 0), alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Measured (True Calories)')
    ax.set_ylabel('Predicted (Calories)')
    ax.set_title('True vs. Predicted Values')
    plot_path = os.path.join("artifacts", "regression_plot.png")
    plt.savefig(plot_path)
    
    # Simpan metrik ke JSON
    metrics = {"r2_score": r2, "mse": mse}
    json_path = os.path.join("artifacts", "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f)

    # Log kedua artefak ke MLflow
    mlflow.log_artifact(plot_path)
    mlflow.log_artifact(json_path)

    print(f"Run complete. R2 Score: {r2:.4f}, MSE: {mse:.4f}")