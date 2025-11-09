import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor  # <-- DIUBAH KE REGRESSOR
from sklearn.metrics import mean_squared_error, r2_score  # <-- DIUBAH METRIKNYA

def train():
    """
    Fungsi utama untuk training model regresi (Calories_Burned).
    Dijalankan oleh 'mlflow run'.
    """
    
    # 1. Set tracking URI dan Experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("gym_model_experiment_ci")

    # Mulai MLflow Run
    with mlflow.start_run() as run:
        
        # --- BAGIAN LOADING DATA ---
        
        # 1. Tentukan Path
        data_path = "gym_preprocessing/"
        
        # 2. Load Data yang Sudah di-preprocess
        try:
            print("Loading preprocessed data...")
            X_train = np.load(data_path + "X_train.npy")
            X_test = np.load(data_path + "X_test.npy")
            
            y_train_df = pd.read_csv(data_path + "y_train.csv")
            y_test_df = pd.read_csv(data_path + "y_test.csv")
            
            # Ubah y dari DataFrame ke 1D array
            y_train = y_train_df.values.ravel()
            y_test = y_test_df.values.ravel()
            
            print("Data loaded successfully.")

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print(f"Pastikan semua file ada di folder '{data_path}'")
            return

        # --- BAGIAN TRAINING (SUDAH DIUBAH) ---

        # 3. Training Model Regresi
        params = {"n_estimators": 100, "random_state": 42}
        mlflow.log_params(params)
        
        # Menggunakan Regressor
        model = RandomForestRegressor(**params) 
        model.fit(X_train, y_train) 
        print("Model training complete.")

        # 4. Evaluasi Model Regresi
        y_pred = model.predict(X_test)
        
        # Menggunakan metrik regresi
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrik yang benar
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        print(f"R2 Score Model: {r2}")
        print(f"MSE Model: {mse}")

        # --- BAGIAN LOGGING (DENGAN PERBAIKAN) ---

        # 5. (KRITIKAL) Log Model
        # Menambahkan input_example untuk menghilangkan warning
        mlflow.sklearn.log_model(
            model,
            "model", # Ini harus "model" agar build-docker berfungsi
            input_example=X_train[:5] # Ambil 5 baris sebagai contoh
        )

        # 6. (KRITIKAL) Cetak Run ID
        # 'Jembatan' ke file ci_pipeline.yml
        print(f"MLFLOW_RUN_ID={run.info.run_id}")


# Main guard
if __name__ == "__main__":
    train()