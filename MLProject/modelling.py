import mlflow
import mlflow.tensorflow # Penting untuk log model TF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd

# Set MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gym_model_experiment_ci")

with mlflow.start_run() as run:
    
    # --- 1. Ambil Parameter ---
    # Ambil parameter dari MLflow run (jika dijalankan via 'mlflow run')
    params = mlflow.active_run().data.params
    epochs = int(params.get("epochs", 50)) # Default 50 epochs untuk Deep Learning
    batch_size = int(params.get("batch_size", 32)) # Default 32

    print(f"Training model TensorFlow dengan {epochs} epochs dan batch size {batch_size}")
    
    # Log parameter ke MLflow
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # --- 2. Load Data (Sesuai Case Anda) ---
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
        print(f"X_train shape: {X_train.shape}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Pastikan file .npy dan .csv ada di folder 'gym_preprocessing/'")
        exit(1) # Keluar jika data tidak ada

    # --- 3. Buat Arsitektur Model REGRESI ---
    print("Building REGRESSION model...")
    model = Sequential([
        # Tentukan input_shape di layer pertama
        Input(shape=(X_train.shape[1],)),
        # Hidden layers
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # Output layer untuk REGRESI
        # Hanya 1 neuron (output), tanpa aktivasi (linear)
        Dense(1) 
    ])
    
    # Compile model untuk REGRESI
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', # Loss untuk regresi
                  metrics=['mse', 'mae'])     # Metrik untuk regresi
    
    print(model.summary())

    # --- 4. Training Model ---
    print("Starting model training...")
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(X_test, y_test),
              verbose=2) # Tampilkan log training

    # --- 5. Evaluasi dan Log Metrik REGRESI ---
    print("Evaluating model...")
    eval_results = model.evaluate(X_test, y_test, verbose=0)
    
    # [loss, mse, mae]
    loss = eval_results[0]
    mse = eval_results[1]
    mae = eval_results[2]
    
    print(f"Evaluation complete. Loss (MSE): {loss:.4f}, MAE: {mae:.4f}")

    # Log metrik REGRESI
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # --- 6. Log Model dan Cetak RUN_ID (KRITIKAL) ---
    print("Logging model to MLflow...")
    # Log model TF, harus dinamai "model" agar 'build-docker' berfungsi
    mlflow.tensorflow.log_model(
        model, 
        "model",
        input_example=X_train[:5] # Tambahkan contoh input
    )

    print(f"Training selesai. Loss: {loss:.4f}, MAE: {mae:.4f}")
    
    # --- INI ADALAH KUNCI UTAMA CI/CD ---
    # Cetak Run ID agar bisa ditangkap oleh GitHub Actions
    print(f"MLFLOW_RUN_ID={run.info.run_id}")