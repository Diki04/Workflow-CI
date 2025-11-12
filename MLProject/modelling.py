import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

def load_data(raw_data_path):
    """Memuat data MENTAH."""
    csv_path = os.path.join(raw_data_path, "gym_tracking.csv")
    try:
        print(f"Memuat data mentah dari {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.drop(['BMI', 'Experience_Level'], axis=1, errors='ignore')
        return df
    except FileNotFoundError:
        print(f"ERROR: File data mentah tidak ditemukan di {csv_path}")
        return None

def main():
    """Fungsi utama untuk training dan logging pipeline."""
    
    with mlflow.start_run(run_name="CI_Pipeline_Run (Smart Pipeline)") as run:
        
        # 1. Definisikan kolom berdasarkan tipe data MENTAH
        numerical_features = [
            'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
            'Resting_BPM', 'Session_Duration (hours)', 'Fat_Percentage', 
            'Water_Intake (liters)', 'Workout_Frequency (days/week)'
        ]
        
        categorical_features = ['Gender', 'Workout_Type']
        
        target_col = "Calories_Burned"

        # 2. Buat transformer untuk setiap tipe kolom
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 3. Gabungkan transformer menggunakan ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        print("Membaca parameter dari MLflow run...")
        
        params = mlflow.active_run().data.params
        
        n_estimators = int(params.get("n_estimators", 100))
        max_depth_str = params.get("max_depth", 'None')
        max_depth = None if max_depth_str == 'None' else int(max_depth_str)
        
        print(f"Menggunakan n_estimators={n_estimators}, max_depth={max_depth}")
        
        
        # 4. Muat data mentah
        df = load_data("gym_preprocessing") 
        if df is None:
            raise Exception("Gagal memuat data mentah. Hentikan run.")

        # Pisahkan fitur dan target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Definisikan Model Regresi
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42
        )
        
        # 6. BUAT PIPELINE LENGKAP
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # 7. Latih pipeline
        print("Melatih pipeline lengkap...")
        pipeline.fit(X_train, y_train)
        
        # 8. Evaluasi
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"R2 Score: {r2}")
        print(f"RMSE: {rmse}")

        # 9. Log Metrik
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        
        # 10. Log model (seluruh pipeline)
        print("Logging pipeline sebagai model...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model_pipeline", 
            input_example=X_train.iloc[0:5]
        )
        
        print(f"Model pipeline berhasil di-log di run {run.info.run_id}")

if __name__ == "__main__":
    main()