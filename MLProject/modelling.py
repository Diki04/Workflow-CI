import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(data_dir):
  """Memuat data training dan testing yang sudah diproses."""
  print("Memuat data...")
  X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
  X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
  y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
  y_test_df = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
  y_train = y_train_df.iloc[:, 0].values
  y_test = y_test_df.iloc[:, 0].values
  return X_train, X_test, y_train, y_test

def main(n_estimators, max_depth):
  """Fungsi utama untuk training dan logging."""
  with mlflow.start_run(run_name="CI_Pipeline_Run") as run:
    n_estimators = int(n_estimators)
    max_depth = None if max_depth == 'None' else int(max_depth)
    
    print("Memuat data dari 'gym_preprocessing'...")
    X_train, X_test, y_train, y_test = load_data('gym_preprocessing')
    
    print(f"Melatih model dengan n_estimators={n_estimators}, max_depth={max_depth}")
    model = RandomForestRegressor(
      n_estimators=n_estimators,
      max_depth=max_depth,
      random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 =  r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R2 Score: {r2}")
    print(f"RMSE: {rmse}")
    
    mlflow.log_param("n_estimators ", n_estimators)
    mlflow.log_param("max_depth ", max_depth)
    
    mlflow.log_metric("r2_score ",r2)
    mlflow.log_metric("rmse ",rmse)
    
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Model berhasil di-log di run {run.info.run_id}")
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--n_estimators", default=100)
  parser.add_argument("--max_depth", default='None')
  
  args = parser.parse_args()
  
  main(args.n_estimators, args.max_depth)
  