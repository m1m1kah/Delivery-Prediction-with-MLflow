import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# ===================================
# 0. Load Data
# ===================================
data = pd.read_csv("Processed_data_for_boosting.csv")

target = "Delivery_Time"
cat_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

# ===================================
# 1. One-hot encode categorical columns
# ===================================
X = pd.get_dummies(data.drop(columns=[target]), columns=cat_cols, drop_first=True)
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================================
# 2. Metrics Function
# ===================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return rmse, mae, r2

# ===================================
# 3. MLflow Experiment Setup
# ===================================
mlflow.set_experiment("Delivery_Time_Prediction_XGBoost")

# ===================================
# 4. Baseline XGBoost Model
# ===================================
with mlflow.start_run(run_name="XGBoost_Baseline"):
    baseline_xgb = XGBRegressor(
        random_state=42,
        tree_method="hist"  # no enable_categorical
    )
    rmse, mae, r2 = evaluate_model(baseline_xgb, X_train, y_train, X_test, y_test)

    mlflow.log_params(baseline_xgb.get_params())
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.xgboost.log_model(baseline_xgb, "XGBoost_Baseline")

print("✅ Baseline RMSE:", rmse, " | MAE:", mae, " | R²:", r2)

# ===================================
# 5. Optuna Objective
# ===================================
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "random_state": 42,
        "tree_method": "hist"
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# ===================================
# 6. Run Optuna Optimization
# ===================================
study_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(), pruner=HyperbandPruner())
study_xgb.optimize(objective_xgb, n_trials=30, timeout=900)

print("✅ Best XGBoost Params:", study_xgb.best_params)

# ===================================
# 7. Log Best XGBoost Model to MLflow
# ===================================
best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, tree_method="hist")

with mlflow.start_run(run_name="XGBoost_Optuna_Best"):
    rmse, mae, r2 = evaluate_model(best_xgb, X_train, y_train, X_test, y_test)
    mlflow.log_params(best_xgb.get_params())
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.xgboost.log_model(best_xgb, "XGBoost_Optuna_Best")

print(f"✅ Best XGBoost RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

# ===================================
# 8. Save Final Model
# ===================================
joblib.dump(best_xgb, "C:\\Users\Admin\\OneDrive\\Career and work\\Labmentix internship\\Project_3_Predict_Delivery_Times\\best_xgboost.pkl")
print("💾 Best XGBoost model saved successfully!")
