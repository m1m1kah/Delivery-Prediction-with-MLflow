import pandas as pd 
import numpy as np 

#data processing for ml 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

#importing the ML models 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#imprting metrics 
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

#for ML flow 
import mlflow
import mlflow.sklearn

#for hyperparameter optimisaiton

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import joblib

data = pd.read_csv("processed_data.csv")

target = "Delivery_Time"
X = data.drop(columns=[target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling your data
# Scale only features (not target)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================
# 3. Evaluation Function
# ===================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Fit model, predict, and return metrics"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds) 
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return rmse, mae, r2

# ===================================
# 4. Baseline Models (without tuning)
# ===================================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

results = {}

mlflow.set_experiment("Delivery_Time_Prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        rmse, mae, r2 = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

        # Log metrics
        mlflow.log_param("model", name)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(model, name)

        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

# Show baseline results
pd.DataFrame(results).T

# ===================================
# 5. Hyperparameter Tuning with Optuna
# ===================================
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse


def objective_gb(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    subsample = trial.suggest_uniform("subsample", 0.5, 1.0)

    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse


# Run optimization
study_rf = optuna.create_study(direction="minimize", sampler=TPESampler(), pruner=HyperbandPruner())
study_rf.optimize(objective_rf, n_trials=30, timeout=600)

study_gb = optuna.create_study(direction="minimize", sampler=TPESampler(), pruner=HyperbandPruner())
study_gb.optimize(objective_gb, n_trials=30, timeout=600)

print("Best RF Params:", study_rf.best_params)
print("Best GB Params:", study_gb.best_params)

# ===================================
# 6. Log Best Models to MLflow
# ===================================
best_rf = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
best_gb = GradientBoostingRegressor(**study_gb.best_params, random_state=42)

for name, model in {"RandomForest_Optuna": best_rf, "GradientBoosting_Optuna": best_gb}.items():
    with mlflow.start_run(run_name=name):
        rmse, mae, r2 = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

        # Log params + metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(model, name)

# Fit the best RF model on the full training set
best_rf.fit(X_train_scaled, y_train)

# Save with joblib
joblib.dump(best_rf, "C:\\Users\\Admin\\OneDrive\\Career and work\\Labmentix internship\\Project_3_Predict_Delivery_Times\\best_random_forest.pkl")
print("✅ Model saved!")