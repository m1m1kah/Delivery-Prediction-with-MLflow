import streamlit as st
import pandas as pd
import joblib

# === Load model ===
model = joblib.load("C:\\Users\\Admin\\OneDrive\\Career and work\\Labmentix internship\\Project_3_Predict_Delivery_Times\\best_xgboost.pkl")

st.set_page_config(page_title="Delivery Time Predictor", layout="centered")
st.title("🚚 Delivery Time Prediction App")
st.write("Enter order details below to estimate delivery time.")


# === Numeric inputs ===
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, step=1)
agent_rating = st.slider("Agent Rating", 0.0, 5.0, 4.5, 0.1)
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
order_hour = st.slider("Order Hour (0–23)", 0, 23, 12)
order_day = st.slider("Order Day of Week (0=Mon)", 0, 6, 3)
order_month = st.slider("Order Month (1–12)", 1, 12, 3)
pickup_minutes = st.number_input("Order to Pickup Minutes", min_value=0.0, step=1.0)

# === Build input dictionary with numeric features ===
input_dict = {
    "Agent_Age": agent_age,
    "Agent_Rating": agent_rating,
    "Distance_km": distance,
    "Order_Hour": order_hour,
    "Order_DayOfWeek": order_day,
    "Order_Month": order_month,
    "Order_to_Pickup_Minutes": pickup_minutes,
}

# === Get model feature names ===
trained_columns = model.get_booster().feature_names

# === Categorical prefixes you want multi-select ===
multi_select_prefixes = ["Weather", "Category", "Vehicle", "Traffic"]

for prefix in multi_select_prefixes:
    # Get all dummy columns for this prefix
    options = [c for c in trained_columns if c.startswith(prefix + "_")]
    # Get readable names
    option_names = [c.replace(prefix + "_", "") for c in options]
    
    # Multi-select for user
    selected = st.multiselect(f"{prefix}:", option_names)
    
    # Fill input_dict for one-hot encoding
    for col in options:
        input_dict[col] = 1 if col.replace(prefix + "_", "") in selected else 0

# Convert to DataFrame
X_input = pd.DataFrame([input_dict])

# Ensure all model columns exist
for col in trained_columns:
    if col not in X_input.columns:
        X_input[col] = 0

# Reorder columns to match training
X_input = X_input[trained_columns]

# === Prediction ===
if st.button("Predict Delivery Time"):
    prediction = model.predict(X_input)
    st.success(f"⏱ Estimated Delivery Time: {prediction[0]:.2f} minutes")