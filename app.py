import streamlit as st
import joblib

demand_model = joblib.load("demand_model.pkl")
vehicle_model = joblib.load("vehicle_model.pkl")

le_location = joblib.load("location_encoder.pkl")
le_vehicle = joblib.load("vehicle_encoder.pkl")
le_demand = joblib.load("demand_encoder.pkl")


st.title("UBER Ride Demand Prediction")

location = st.selectbox(
    "Pickup Location",
    le_location.classes_
)

if st.button("Predict"):

    location_encoded = le_location.transform([location])[0]

    demand_pred = demand_model.predict([[location_encoded]])[0]

    demand = le_demand.inverse_transform([demand_pred])[0]

    vehicle_pred = vehicle_model.predict(
        [[location_encoded, demand_pred]]
    )[0]

    vehicle = le_vehicle.inverse_transform([vehicle_pred])[0]

    st.success(f"Demand : {demand}")
    st.success(f"Recommended Vehicle : {vehicle}")