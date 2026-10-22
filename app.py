import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    layout="wide"
)

st.title("✈ Airline Passenger Satisfaction Prediction")

# ===============================
# Load Model Files
# ===============================
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

# ===============================
# Input Layout
# ===============================
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", encoders["Gender"].classes_)
    Customer_Type = st.selectbox("Customer Type", encoders["Customer Type"].classes_)
    Age = st.number_input("Age", 0, 100, 25)
    Type_of_Travel = st.selectbox("Type of Travel", encoders["Type of Travel"].classes_)
    Class = st.selectbox("Class", encoders["Class"].classes_)
    Flight_Distance = st.number_input("Flight Distance", 0, 10000, 500)

with col2:
    Inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 3)
    Departure_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
    Ease_of_Online_booking = st.slider("Ease of Online booking", 0, 5, 3)
    Gate_location = st.slider("Gate location", 0, 5, 3)
    Food_and_drink = st.slider("Food and drink", 0, 5, 3)
    Online_boarding = st.slider("Online boarding", 0, 5, 3)
    Seat_comfort = st.slider("Seat comfort", 0, 5, 3)
    Inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    Onboard_service = st.slider("On-board service", 0, 5, 3)
    Leg_room_service = st.slider("Leg room service", 0, 5, 3)
    Baggage_handling = st.slider("Baggage handling", 0, 5, 3)
    Checkin_service = st.slider("Checkin service", 0, 5, 3)
    Inflight_service = st.slider("Inflight service", 0, 5, 3)
    Cleanliness = st.slider("Cleanliness", 0, 5, 3)
    Departure_Delay = st.number_input("Departure Delay in Minutes", 0, 1000, 0)
    Arrival_Delay = st.number_input("Arrival Delay in Minutes", 0.0, 1000.0, 0.0)

# ===============================
# Prediction
# ===============================
if st.button("Predict"):

    # ===============================
    # Rule-Based Override
    # ===============================

    rating_values = [
        Inflight_wifi_service,
        Departure_time_convenient,
        Ease_of_Online_booking,
        Gate_location,
        Food_and_drink,
        Online_boarding,
        Seat_comfort,
        Inflight_entertainment,
        Onboard_service,
        Leg_room_service,
        Baggage_handling,
        Checkin_service,
        Inflight_service,
        Cleanliness
    ]

    # If any rating is zero → dissatisfied
    if 0 in rating_values:
        result = "neutral or dissatisfied"
    else:
        # ===============================
        # Normal ML Prediction
        # ===============================
        input_dict = {
            "Gender": encoders["Gender"].transform([Gender])[0],
            "Customer Type": encoders["Customer Type"].transform([Customer_Type])[0],
            "Age": Age,
            "Type of Travel": encoders["Type of Travel"].transform([Type_of_Travel])[0],
            "Class": encoders["Class"].transform([Class])[0],
            "Flight Distance": Flight_Distance,
            "Inflight wifi service": Inflight_wifi_service,
            "Departure/Arrival time convenient": Departure_time_convenient,
            "Ease of Online booking": Ease_of_Online_booking,
            "Gate location": Gate_location,
            "Food and drink": Food_and_drink,
            "Online boarding": Online_boarding,
            "Seat comfort": Seat_comfort,
            "Inflight entertainment": Inflight_entertainment,
            "On-board service": Onboard_service,
            "Leg room service": Leg_room_service,
            "Baggage handling": Baggage_handling,
            "Checkin service": Checkin_service,
            "Inflight service": Inflight_service,
            "Cleanliness": Cleanliness,
            "Departure Delay in Minutes": Departure_Delay,
            "Arrival Delay in Minutes": Arrival_Delay
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        result = encoders["satisfaction"].inverse_transform([prediction])[0]

    # ===============================
    # Display Result
    # ===============================
    if result == "satisfied":
        st.success(f"Prediction: {result}")
    else:
        st.error(f"Prediction: {result}")

    # ===============================
    # Save History
    # ===============================
    save_df = pd.DataFrame([{
        "Gender": Gender,
        "Customer Type": Customer_Type,
        "Age": Age,
        "Type of Travel": Type_of_Travel,
        "Class": Class,
        "Flight Distance": Flight_Distance,
        "Prediction": result
    }])

    if os.path.exists("history.csv"):
        save_df.to_csv("history.csv", mode="a", header=False, index=False)
    else:
        save_df.to_csv("history.csv", index=False)


# ===============================
# View History
# ===============================
if os.path.exists("history.csv"):
    if st.checkbox("Show Prediction History"):
        history = pd.read_csv("history.csv")
        st.dataframe(history)
