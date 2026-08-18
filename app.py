import numpy as np 
import pandas as pd 
import streamlit as st 
import requests
from PIL import Image

st.title("Delivery Pulse Project")

Delivery_Img = Image.open(r"C:\Users\HARSHAL\OneDrive\Desktop\Delivery Pulse\Delivery.png")
st.image(Delivery_Img)

with st.expander("🛵 🚴‍♂️  About Delivery Pulse Dataset  🚴‍♂️ 🛵"):
    st.markdown("""
    ### 📊 Delivery Pulse Dataset Overview
    
    **Delivery Pulse** is an end-to-end Machine Learning system designed to predict food and package delivery times (in minutes) based on real-time operational and environmental variables.

    ---

    #### 🔑 Key Features in the Dataset:
    * **Courier Profile:** Driver Age, Performance Rating, Vehicle Condition Code.
    * **Order Constraints:** Number of Multiple Deliveries assigned in a single trip.
    * **Environmental Factors:** Weather Conditions (*Sunny, Fog, Sandstorms, etc.*) and Road Traffic Density (*Low, Medium, High, Jam*).
    * **Geography:** City Type (*Metropolitian, Urban, Semi-Urban*).
    * **Temporal Features:** Day of the Week, Weekend Indicator, Order Hour of the day, and Kitchen Prep Time (Minutes).

    ---

    #### 🎯 Objective & Business Impact:
    * **Target Variable:** Total estimated delivery time in minutes.
    * **Value Proposition:** Optimizes delivery route planning, sets accurate customer ETA expectations, and reduces order delays for logistics platforms.
    """)


with st.expander("Click To Open Prediction Form ........"):
    st.header("⏱️ Predict Delivery Time .....")

    # Simple Inputs
    age = st.number_input("Driver Age", value=25)
    rating = st.slider("Driver Rating", 1.0, 5.0, 4.5)
    vehicle = st.selectbox("Vehicle Condition", [0, 1, 2])
    deliveries = st.selectbox("Multiple Deliveries", [0.0, 1.0, 2.0])

    weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
    city = st.selectbox("City", ["Metropolitian", "Urban", "Semi-Urban"])

    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
    prep_time = st.number_input("Prep Time (Minutes)", value=15.0)

    if st.button('Predict'):
        input_df = {
            "Delivery_person_Age" : float(age),
            "Delivery_person_Ratings" : float(rating),
            "Weather_conditions" : weather,
            "Road_traffic_density" : traffic,
            "Vehicle_condition" : int(vehicle),
            "multiple_deliveries" : float(deliveries),
            "City" : city,
            "day_of_week": int(day_of_week),
            "is_weekend": 1 if day_of_week in [5,6] else 0,
            "order_hour": int(order_hour),
            "prep_time_min": float(prep_time)
        }

        try:
            #send request to FASTAPI
            response = requests.post("http://127.0.0.1:8000/predict", json=input_df)
            if response.status_code == 200:
                result = response.json()
                st.success(f"⏱️ Estimated Delivery Time: **{result['predicted_time']} minutes**")
            else:
                st.error(f"Error making prediction: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server. Ensure Uvicorn is running on port 8000.")
