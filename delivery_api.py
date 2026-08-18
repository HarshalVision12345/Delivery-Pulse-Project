from fastapi import FastAPI
import joblib
import numpy as np 
import pandas as pd 
from pydantic import BaseModel

app = FastAPI(title="Delivery Pulse API")

model = joblib.load("model.pkl")

class DeliveryData(BaseModel):
    Delivery_person_Age : float
    Delivery_person_Ratings : float
    Weather_conditions : str
    Road_traffic_density : str
    Vehicle_condition : int
    multiple_deliveries : float
    City : str
    day_of_week: int
    is_weekend: int
    order_hour: int
    prep_time_min: float

@app.post("/predict")
def predict(data:DeliveryData):

    # CONVERT INPUT DATA TO DATA FRAME -------------->
    data_dict = data.dict()

    input_df = pd.DataFrame([data_dict])

    # Predict directly (the loaded pipeline handles preprocessing automatically)
    prediction = model.predict(input_df)

    return {"predicted_time": round(float(prediction[0]), 2)}