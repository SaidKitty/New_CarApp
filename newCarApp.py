import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the model and encoder
model = joblib.load('xgboost_model_onehot2.pkl')
encoder = joblib.load('onehot_encoder.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the API request body
class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    condition: str
    mileage: float
    fuel_type: str
    volume: float
    color: str
    transmission: str
    drive_unit: str
    segment: str

# API endpoint to predict car price
@app.post("/predict")
def predict_price(car: CarFeatures):
    # Convert input to DataFrame
    input_data = pd.DataFrame([car.dict()])

    # Apply encoding
    input_data_encoded = encoder.transform(input_data)

    # Predict the price
    prediction = model.predict(input_data_encoded)
    return {"predicted_price": round(prediction[0], 2)}

# Run FastAPI in a separate thread
import threading
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()

# --------------------------------------
# Streamlit UI
st.title("Belarus Car Price Prediction")

make = st.selectbox("Make", options=['Toyota', 'Honda', 'BMW', 'Audi', 'Mercedes'])
model_name = st.selectbox("Model", options=['Fortuner', 'Civic', 'X5', 'A4', 'C-Class'])
year = st.slider("Year", min_value=1990, max_value=2024, value=2015)
condition = st.selectbox("Condition", options=['With Mileage', 'New', 'Used', 'Certified Pre-Owned'])
mileage = st.number_input("Mileage (kilometers)", value=9500.0)
fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'Electric', 'Hybrid'])
volume = st.number_input("Engine Volume (cm³)", value=1500.0)
color = st.selectbox("Color", options=['Red', 'Blue', 'Black', 'White', 'Silver'])
transmission = st.selectbox("Transmission", options=['Mechanics', 'Automatic', 'Manual'])
drive_unit = st.selectbox("Drive Unit", options=['Front-Wheel Drive', 'Rear-Wheel Drive', 'All-Wheel Drive'])
segment = st.selectbox("Segment", options=['B', 'C', 'D', 'E', 'F'])

if st.button("Predict"):
    input_data = {
        "make": make,
        "model": model_name,
        "year": year,
        "condition": condition,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "volume": volume,
        "color": color,
        "transmission": transmission,
        "drive_unit": drive_unit,
        "segment": segment
    }
    
    response = predict_price(CarFeatures(**input_data))
    predicted_price = f"${response['predicted_price']:,.2f}"
    
    st.markdown(
        f"<h1 style='color: #FF1493; font-weight: bold; font-size: 36px;'>Predicted Price (USD): {predicted_price}</h1>",
        unsafe_allow_html=True
    )
