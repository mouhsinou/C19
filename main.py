from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the trained model
model = joblib.load('model_rf.joblib')

app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    D1: float
    D2: float
    D3: float
    D4: float
    D5: float
    D6: float
    D7: float
    D8: float
    D9: float
    D10: float
    D11: float
    D12: float
    D13: float
    D14: float
    D15: float
    D16: float
    D17: float
    D18: float
    D19: float
    D20: float
    D21: float
    D22: float
    D23: float
    D24: float
    D25: float
    D26: float
    D27: float
    D28: float
    D29: float
    D30: float
    D31: float
    D32: float
    D33: float
    D34: float
    D35: float
    D36: float
    D37: float
    D38: float
    D39: float
    D40: float
    D41: float
    D42: float
    D43: float
    D44: float
    D45: float
    D46: float
    D47: float
    D48: float
    D49: float
    D50: float
    D51: float
    D52: float
    D53: float
    D54: float
    D55: float
    D56: float
    D57: float
    D58: float
    D59: float
    D60: float
    D61: float
    D62: float
    D63: float
    D64: float

@app.post("/predict/")
async def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return {"prediction": prediction[0]}

@app.post("/predict_batch/")
async def predict_batch(data: list[InputData]):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([item.dict() for item in data])
    
    # Make predictions
    predictions = model.predict(input_df)
    
    return {"predictions": predictions.tolist()}
