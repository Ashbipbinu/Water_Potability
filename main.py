from fastapi import FastAPI
import pandas as pd
import pickle

from data_interface import WaterModel  

app = FastAPI(
    title="Water Potability Prediction",    
    description="Prdicting the potability of the water"
) 

with open(r'H:\Boston-Housing-Price-MLOPS\model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return "Hello"

@app.post('/predict')
def model_prediction(water: WaterModel):
    sample = pd.DataFrame({
        "ph" : [water.ph],
        "Hardness" : [water.Hardness] ,
        "Solids" : [water.Solids],
        "Chloramines" : [water.Chloramines],
        "Sulfate" : [water.Sulfate],
        "Conductivity" : [water.Conductivity],
        "Organic_carbon" : [water.Organic_carbon],
        "Trihalomethanes" : [water.Trihalomethanes],
        "Turbidity" : [water.Turbidity],
    })

    predicted_val = model.predict(sample) 

    if predicted_val == 1:
        return "Water is consumable"
    else:
        return "Water is not consumable"