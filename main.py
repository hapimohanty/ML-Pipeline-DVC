from fastapi import FastAPI
import pandas as pd 
import pickle
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction API",
    description="An API to predict water potability using a trained RandomForest model",
)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file) 
@app.get("/")
def read_root():
    return {"message": "Welcome to the Water Potability Prediction API!"}       
@app.post("/predict/")
def predict_potability(Water_data: Water):
    input_data = pd.DataFrame({
        'ph': [Water_data.ph],
        'Hardness': [Water_data.Hardness],  
        'Solids': [Water_data.Solids],
        'Chloramines': [Water_data.Chloramines],
        'Sulfate': [Water_data.Sulfate],
        'Conductivity': [Water_data.Conductivity],  
        'Organic_carbon': [Water_data.Organic_carbon],
        'Trihalomethanes': [Water_data.Trihalomethanes],
        'Turbidity': [Water_data.Turbidity]
    })
    prediction = model.predict(input_data)
    print(prediction)
    if prediction == 1:
        return "Water is Consumable"
    else:
        return "Water is Not Consumable"
    #potability = "Water is Consumable" if prediction[0] == 1 else "Water is Not Consumable"
   # return {"prediction": potability}

