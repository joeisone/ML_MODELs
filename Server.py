from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
from Models import Price
app=FastAPI()
#loading ML models
P=joblib.load('Price_prediction')

@app.get('/')
def index():
    return {'message':'Hello, stranger'}

@app.get('/{name}')
def get_name(name:str):
    return {'message':f'Hello,{name}'}

@app.get("/hello")
async def hello():
    return "welcome"


@app.post('/predict')
def predict_price(data:Price):
    test_data=[[
        data.airline,
        data.flight,
        data.source_city,
        data.departure_time,
        data.stops,
        data.arrival_time,
        data.destination_city,
        data.c_lass,
        data.duration,
        data.days_left,

    ]]
    print (P.predict([[data.airline,data.flight,data.source_city,
                       data.departure_time,data.stops,data.arrival_time,
                       data.destination_city,data.c_lass,data.duration,
                       data.days_left]]))

    print('Hello')
    prediction=P.predict([[data.airline,data.flight,data.source_city,
                       data.departure_time,data.stops,data.arrival_time,
                       data.destination_city,data.c_lass,data.duration,
                       data.days_left]])
    if (prediction[0]>7000):
        prediction="High price ticket"
    else:
        prediction='Low price ticket'
    return {
          'prediction':prediction
      }



# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__=='main':
    uvicorn.run(app,host='127.0.0.1',port=8000)

# uvicorn Server:app --reload
