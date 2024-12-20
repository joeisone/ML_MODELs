from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
from Models import Bank

app=FastAPI()

#loading ML models
B=joblib.load('Bankruptcy_Model')
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
def Bankruptcy(data:Bank):
    test_data=[[
        data.company_name,
        data.year,
        data.Current_Assets,
        data.COGS,
        data.Depreciation__Amortization,
        data.EDITDA,
        data.Inventory,
        data.Net_Income,
        data.Receivables,
        data.MV,
        data.Net_Sales,
        data.TA,
        data.TL_Debt,
        data.EBIT,
        data.GR,
        data.TCL,
        data.RETAINED_EARNINGS,
        data.REVENUE,
        data.TL,
        data.Operating_Expense,

    ]]
    print (B.predict([[data.company_name,data.year,data.Current_Assets,
        data.COGS,data.Depreciation__Amortization,data.EDITDA,data.Inventory,data.Net_Income,
        data.Receivables,data.MV,data.Net_Sales,data.TA,data.TL_Debt,data.EBIT,data.GR,data.TCL,
        data.RETAINED_EARNINGS,data.REVENUE,data.TL,
        data.Operating_Expense]]))

    print('Hello')
    prediction=B.predict([[data.company_name,data.year,data.Current_Assets,
        data.COGS,data.Depreciation__Amortization,data.EDITDA,data.Inventory,data.Net_Income,
        data.Receivables,data.MV,data.Net_Sales,data.TA,data.TL_Debt,data.EBIT,data.GR,data.TCL,
        data.RETAINED_EARNINGS,data.REVENUE,data.TL,
        data.Operating_Expense]])
    if (prediction[0]==1):
        prediction="Potential insolvency"
    else:
        prediction='Likely solvent'
    return {
          'prediction':prediction
      }



# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__=='main':
    uvicorn.run(app,host='127.0.0.1',port=8000)

# uvicorn Server_2:app --reload
