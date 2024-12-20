from pydantic import BaseModel

class Price(BaseModel):
    airline:int
    flight:int
    source_city:int
    departure_time:int
    stops:int
    arrival_time:int
    destination_city:int
    c_lass:int
    duration:float
    days_left:int

class Bank(BaseModel):
    company_name:int
    year:int
    Current_Assets:float
    COGS:float
    Depreciation__Amortization:float
    EDITDA:float
    Inventory:float
    Net_Income:float
    Receivables:float
    MV:float
    Net_Sales:float
    TA:float
    TL_Debt:float
    EBIT:float
    GR:float
    TCL:float
    RETAINED_EARNINGS:float
    REVENUE:float
    TL:float
    Operating_Expense:float

