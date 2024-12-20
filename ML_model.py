import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
import json


# Classification Bankruptcy model

df1=pd.read_csv(r"C:\Users\joseph.boateng\OneDrive - Minerals Income Investment Fund\Documents\Sets\american_bankruptcy.csv")
print(df1.to_string())

# Step 1:labelling
le=LabelEncoder()
df1['status_label']=le.fit_transform(df1['status_label'])
df1['company_name']=le.fit_transform(df1['company_name'])
print(df1)

#step 2:Renaming columns
df1['Current_Assets']=df1['X1']
df1['COGS']=df1['X2']
df1['Depreciation__Amortization']=df1['X3']
df1['EDITDA']=df1['X4']
df1['Inventory']=df1['X5']
df1['Net_Income']=df1['X6']
df1['Receivables']=df1['X7']
df1['MV']=df1['X8']
df1['Net_Sales']=df1['X9']
df1['TA']=df1['X10']
df1['TL_Debt']=df1['X11']
df1['EBIT']=df1['X12']
df1['GR']=df1['X13']
df1['TCL']=df1['X14']
df1['RETAINED_EARNINGS']=df1['X15']
df1['REVENUE']=df1['X16']
df1['TL']=df1['X17']
df1['Operating_Expense']=df1['X18']
print(df1)

#Drop some columns
df1=df1.drop(['X1','X2','X3','X4','X5','X6',
              'X7','X8','X9','X10','X11','X12',
              'X13','X14','X15','X16','X17','X18'],axis='columns')
print(df1)

#feature enginearing using Z score
df1['Z-score']=(df1['REVENUE']-df1['REVENUE'].mean())/df1['REVENUE'].std()
print(df1)
upper=df1[df1['Z-score']>=3]
lower=df1[df1['Z-score']<-0.3]
print(upper)
print(lower)
#print(df4[df4['price_per_sqft']>upper_limit])
#new dataframe
df1=df1[(df1['Z-score']>-0.3) & (df1['Z-score']<3)]
print(df1)


# spliting data
x=df1.drop(['status_label','Z-score'],axis='columns')
y=df1['status_label']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)

# Logistic Model
Log=LogisticRegression()
Log.fit(x_train,y_train)
print(Log.score(x_test,y_test))

# Knearestneighbor
Kn=KNeighborsClassifier()
Kn.fit(x_train,y_train)
print(Kn.score(x_test,y_test))

#save ML Model
joblib.dump(Kn, 'Bankruptcy_Model')
#Load ML Model
mj=joblib.load('Bankruptcy_Model')

# DecisionTree
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
print(DT.score(x_test,y_test))

# RandomForestTree
RT=RandomForestClassifier()
RT.fit(x_train,y_train)
print(RT.score(x_test,y_test))

# SVM
S=SVC()
S.fit(x_train,y_train)
print(S.score(x_test,y_test))

# K nearest neighbor is the highest scoring model for the dataset.


df2=pd.read_csv(r"C:\Users\joseph.boateng\OneDrive - Minerals Income Investment Fund\Documents\Sets\flight\Clean_Dataset.csv")
print(df2.to_string())

le2=LabelEncoder()
df2['airline']=le2.fit_transform(df2['airline'])
df2['flight']=le2.fit_transform(df2['flight'])
df2['source_city']=le2.fit_transform(df2['source_city'])
df2['departure_time']=le2.fit_transform(df2['departure_time'])
df2['stops']=le2.fit_transform(df2['stops'])
df2['arrival_time']=le2.fit_transform(df2['arrival_time'])
df2['destination_city']=le2.fit_transform(df2['destination_city'])
df2['class']=le2.fit_transform(df2['class'])
print(df2.to_string())

#feature engineering
df2['Z-score']=(df2['price']-df2['price'].mean())/df2['price'].std()
lower2=df2[df2['Z-score']<-0.85]
upper2=df2[df2['Z-score']>3]
print(lower2)
print(upper2)
df2=df2[(df2['Z-score']>-0.85)&(df2['Z-score']<3)]
df2=df2.drop(['Unnamed: 0','Z-score'],axis='columns')
print(df2)
# save the data into json
columns={'data_columns':[col.lower() for col in df2.columns]

}
with open ('columns.json','w') as f:
    f.write(json.dumps(columns))

x2=df2.drop('price',axis='columns')
y2=df2['price']


x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=20)

#Reg

Reg=LinearRegression()
Reg.fit(x2_train,y2_train)
print(Reg.score(x2_test,y2_test))

joblib.dump(Reg,'Price_prediction')

Pred=joblib.load('Price_prediction')

mj=joblib.load('Bankruptcy_Model')
print(Pred.predict([[4,1408,2,1,1,2,3,1,2.33,1]]))
print(mj.predict([[2,2010,500.1,600.1,
                   20.2,90.4,400.2,53,
                   100.3,250.4,503.3,
                   400.4,110.1,40.4,200.1,420.4,440.4
                   ,320.0,200.0,100.0]]))
