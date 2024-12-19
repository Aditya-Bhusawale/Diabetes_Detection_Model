import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv(r"C:\Users\bhusa\Desktop\Diabeties\diabetes.csv")

x=df.drop('Outcome',axis=1)
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

regression=LogisticRegression()
sv=regression.fit(x_train,y_train)

pickle.dump(sv,open('diabetes.pkl','wb'))