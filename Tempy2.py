import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la temperatura  ''')
st.image("ifoto.png", caption="Diferentes temperaturas")

st.header('Datos')

def user_input_features():
  # Entrada
  Ciudad = st.number_input('Ciudad (Acapulco=0, Acuña=1, Aguascalientes=2):',  min_value=0, max_value=2, value = 0)
  Año = st.number_input('Año', min_value=0, max_value=2020, value = 0)
  Mes = st.number_input('Circunferencia de la cintura (en cm):', min_value=1, max_value=12, value = 0.0)
  



  user_input_data = {'City': Ciudad,
                     'Year': Año,
                     'Month': Mes
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('PRIMEROexamenFINAL.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615160)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['Year'] + b1[2]*df['Month']


st.subheader('Temperatura')
st.write('La temperatura es ', prediccion)
