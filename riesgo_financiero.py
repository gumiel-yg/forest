import pandas as pd
import stream as st 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selectiion import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.matrics import classification_report, confusion_matrix

#Mostrar o cargar los datos
#ds = pd.read_csv("dataset_riesgo_financiero.csv")

#Colocar un titulo principal en la pagina web
st.title("Predicción de Riesgo Financiero")

#Cargar los datos en la memoria CACHE para mejorar la velocidad del acceso al conjunto de datos
@st.cache_data

#Hacemos una función que se llama cargar_datos. leemos el archivo en una variable y 
#retornamos la variable al que llama a la función. En esta caso la varible que 
#queremos retornamos se llama "ds" , abreviatura de "dataset".

def cargar_datos():
    ds = pd.read_csv("dataset_riesgo_financiero.csv")
    return ds

ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())


