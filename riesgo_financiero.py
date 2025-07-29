import pandas as pd 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Colocar un título principal en la página Web
st.title("Predicción de Riesgo Financiero")

# Cargar los datos en la memoria CACHE para mejorar la velocidad del acceso al conjunto de datos
@st.cache_data
def cargar_datos():
    ds = pd.read_csv("dataset_riesgo_financiero.csv")
    return ds

# Cargar y mostrar los datos
ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())
