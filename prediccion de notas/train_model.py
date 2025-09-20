import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

def train_model(csv_file='estudiantes.csv'):
    # 1. Cargar los Datos...
    print(f"Cargando Datos...!!!'{csv_file}'...")
    datos=pd.read_csv(csv_file)
    
    # 2. Mostrar Informacion...
    print('*'*20)
    print('ANALISIS EXPLORATORIO')
    print('*'*20)
    print('Dimensiones de los Datos: ',datos.shape)
    print('Primeras 5 Filas...')
    print(datos.head())
    print('Información de la BD...')
    print(datos.info())
    print('Resumen Estadistico...')
    print(datos.describe())
    
    # Preparar las Variables
    X=datos[['horas_estudio','asistencia','trabajos_completados','nota_parcial']]
    y=datos['nota_final']
    
    # Dividir datos en Entrenamiento y Prueba
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)
    print('Datos de Entrenamiento:',X_train.shape[0],'muestras')
    print('Datos de Prueba:',X_test.shape[0],'muestras')
    
    # Entrenar El modelo
    print('Entrenando nuestro modelo de Regresión Lineal')
    modelo=LinearRegression()
    modelo.fit(X_train,y_train)
    
    # 6. Hacer predicciones
    # ------------------------
    #y_pred = modelo.predict(X_test)

    # 7. Guardar modelo
    # ------------------------
    with open('modelo_rendimiento.pkl', 'wb') as file:
        pickle.dump(modelo, file)
    print("\nModelo guardado como 'modelo_rendimiento.pkl'")

    # 8. Gráfico comparativo
    # ------------------------
    """ plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([0, 20], [0, 20], 'r--')  # línea ideal
    plt.xlabel("Notas reales")
    plt.ylabel("Notas predichas")
    plt.title("Comparación: valores reales vs predicciones")
    plt.show() """
    
    return modelo 
    
train_model()