# Importamos las librerías que vamos a usar
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os
from train_model import train_model

# Creamos la aplicación Flask
app = Flask(__name__)

# Aquí guardaremos el modelo para usarlo después
modelo = None

# Función para cargar el modelo que ya entrenamos
def cargar_modelo():
    global modelo
    try:
        # Primero revisamos si ya tenemos el archivo con el modelo entrenado
        if not os.path.exists('modelo_rendimiento.pkl'):
            print("No encontramos el modelo, así que vamos a entrenar uno nuevo...")
            modelo = train_model()  # Entrenamos el modelo
        else:
            # Si el archivo está, lo cargamos para usarlo
            with open('modelo_rendimiento.pkl', 'rb') as archivo:
                modelo = pickle.load(archivo)
            print("Modelo cargado correctamente, listo para usar")
    except Exception as e:
        print(f"Hubo un error al cargar el modelo: {e}")
        modelo = None

# Esta función nos ayuda a interpretar la nota predicha
# Nos dice qué nivel de rendimiento tiene el estudiante y da recomendaciones
def evaluar_rendimiento(nota):
    if nota >= 17:
        return {
            'nivel': 'Excelente',
            'clase': 'excelente',
            'mensaje': 'El estudiante tiene un rendimiento sobresaliente.',
            'recomendacion': 'Puedes darle trabajos o proyectos más avanzados para que siga motivado.'
        }
    elif nota >= 14:
        return {
            'nivel': 'Bueno',
            'clase': 'bueno',
            'mensaje': 'El estudiante está rindiendo bien.',
            'recomendacion': 'Que mantenga el ritmo y refuerce las áreas donde tenga dificultades.'
        }
    elif nota >= 10.5:
        return {
            'nivel': 'Regular',
            'clase': 'regular',
            'mensaje': 'El estudiante está en el mínimo para aprobar.',
            'recomendacion': 'Es bueno que dedique más tiempo al estudio y complete todas las tareas.'
        }
    else:
        return {
            'nivel': 'Insuficiente',
            'clase': 'insuficiente',
            'mensaje': 'El estudiante no alcanza el nivel mínimo para aprobar.',
            'recomendacion': 'Necesita un plan urgente de mejora, con tutorías y seguimiento cercano.'
        }

# Aquí definimos la página principal donde está el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar los datos que nos envían desde el formulario y hacer la predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        try:
            # Primero nos aseguramos de que el modelo esté cargado
            if modelo is None:
                cargar_modelo()
                if modelo is None:
                    return render_template('index.html', error="No pudimos cargar el modelo, inténtalo más tarde.")
            
            # Tomamos los datos que nos mandan desde el formulario
            horas_estudio = float(request.form['horas_estudio'])
            asistencia = float(request.form['asistencia'])
            trabajos_completados = float(request.form['trabajos_completados'])
            nota_parcial = float(request.form['nota_parcial'])
            
            # Validamos que los datos estén dentro de un rango lógico
            if not (0 <= horas_estudio <= 40):
                return render_template('index.html', error="Las horas de estudio deben estar entre 0 y 40.")
            if not (0 <= asistencia <= 100):
                return render_template('index.html', error="La asistencia debe estar entre 0% y 100%.")
            if not (0 <= trabajos_completados <= 10):
                return render_template('index.html', error="Los trabajos completados deben estar entre 0 y 10.")
            if not (0 <= nota_parcial <= 20):
                return render_template('index.html', error="La nota parcial debe estar entre 0 y 20.")
            
            # Preparamos los datos para enviarlos al modelo
            datos_entrada = np.array([
                horas_estudio, asistencia, trabajos_completados, nota_parcial
            ]).reshape(1, -1)  # Aquí lo dejamos listo en formato que el modelo entiende
            
            # Hacemos la predicción con el modelo
            prediccion = modelo.predict(datos_entrada)[0]  # [0] porque solo es un dato
            
            # Nos aseguramos que la nota esté dentro de 0 y 20
            prediccion = max(0, min(20, prediccion))
            prediccion_redondeada = round(prediccion, 1)
            
            # Evaluamos qué significa esa nota para el rendimiento
            evaluacion = evaluar_rendimiento(prediccion)
            
            # Guardamos los datos que nos enviaron para mostrarlos de nuevo
            datos_ingresados = {
                'horas_estudio': horas_estudio,
                'asistencia': asistencia,
                'trabajos_completados': trabajos_completados,
                'nota_parcial': nota_parcial
            }
            
            # Finalmente mostramos la página con el resultado
            return render_template(
                'index.html',
                prediccion=prediccion_redondeada,
                nivel=evaluacion['nivel'],
                mensaje=evaluacion['mensaje'],
                recomendacion=evaluacion['recomendacion'],
                clase_resultado=evaluacion['clase'],
                datos=datos_ingresados
            )
            
        except Exception as e:
            # Si algo falla, mostramos el error para que lo puedas ver
            return render_template('index.html', error=f"Error en la predicción: {str(e)}")
    
    # Si intentan acceder por otro método que no sea POST, los mandamos al inicio
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Al arrancar la app, cargamos el modelo para tenerlo listo
    cargar_modelo()
    app.run(debug=True)

