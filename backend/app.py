from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import pandas as pd
from werkzeug.utils import secure_filename
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Crear carpeta uploads si no existe
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Cargar los modelos entrenados
log_model = joblib.load('log_model.pkl')
nn_model = joblib.load('nn_model.pkl')
scaler = joblib.load('scaler.pkl')  

# Columnas esperadas del dataset Wisconsin Breast Cancer (Principales)
COLUMNAS_PRINCIPALES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean'
]

# Columnas esperadas del dataset Wisconsin Breast Cancer (Todas)
COLUMNAS_ESPERADAS = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Obtener las 30 características del formulario
            data = [float(request.form.get(col, 0)) for col in COLUMNAS_ESPERADAS]

            # Escalar los datos
            datos_escalados = scaler.transform([data])

            # Predicciones
            pred_log = log_model.predict(datos_escalados)[0]
            prob_log = log_model.predict_proba(datos_escalados)[0][1]
            
            pred_nn = nn_model.predict(datos_escalados)[0]

            # Asegurar que las probabilidades estén entre 0-1
            prob_nn = nn_model.predict_proba(datos_escalados)[0][1]

            nivel_log = 'Maligno' if pred_log == 1 else 'Benigno'
            nivel_nn = 'Maligno' if prob_nn >= 0.5 else 'Benigno'

            return render_template(
                "predict.html",
                resultado=True,
                log_pred=int(pred_log),
                nn_pred=int(1 if prob_nn >= 0.5 else 0),
                log_prob=round(prob_log * 100, 2),
                nn_prob=round(prob_nn * 100, 2),
                log_nivel=nivel_log,
                nn_nivel=nivel_nn
            )
        except Exception as e:
            return render_template('predict.html', error=f"Error: {str(e)}")
        
    return render_template('predict.html')

# Predicción por lotes
@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        try:
            # Validar que hay un archivo
            if 'file' not in request.files:
                return render_template('batch.html', error='No se seleccionó archivo')
            
            file = request.files['file']
            if file.filename == '':
                return render_template('batch.html', error='Archivo vacío')
            
            if not file.filename.endswith('.csv'):
                return render_template('batch.html', error='Solo se aceptan archivos CSV')
            
            # Guardar y leer archivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Leer datos
            # Leer el archivo
            datos = pd.read_csv(filepath, header=None)

            # Si el archivo tiene 32 columnas (como el original de UCI que trae ID y Diagnóstico)
            if datos.shape[1] == 32:
                # Le ponemos los nombres ignorando las primeras 2 columnas (ID y Diagnóstico)
                columnas_uci = ['id', 'diagnosis'] + COLUMNAS_ESPERADAS
                datos.columns = columnas_uci
            # Si el archivo tiene 30 columnas (solo los datos para predecir)
            elif datos.shape[1] == 30:
                datos.columns = COLUMNAS_ESPERADAS
            
            # Validar columnas (todas las 30 características)
            columnas_faltantes = [col for col in COLUMNAS_ESPERADAS if col not in datos.columns]
            if columnas_faltantes:
                os.remove(filepath)
                return render_template('batch.html', error=f"El archivo CSV debe contener las 30 columnas requeridas. Faltan: {', '.join(columnas_faltantes)}")
            
            # Escalar datos
            datos_escalados = scaler.transform(datos[COLUMNAS_ESPERADAS])
            
            # Hacer predicciones
            pred_log = log_model.predict(datos_escalados)
            pred_nn = nn_model.predict(datos_escalados).flatten()
            
            # Redimensionar si es necesario
            if pred_nn.ndim > 1:
                pred_nn = pred_nn[:, 0]
            
            prob_log = log_model.predict_proba(datos_escalados)[:, 1]
            prob_nn = np.where(pred_nn <= 1, pred_nn, pred_nn / 100)
            
            # Convertir predicciones NN a 0/1
            pred_nn_binary = np.where(prob_nn >= 0.5, 1, 0)
            
            # Crear dataframe de resultados
            resultados = datos.copy()
            resultados['prediccion_log'] = pred_log
            resultados['prediccion_nn'] = pred_nn_binary
            resultados['prob_log'] = (prob_log * 100).round(2)
            resultados['prob_nn'] = (prob_nn * 100).round(2)
            resultados['diagnostico_log'] = np.where(pred_log == 1, 'Maligno', 'Benigno')
            resultados['diagnostico_nn'] = np.where(prob_nn >= 0.5, 'Maligno', 'Benigno')
            
            # Guardar resultados
            resultado_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.csv')
            resultados.to_csv(resultado_path, index=False)
            
            # Calcular métricas solo si hay columna de diagnosis
            metricas = None
            if 'diagnosis' in datos.columns:
                # Convertir M/B a 1/0
                y_true = (datos['diagnosis'] == 'M').astype(int).values
                
                # Calcular matrices de confusión
                cm_log = confusion_matrix(y_true, pred_log)
                cm_nn = confusion_matrix(y_true, pred_nn_binary)
                
                # Calcular métricas para Regresión Logística
                metricas_log = {
                    'precision': precision_score(y_true, pred_log),
                    'recall': recall_score(y_true, pred_log),
                    'f1': f1_score(y_true, pred_log),
                    'accuracy': accuracy_score(y_true, pred_log),
                    'cm': cm_log.tolist()
                }
                
                # Calcular métricas para Red Neuronal
                metricas_nn = {
                    'precision': precision_score(y_true, pred_nn_binary),
                    'recall': recall_score(y_true, pred_nn_binary),
                    'f1': f1_score(y_true, pred_nn_binary),
                    'accuracy': accuracy_score(y_true, pred_nn_binary),
                    'cm': cm_nn.tolist()
                }
                
                metricas = {
                    'log': metricas_log,
                    'nn': metricas_nn
                }
            
            # Limpiar archivo original
            os.remove(filepath)
            
            return render_template(
                'batch.html',
                resultado=True,
                tabla=resultados.head(20).to_html(classes='table table-striped'),
                total_registros=len(resultados),
                positivos_log=int(sum(pred_log)),
                positivos_nn=int(sum(pred_nn_binary)),
                metricas=metricas
            )
            
        except Exception as e:
            return render_template('batch.html', error=f'Error procesando archivo: {str(e)}')
    
    return render_template('batch.html')

# Descargar resultados
@app.route('/descargar-resultados')
def descargar_resultados():
    try:
        resultado_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.csv')
        if os.path.exists(resultado_path):
            return send_file(resultado_path, as_attachment=True, download_name='resultados_diagnostico.csv')
        else:
            return "No hay resultados disponibles", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

# 🔹 Ejecutar app
if __name__ == "__main__":
    app.run(debug=True)