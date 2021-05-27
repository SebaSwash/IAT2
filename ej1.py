# ------------------------------------------
# Tarea 2 - Inteligencia Artificial (01-2021)
# Ejercicio 1: Clasificador y predictor de forma separada
# Sebastián Ignacio Toro Severino (sebastian.toro1@mail.udp.cl)
# ------------------------------------------
import os
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Creación del directorio outputs en caso de que no exista
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

output_file = open('./outputs/results_ej1.txt', 'w')
DATASET_FILEPATH = './datasets/nasa.csv'

# Función para imprimir por terminal (y en archivo) las características del dataset
def print_dataset_info(dataset):
    print_both(output_file, '\n --------------------------------------- Dataset ---------------------------------------')
    print_both(output_file, '* Columnas seleccionadas: ')
    print_both(output_file, list(dataset.columns))
    print_both(output_file, '* Dimensión de la matriz: ' + str(dataset.shape))
    print_both(output_file, '* Resumen del dataset:')
    print_both(output_file, dataset)
    print_both(output_file, '')

# Función para imprimir por terminal (y en archivo) los distintos set de entrenamiento y prueba
def print_train_test_sets(x_train, x_test, y_train, y_test):
    print_both(output_file, '-------------------------- Set de entrenamiento --------------------------')
    print_both(output_file, x_train)
    print_both(output_file, y_train)
    print_both(output_file, '')
    print_both(output_file, '-------------------------- Set de prueba --------------------------')
    print_both(output_file, x_test)
    print_both(output_file, y_test)
    print_both(output_file, '')

# Función para separar el dataset en 2 arreglos con características y etiquetas para luego
# particionar las filas en un set de entrenamiento y uno de pruebas
def split_dataset(dataset):
    # Se obtiene la cantidad de columnas según el dataset cargado
    columns_length = dataset.shape[1]

    # Se separa el dataset para generar 2 listas.
    # Donde 'x' contiene las características, mientras que 'y' contiene las etiquetas
    # En este caso, la etiqueta corresponde a si el asteroide es peligroso o no
    x = dataset.iloc[:, 0:columns_length - 1] 
    y = dataset.iloc[:, columns_length - 1]

    # Posteriormente se separa nuevamente el dataset para obtener 2 conjuntos,
    # uno de entrenamiento y uno de pruebas (teniendo una cantidad de 50% de las filas del dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

    return x_train, x_test, y_train, y_test


# Función para imprimir texto en terminal y además almacenar en un archivo
def print_both(file, *args):
    to_print = ' '.join([str(arg) for arg in args])
    print(to_print)
    file.write(to_print + '\n')

# Función para imprimir datos de la instancia de la regresión lineal
def print_regression_stats(regression, y_pred):
    print_both(output_file, '')
    print_both(output_file, '--------- Datos obtenidos de la regresión lineal ---------')
    print_both(output_file, '- Coeficientes del hiperplano: ' + str(regression.coef_))
    print_both(output_file, '- Valor de intersección: ' +str(regression.intercept_))
    print_both(output_file, '- Score del modelo: ' + str(regression.score(x_train, y_train)))
    # Se imprime el error cuadrático medio obtenido
    mse = mean_squared_error(y_test, y_pred)
    print_both(output_file, '- Error cuadrático medio: ' +str(mse))
    print_both(output_file, '')

    print_both(output_file, '-------------------------- Comparación de valores (obtenido vs real) --------------------------')

    print_both(output_file, '')
    print_both(output_file, 'Etiqueta predicha vs Etiqueta real')

    for predicted, real in zip(y_pred, y_test):
        print_both(output_file, str(predicted) + '\t' + str(real))
    
    print_both(output_file, '')


if __name__ == '__main__':
    columns = [3, 4, 13, 16, 24, 29, 31, 32, 37] # Índices de las columnas a filtrar
    dataset = pd.read_csv(DATASET_FILEPATH, usecols=columns) # Se carga el dataset según las columnas

    # Se imprimen las características principales del dataset y el resumen del mismo
    print_dataset_info(dataset)

    # Se realiza la clasificación con todos los datos del dataset para obtener las etiquetas
    # entregadas por el modelo

    # Se mide el tiempo de ejecución para el proceso de clasificación y predicción
    clustering_start_time = time.time()

    clustering = KMeans(n_clusters=2)
    clustering.fit(dataset) # Se ejecuta el algoritmo de clustering

    # Se marca el tiempo de término del clasificador
    clustering_end_time = time.time()

    # Se obtienen las etiquetas asignadas a cada fila del dataset y se imprimen
    print_both(output_file, '--------- Distribución de etiquetas obtenidas ---------')
    print_both(output_file, '- Cluster 0: ' + str(list(clustering.labels_).count(0)))
    print_both(output_file, '- Cluster 1: ' + str(list(clustering.labels_).count(1)))
    print_both(output_file, '')

    # Posteriormente, se agrega al dataset la columna obtenida de la clasificación
    dataset['Cluster'] = list(clustering.labels_)
    print_both(output_file, '--------- Dataset actualizado con etiquetas obtenidas ---------')
    print_dataset_info(dataset)
    print_both(output_file, '')

    # Posteriormente, se realiza una separación del dataset actualizado para
    # utilizar el modelo de predicción
    # En este caso y_train e y_test corresponden a la columna obtenida del modelo de clasificación previo
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    print_train_test_sets(x_train, x_test, y_train, y_test)

    # Se crea el modelo de regresión en base al set de entrenamiento generado

    # Se marca el tiempo de inicio del modelo de regresión
    regression_start_time = time.time()

    regression = LinearRegression()
    regression.fit(x_train, y_train)

    # Se obtiene la predicción en base al set de pruebas
    y_pred = regression.predict(x_test)

    regression_end_time = time.time()
    
    # Se imprimen los datos obtenidos de la regresión modelada
    print_regression_stats(regression, y_pred)

    # Se imprimen los tiempos de ejecución obtenidos
    print_both(output_file, '- Tiempo de ejecución total (clasificación + predicción): ' + str(regression_end_time - clustering_start_time) + ' seg')
    print_both(output_file, '- TIempo de ejecución de la clasificación: ' + str(clustering_end_time - clustering_start_time) + ' seg')
    print_both(output_file, '- Tiempo de ejecución de la predicción: ' + str(regression_end_time - regression_start_time) + ' seg')
    