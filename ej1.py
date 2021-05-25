# ------------------------------------------
# Tarea 2 - Inteligencia Artificial (01-2021)
# Ejercicio 1: Clasificador y predictor de forma separada
# Sebastián Ignacio Toro Severino (sebastian.toro1@mail.udp.cl)
# ------------------------------------------
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Creación del directorio outputs en caso de que no exista
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

scaler = StandardScaler()
output_file = open('./outputs/results_ej1.txt', 'w')
DATASET_FILEPATH = './datasets/nasa.csv'

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

# Función para imprimir por terminal (y en archivo) las características del dataset
def print_dataset_info(dataset):
    print_both(output_file, '\n --------------------------------------- Dataset ---------------------------------------')
    print_both(output_file, '* Columnas seleccionadas: ')
    print_both(output_file, list(dataset.columns))
    print_both(output_file, '* Dimensión de la matriz: ' + str(dataset.shape))
    print_both(output_file, '* Resumen del dataset:')
    print_both(output_file, dataset.head())
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


# Función para imprimir texto en terminal y además almacenar en un archivo
def print_both(file, *args):
    to_print = ' '.join([str(arg) for arg in args])
    print(to_print)
    file.write(to_print + '\n')


if __name__ == '__main__':
    columns = [3, 4, 13, 16, 24, 29, 31, 32, 37, 39] # Índices de las columnas a filtrar
    dataset = pd.read_csv(DATASET_FILEPATH, usecols=columns) # Se carga el dataset según las columnas
    x_train, x_test, y_train, y_test = split_dataset(dataset)

    # Se estandarizan los valores de los sets de entrenamiento y pruebas
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)