from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import logging, os
logging.disable(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_MNIST_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

class Adaboost:
    #* Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        self.numClasifs = T
        self.intentosClasif = A
        self.clasificadores = []
        self.clases = range(10)
    
    #* Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose = False):
        numObservaciones, numCaracteristics = X.shape                       # Obtener el número de observaciones y de características por observación de X
        Dpesos = np.full(numObservaciones, (1 / numObservaciones))          # Iniciar pesos de las observaciones a 1/n_observaciones
        self.clasificadores = []
        
        for t in range(1,self.numClasifs+1):                                # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
            best_clasificador = None
            predicciones_mejor = None
            best_error = np.inf
            
            for f in range(1,self.intentosClasif+1):                            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir

                clasif = DecisionStump(numCaracteristics)                       # Crear un nuevo clasificador débil aleatorio
                predictions = clasif.predict(X)                                 # Calcular predicciones de ese clasificador para todas las observaciones
                    
                # mal_clasificadas = Dpesos[Y != predictions]
                # error = sum(mal_clasificadas)                                 # Calcular el error: comparar predicciones con los valores deseados
                                                                                # y acumular los pesos de las observaciones mal clasificadas
                error = np.sum(Dpesos * (predictions != Y))                     #! Esta forma es mas rapida

                if error < best_error:                                          # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                    best_error = error
                    best_clasificador = clasif
                    predicciones_mejor = predictions                            # las predicciones del mejor clasificador débil
                
            #! Performance
            alfa = (0.5 * np.log((1-best_error)/
                                 (best_error+1e-15))) # Calcular el valor de alfa
            best_clasificador.alfa = alfa
            Dpesos = (Dpesos * np.exp(-alfa * Y * predicciones_mejor))      # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            Dpesos = Dpesos/np.sum(Dpesos)                                  # Normalizar a 1 los pesos
            
            self.clasificadores.append(best_clasificador)                   # Guardar el clasificador en la lista de clasificadores de Adaboost
            
            if verbose:
                print(f"Añadido clasificador {t}: {best_clasificador.caracteristica}, "
                        f"{best_clasificador.umbral:.4f}, "
                        f"{'+' if best_clasificador.polaridad == 1 else '-'}1, "
                        f"{best_error:.6f}")

        return self.clasificadores
                     
    #* Método para obtener una predicción con el clasificador FUERTE Adaboost
    def predict(self, X):
        # Inicializar un array de ceros para las predicciones finales
        predicciones_finales = np.zeros(X.shape[0])

        # Sumar las predicciones de cada clasificador débil ponderadas por su alfa
        for clasif in self.clasificadores:
            predicciones_finales += clasif.predict(X) * clasif.alfa

        # Decidir la clase en función del signo de la suma ponderada
        #! predicciones_finales = np.sign(predicciones_finales)

        return predicciones_finales
    
    def predictBin(self, X):
        # Inicializar un array de ceros para las predicciones finales
        predicciones_finales = np.zeros(X.shape[0])

        # Sumar las predicciones de cada clasificador débil ponderadas por su alfa
        for clasif in self.clasificadores:
            predicciones_finales += clasif.predict(X) * clasif.alfa

        # Decidir la clase en función del signo de la suma ponderada
        predicciones_finales = np.sign(predicciones_finales)

        return predicciones_finales

class DecisionStump:
    #* Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.caracteristica = np.random.randint(0, n_features)
        self.umbral = np.random.rand()
        self.polaridad = np.random.choice([1, -1])
        self.alfa = None
    
    #* Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        
        numObservaciones = X.shape[0]
        predicciones = np.ones(numObservaciones)
        columna_X = X[:, self.caracteristica]
        
        if self.polaridad == 1:
            predicciones[ columna_X < self.umbral] = -1      # Si la característica que comprueba este clasificador es menor que el umbral y la polaridad es 1
        
        elif self.polaridad == -1:
            predicciones[ columna_X > self.umbral] = -1      # o si es mayor que el umbral y la polaridad es -1, devolver -1 (no pertenece a la clase)
        
        # Si no, devolver 1 (pertenece a la clase)

        return predicciones

def balancear_clases(X, Y, clase):
    
    positivos = np.where(Y == clase)[0]
    negativos = np.where(Y != clase)[0]

    # Escoger la cantidad de -1s aleatorios = n 1s
    negativos_random = np.random.choice(negativos, size=len(positivos), replace=False)

    # Combinar los índices de las clases 1 y -1 escogidos
    indices_combinados = np.concatenate((positivos, negativos_random))
    np.random.shuffle(indices_combinados)

    X_balanceado = X[indices_combinados]
    Y_balanceado = np.where(Y[indices_combinados] == clase, 1, -1)

    return X_balanceado, Y_balanceado

def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose):
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    clasifFuerte = Adaboost(T,A)
    
    if verbose:
        print(f"Entrenando clasificador Adaboost para el dígito {clase}, T={T}, A={A}")
    
    Y_train_bin = np.where(Y_train == clase, 1, -1)
    Y_test_bin = np.where(Y_test == clase, 1, -1)
    
    start = time.time()
    clasifFuerte.fit(X_train, Y_train_bin, verbose)
    end = time.time()
    tiempo = end - start
    
    Y_train_pred = clasifFuerte.predictBin(X_train)
    Y_test_pred = clasifFuerte.predictBin(X_test)
    
    aciertos_train = np.sum(Y_train_bin == Y_train_pred)
    precision_train = (aciertos_train / len(Y_train_bin)) * 100  # Usa Y_train_bin
    
    aciertos_test = np.sum(Y_test_bin == Y_test_pred)
    precision_test = (aciertos_test / len(Y_test_bin)) * 100  # Usa Y_test_bin
    
    if verbose:
        print(f"Tasas de acierto (train, test) y tiempo: {precision_train:.2f}%, {precision_test:.2f}%, {tiempo:.3f} s")
    return [precision_train, precision_test, tiempo]  

def tarea_1C_graficas_rendimiento():
    rango_T = range(10, 30, 5)
    rango_A = range(100, 500, 50)
    resultados = {t: [] for t in rango_T}
    
    for t in rango_T:
        for a in rango_A:
            precision_train, precision_test, tiempo = tareas_1A_y_1B_adaboost_binario(clase=9, T=t, A=a, verbose=False)
            resultados[t].append((a, precision_train, precision_test, tiempo))
    
    for t in rango_T:
        As = []
        precisiones_train = []
        precisiones_test = []
        tiempos = []

        # Recorrer la lista de resultados y acumular los valores en las listas
        for resultado in resultados[t]:
            As.append(resultado[0])
            precisiones_train.append(resultado[1])
            precisiones_test.append(resultado[2])
            tiempos.append(resultado[3])
        
       
        fig, ax1 = plt.subplots()

        # Tasa de acierto
        color = 'tab:red'
        ax1.set_xlabel('A')
        ax1.set_ylabel('Tasa de Acierto', color=color)
        ax1.plot(As, precisiones_train, label='Train', marker='x', color=color)
        color = 'tab:green'
        ax1.plot(As, precisiones_test, label='Test', marker='o', color=color, linestyle='dashed')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        # Tiempos de entrenamiento
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Tiempo de Entrenamiento (s)', color=color)
        ax2.plot(As, tiempos, label='Tiempo de entrenamiento', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title(f'Tasa de Acierto y Tiempo de Entrenamiento para T={t}')
        fig.tight_layout()
        plt.show()

def tarea_1D_predict_multiclase(X_test, clasificadores):
    predicciones = np.zeros((X_test.shape[0], 10))
    
    for clase, clasif in clasificadores.items():
            predicciones[:, clase] = clasif.predict(X_test)
            
    return np.argmax(predicciones, axis=1)

def tarea_1D_adaboost_multiclase(T, A, verbose = False):
    clases = range(10)
    clasificadores = {}
    
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    t0 = time.time()
    for clase in clases:
        if verbose:
            print(f"Entrenando clasificador Adaboost para el dígito {clase}, T={T}, A={A}")
            
        X_train_balanceado, Y_train_balanceado = balancear_clases(X_train, Y_train, clase)

        adaboost  = Adaboost(T,A)
        adaboost.fit(X_train_balanceado, Y_train_balanceado, False)
        clasificadores[clase] = adaboost
    
    tf = time.time() - t0
    predicciones = tarea_1D_predict_multiclase(X_test, clasificadores)
    correctas = np.sum(predicciones == Y_test)
    total = len(Y_test)
    accuracy = correctas / total
    
    if verbose:
        print(f"La precisión del clasificador multiclase es: {accuracy*100:.2f}%, {tf:.3f} s")
    
    return accuracy, tf

def tarea_2A_AdaBoostClassifier_default(n_estimators, verbose = False):
    
    if verbose:
        print(f"Entrenando AdaBoostClassifier con T = {n_estimators}")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    adaboost_sklearn = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1)
    t0 = time.time()
    adaboost_sklearn.fit(X_train,Y_train)
    tf = time.time() - t0
    
    Y_pred_train = adaboost_sklearn.predict(X_train)
    Y_pred_test = adaboost_sklearn.predict(X_test)
    
    accuracy_test = accuracy_score(Y_test, Y_pred_test)
    accuracy_train = accuracy_score(Y_train, Y_pred_train)
    
    if verbose:
        print(f"Tasas de acierto (train, test) y tiempo: {(accuracy_train * 100):.2f}%, {(accuracy_test * 100):.2f}%, {tf:.3f} s")
    
    return accuracy_test,tf

def tarea_2B_graficas_rendimiento():
    rango_T = [10, 20, 40]

    resultados_MI_Adaboost = {T: [] for T in rango_T}
    resultados_Sklearn = {T: [] for T in rango_T}

    for T in rango_T:
        #Adaboost multiclase
        for A in rango_T:
            precision_adaboost, tiempo_adaboost = tarea_1D_adaboost_multiclase(T, A)
            resultados_MI_Adaboost[T].append((A, precision_adaboost, tiempo_adaboost))

        #scikit-learn
        for A in rango_T:
            precision_sklearn, tiempo_sklearn = tarea_2A_AdaBoostClassifier_default(T)
            resultados_Sklearn[T].append((A, precision_sklearn, tiempo_sklearn))

    plt.figure(figsize=(15, 6))

    for T in rango_T:
        As_1D, precisiones_1D, tiempos_1D = [], [], []
        precisiones_2A, tiempos_2A = [], []
        
        for resultado in resultados_MI_Adaboost[T]:
            As_1D.append(resultado[0])
            precisiones_1D.append(resultado[1])
            tiempos_1D.append(resultado[2])
            
        for resultado in resultados_Sklearn[T]:
            precisiones_2A.append(resultado[1])
            tiempos_2A.append(resultado[2])
        
        fig, ax1 = plt.subplots()

        # Tasa de acierto
        ax1.set_xlabel('A')
        ax1.set_ylabel('Tasa de Acierto', color='red')
        ax1.plot(As_1D, precisiones_1D, label=f'Mi Adaboost T={T}', color='tab:red', marker='o')
        ax1.plot(As_1D, precisiones_2A, label=f'Sklearn T={T}', color='magenta')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        # Tiempo de entrenamiento
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Tiempo de Entrenamiento (s)', color='tab:blue')
        ax2.plot(As_1D, tiempos_1D, label=f'Tiempo Mi Adaboost T={T}', color='forestgreen')
        ax2.plot(As_1D, tiempos_2A, label=f'Tiempo Sklearn T={T}', color='lime')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title(f'Tasa de Acierto y Tiempo de Entrenamiento para T={T}')
        fig.tight_layout()
        plt.show()
  
def tarea_2D_clasificador_MLP_para_MNIST_con_Keras():
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    model = Sequential()
    # Formatear imágenes
    model.add(Flatten(input_shape=(28*28,)))
    model.add(Dense(128, activation='relu')) # capa oculta
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax')) #10 clases del 0 al 9

    # Configurar el modelo
    model.compile(optimizer=Adam(learning_rate=0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    t0 = time.time()
    #* fit(X_train , Y_train, )
    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    tf = time.time() - t0
    print(f"Precision del Test: {test_acc*100:.2f}%, {tf:.2f} segundos")
    
    return test_acc,tf

def tarea_2F_graficas_rendimiento():
    rango_T = [10, 20, 40]

    resultados_MI_Adaboost = {T: [] for T in rango_T}
    resultados_Sklearn = {T: [] for T in rango_T}
    resultados_MLP = {T: [] for T in rango_T}

    # Obtener resultados de Adaboost y Sklearn
    for T in rango_T:
        for A in rango_T:
            precision_adaboost, tiempo_adaboost = tarea_1D_adaboost_multiclase(T, A)
            resultados_MI_Adaboost[T].append((A, precision_adaboost, tiempo_adaboost))

        for A in rango_T:
            precision_sklearn, tiempo_sklearn = tarea_2A_AdaBoostClassifier_default(T)
            resultados_Sklearn[T].append((A, precision_sklearn, tiempo_sklearn))
            
        for A in rango_T:
            precision_MLP, tiempo_MLP = tarea_2D_clasificador_MLP_para_MNIST_con_Keras()
            resultados_MLP[T].append((A,precision_MLP,tiempo_MLP)) 

    # Generar gráficos
    plt.figure(figsize=(20, 6))

    for T in rango_T:
        As_1D, precisiones_1D, tiempos_1D = [], [], []
        precisiones_2A, tiempos_2A = [], []
        precisiones_Keras, tiempos_Keras = [],[]

        for resultado in resultados_MI_Adaboost[T]:
            As_1D.append(resultado[0])
            precisiones_1D.append(resultado[1])
            tiempos_1D.append(resultado[2])

        for resultado in resultados_Sklearn[T]:
            precisiones_2A.append(resultado[1])
            tiempos_2A.append(resultado[2])
            
        for resultado in resultados_MLP[T]:
            precisiones_Keras.append(resultado[1])
            tiempos_Keras.append(resultado[2])

        fig, ax1 = plt.subplots()

        # Tasa de acierto
        ax1.set_xlabel('A')
        ax1.set_ylabel('Tasa de Acierto', color='tab:red')
        ax1.plot(As_1D, precisiones_1D, label=f'Mi Adaboost T={T}', color='tab:red')
        ax1.plot(As_1D, precisiones_2A, label=f'Sklearn T={T}', color='magenta')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Tiempo de entrenamiento
        ax2 = ax1.twinx()
        ax2.set_ylabel('Tiempo de Entrenamiento (s)', color='tab:blue')
        ax2.plot(As_1D, tiempos_1D, label=f'Tiempo Mi Adaboost T={T}', color='forestgreen')
        ax2.plot(As_1D, tiempos_2A, label=f'Tiempo Sklearn T={T}', color='lime')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Añadir resultados MLP
        ax1.plot(As_1D, precisiones_Keras, label=f'MLP Keras', color='orange')
        ax2.plot(As_1D, tiempos_Keras, label='MLP Keras Tiempo', color='yellow')

        leyendas_ax1, etiquetas_ax1 = ax1.get_legend_handles_labels()
        leyendas_ax2, etiquetas_ax2 = ax2.get_legend_handles_labels()
        
        plt.legend(leyendas_ax1 + leyendas_ax2, etiquetas_ax1 + etiquetas_ax2, loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.title(f'Tasa de Acierto y Tiempo de Entrenamiento para T={T} incluyendo MLP')
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":

    rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9, T=25, A=1000, verbose=True)
    
    #! tarea_1C_graficas_rendimiento()

    #!tarea_1D_adaboost_multiclase(T=25, A=300, verbose=True)
    
    #!tarea_2A_AdaBoostClassifier_default(100)
    
    #!tarea_2B_graficas_rendimiento()

    #!tarea_2D_clasificador_MLP_para_MNIST_con_Keras()
    
    #tarea_2F_graficas_rendimiento()