import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

import preproceso, train

def entrenamiento():
    dfTrain = pd.read_csv("datasets/data.csv")

    eleccion = int(input('''Elija el preproceso deseado para la prueba:
                            (1) BOW
                            (2) Topic Modeling
                            (3) Salir
                     '''))

    # Aplico el preproceso al train
    if eleccion == 1:
        print("Ha elegido BOW")
        preproceso.bowTrain(dfTrain)

    elif eleccion == 2:
        print("Ha elegido Topic Modeling")
        num_topics = int(input("Introduzca el numero de topicos deseado"))
        preproceso.topicosTrain(dfTrain, num_topics)

    elif eleccion == 3:
        print("Saliendo . . .")
        return

    else:
        print("Seleccion incorrecta")
        entrenamiento()

    # Entreno el modelo del algoritmo elegido
    train.train(dfTrain)


def testeo():
    # La parte de test cambia segun el preproceso empleado para el train
    dfDev = pd.read_csv("datasets/dev.csv")
    dicc = pickle.load("modelos/dicc.sav")
    dfDev = preproceso.topicosTest(dfDev, dicc)     # Si quiero hacer un tf-idf tendre que hacer una seleccion

    location = ""   # path donde se encuentre el modelo entrenado
    model = pickle.load(location)


def main():
    print('''BIENVENIDO AL AGRUPADOR DE DOCUMENTOS MÉDICOS
    
        Pulse el número según lo que que desee ejecutar:
            (1) Entrenar el modelo 
            (2) Testear un modelo previamente entrenado
            (3) Salir

        Por Marcos Merino\n''')

    eleccion = int(input())

    if eleccion == 1:
        print("Ha elegido entrenar un modelo")
        entrenamiento()
        main()

    elif eleccion == 2:
        print("Ha elegido testear un modelo ")
        testeo()
        main()

    elif eleccion == 3:
        print("SALIENDO...")
        return

    else:
        print("Seleccion incorrecta\n\n")
        main()

if __name__ == '__main__':
    # main()

    # Ejecutar una vez para hacer el split
    df = pd.read_csv("datasets/data.csv")
    dfTrain, dfDev = train_test_split(df, test_size=0.2, random_state=42)
    dfTrain.to_csv("datasets/data.csv")
    dfDev.to_csv("datasets/dev.csv")


