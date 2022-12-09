import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
import scikitplot.metrics as skplt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def testLightGBM(df):
    # Cargo el modelo entrenado
    file = open("modelos/lightGBM.sav", "rb")
    lgbm = pickle.load(file)
    file.close()

    # Separo las instancias de su label
    X_test = df.drop("Chapter", axis=1)
    y_test = df["Chapter"]

    # Realizo el test
    y_pred = lgbm.predict(X_test)

    # Comparo las predicciones con la clase real
    print("La accuracy es:", accuracy_score(y_test, y_pred))
    print("La precision es:", precision_score(y_test, y_pred, average='weighted'))
    print("El f1 score es:", f1_score(y_test, y_pred, average='weighted'))

    error = {"aciertos": 0, "errores": 0}
    for y, label in zip(list(y_pred), y_test):
        if y - label == 0:
            error["aciertos"] = error["aciertos"] + 1
        else:
            error["errores"] = error["errores"] + 1
    print(error)
    errorTotal = error["errores"] / (error["errores"] + error["aciertos"])
    print("El error es de: " + str(errorTotal))

    skplt.plot_confusion_matrix(y_test, y_pred)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig('imagenes/matrizLightGBM.png')
    plt.show()


def testLDA(df):
    print("lol")