import pickle
import pandas as pd
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


def trainLightGBM(df):
    # Separo las labels del entrenamiento
    X = df.drop("Chapter", axis=1)
    y = df["Chapter"]

    print(X.head(5))

    lgbm = LGBMClassifier(random_state=42)
    # Definir mas hiperparámetros (
    # n_estimators
    # learning_rate
    # num_leaves
    # max_depth
    # min_split_gain
    # boosting type)

    # Entreno el modelo
    lgbm.fit(X, y)

    # Guardo el modelo generado
    file = open("./modelos/lightGBM.sav", "wb")
    pickle.dump(lgbm, file)
    file.close()

def trainLDA():
    # El modelo LDA debería estar ya creado por el preproceso realizado
    print("lol")
