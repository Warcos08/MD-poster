import pickle
import pandas as pd
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC

import warnings

import test

warnings.filterwarnings('ignore')


def trainLightGBM(df):
    # Separo las labels del entrenamiento
    X = df.drop("Chapter", axis=1)
    y = df["Chapter"]

    print(X.head(5))
    print(X.columns)

    # Debido a que el conjunto esta desbalanceado, aplicamos oversampling
    '''dist = {1:725, 2:245, 5:245, 11:573, 12:497, 13:245, 16:245, 18:819, 20:276, 22:245, 23:545}
    oversampler = RandomOverSampler(random_state=42)
    X, y = oversampler.fit_resample(X, y)'''

    lgbm = LGBMClassifier()
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

    dfTest = pd.read_csv("datasets/bow/testBOW.csv")
    test.testSVM(dfTest)

def trainSVM(df):
    # Separo las labels del entrenamiento
    X = df.drop("Chapter", axis=1)
    y = df["Chapter"]

    print(X.head(5))

    # Debido a que el conjunto esta desbalanceado, aplicamos oversampling
    '''dist = {1:725, 2:245, 5:245, 11:573, 12:497, 13:245, 16:245, 18:819, 20:276, 22:245, 23:545}
    oversampler = RandomOverSampler(random_state=42, sampling_strategy=dist)
    X, y = oversampler.fit_resample(X, y)'''

    svc = SVC(C=1, gamma=1, random_state=42)

    # Entreno el modelo
    svc.fit(X, y)

    # Guardo el modelo generado
    file = open("./modelos/SVC.sav", "wb")
    pickle.dump(svc, file)
    file.close()

    dfTest = pd.read_csv("datasets/lda/testLDA.csv")
    test.testSVM(dfTest)