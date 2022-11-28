import pickle

def train(dfTrain):




    # Guardo el modelo generado
    file = open("./modelos/dbscan.sav", "wb")
    pickle.dump(dbscan, file)
    file.close()