import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def limpiar_texto(texto):
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto


def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def lematizar(tokens):
    return [wnl.lemmatize(token) for token in tokens]


def eliminar_palabras_concretas(tokens):
    palabras_concretas = {"hospit", "die", "death", "doctor", "deceas", "person", "servic", "nurs", "client", "peopl", "patient",                   #ELEMENTOS DE HOSPITAL QUE NO APORTAN INFO SOBRE ENFERMEDAD
                          "brother", "father","respondetn","uncl","famili","member","husband","son", "daughter","marriag",
                          "day", "year", "month", "april", "date", "feb", "jan", "time", "place","later","hour",                                    #FECHAS QUE NO APORTAN INFO SOBRE ENFERMEDD
                          "interview", "opinion", "thousand", "particip", "admit", "document", "inform", "explain", "said", "respond","interviewe",                                                                                                #PALABRAS QUE TIENEN QUE VER CON LA ENTREVISTA
                          "write", "commend", "done", "told", "came", "done", "think", "went", "took", "got",                                       #OTROS VERBOS
                          "brought","becam","start",
                          "even", "also", "sudden", "would", "us", "thank","alreadi","rather","p","none","b",                                       #PALABRAS QUE NO APORTAN SIGNIFICADO
                          "caus", "due", "suffer", "felt", "consequ"}                                                                               #PALABRAS SEGUIDAS POR SINTOMAS


    return [token for token in tokens if token not in palabras_concretas]


def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]


def limpieza_comun(df):
    df = df[["open_response"]]

    # 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
    df["Tokens"] = df.open_response.apply(limpiar_texto)

    # 2.- Tokenizamos
    tokenizer = ToktokTokenizer()
    df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

    # 3.- Eliminar stopwords y digitos
    df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

    # 4.- ESTEMIZAR / LEMATIZAR ???
    df["Tokens"] = df.Tokens.apply(estemizar)

    # 5.- ELIMINAMOS PALABRAS CONCRETAS QUE APARECEN MUCHO PERO NO APORTAN SIGNIFICADO
    df["Tokens"] = df.Tokens.apply(eliminar_palabras_concretas)

    return df

