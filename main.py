## Importando librerias
import optuna
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pickle
def separate_target(df,target="Target"):
    y = df.loc[:,target]
    X = df.loc[:,df.columns != target]
    return y , X
#Crearemos una variables
def FE(df):
    df['Z'] = (df["SD"]*df["SD"] - df["Mean_Integrated"])/df["SD"]
    df['imp'] = (df["EK"]*10)*(df["EK"])+ df['SD_DMSNR_Curve']
    return df
## Cargando base de datos (1)
df = pd.read_csv(r"Dataset/Pulsar.csv")
y , X= separate_target(df,target="Class")

selected_var = ['EK', 'Mean_Integrated', 'Skewness', 'imp', 'SD_DMSNR_Curve', 'Mean_DMSNR_Curve']
file_name = "model/xgb_reg.pkl"
xgb_model_loaded = pickle.load(open(file_name, "rb"))
preprocessor_loaded = pickle.load(open('model/PII_model.pickle', "rb"))

X = preprocessor_loaded.transform(FE(X).loc[:,selected_var])
y_pred = xgb_model_loaded.predict(X)