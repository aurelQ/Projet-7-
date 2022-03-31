# Import Needed Libraries
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
import sklearn.metrics
import pickle
 
# FastAPI libray
from fastapi import FastAPI

# Initiate app instance
app = FastAPI(title='Placement Analytics', version='0.24.2',
              description='Lightgbm model is used for prediction')

# Récupération du jeu de données Test
X_test=pd.read_csv('./model/X_test_ech.csv')

# Lecture du fichier pickle
with open('./model/score_objects2.pkl', 'rb') as handle:
    clf_xgb_w, explainer_xgb = pickle.load(handle)  

# This struture will be used for Json validation.
class Data(BaseModel):
    test_id: int


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():

    return {"Information': 'L'API est déployée"}

# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict")
def predict(data: Data):

    # Création d'un dictionnaire
    data_dict = data.dict() 
    var=data_dict['test_id']
    X_test_df=X_test[X_test["Unnamed: 0"]==var]  
    X_test_df=X_test_df.to_dict('series')
    del X_test_df['Unnamed: 0']
    X_test_df=pd.DataFrame.from_dict([X_test_df])
    print("X_test_df :", X_test_df )

    # Créer la prediction et le score
    seuil=0.2
    prediction_label = clf_xgb_w.predict_proba(X_test_df)[0,1]
    print("prediction_label", type(prediction_label)) #    prediction = clf.predict_proba(data_df_3)
    if prediction_label > seuil:
        prediction = 1
    else:
        prediction=0

    print("prediction :", prediction )
    score = float(prediction_label)
    print("score :", score )

    #Création du dictionnaire de données de sortie 
    result_dict={"prediction":prediction, "score":score}

    # Retourne le dictionnaire
    return result_dict

if __name__ == '__main__':
    uvicorn.run("main_test:app", host="0.0.0.0", port=8000, reload=True)

    