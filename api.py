from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = FastAPI(
    title="Fraud Detection ML Engine",
    description="API de prédiction des transactions frauduleuses en temps réel."
)

try:
    model = joblib.load('models/random_forest_fraud_model.pkl')
except Exception as e:
    model = None
    print(f"Erreur critique : Impossible de charger le modèle. Détails : {e}")

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "Opérationnel", "message": "API de détection de fraude en ligne."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle de prédiction n'est pas disponible.")
    
    try:
        input_data = np.array(transaction.features).reshape(1, -1)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        resultat = "Fraude" if prediction[0] == 1 else "Normal"
        
        return {
            "prediction": resultat,
            "probabilite_fraude": round(float(probability), 4)
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Erreur de format des données : {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {e}")
