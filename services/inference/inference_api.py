import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from mlflow.tracking import MlflowClient
from fastapi.responses import Response


##################### Prometheus monitoring imports #######################

from dotenv import load_dotenv
from prometheus_client import Counter, Histogram


from fastapi import Request
from prometheus_client import generate_latest
from starlette_exporter import PrometheusMiddleware, handle_metrics

app = FastAPI()

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

# Load environment variables
load_dotenv()

# Prometheus Metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
PATIENTS_WITH_EPILEPSY = Counter('patients_with_epilepsy_total', 'Total patients predicted with epilepsy')
PATIENTS_WITHOUT_EPILEPSY = Counter('patients_without_epilepsy_total', 'Total patients predicted without epilepsy')
EPILEPTIC_RECORDINGS = Counter('epileptic_recordings_total', 'Total epileptic recordings predicted')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Latency of predictions')
MODEL_RELOAD_COUNT = Counter('model_reloads_total', 'Total model reloads')
REQUEST_COUNT_INF = Counter('inference_http_requests_total', 'Total HTTP Requests for inference', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY_INF = Histogram('inference_http_request_duration_seconds', 'HTTP Request Latency for inference', ['method', 'endpoint'])

#########################################################################


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#app = FastAPI()

bearer_scheme = HTTPBearer()

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

def verify_token(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = creds.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise JWTError("No username in token")
        return username
    except JWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

BASE_DATA_PATH = os.getenv("DATA_PATH", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./production/model/data/model.keras")
PATIENT_DATA_PATH = os.path.join(BASE_DATA_PATH, "patients_inference/patients_data_updated.csv")

df = None
model = None

# --- Charger le meilleur modèle depuis MLflow ---

def load_best_model_from_mlflow():
    """Charge le dernier modèle disponible depuis le registre MLflow"""
    global model
    try:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        model_uri = f"models:/{os.getenv('MLFLOW_MODEL_NAME', 'epilepsy_model')}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Chargé le modèle depuis MLflow avec succès.")
    except Exception as e:
        logger.error(f"❌ Échec du chargement du modèle depuis MLflow: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    global df, model
    try:
        if not os.path.exists(PATIENT_DATA_PATH):
            logger.error(f"Patient data file not found: {PATIENT_DATA_PATH}")
            raise FileNotFoundError(f"Patient data file not found: {PATIENT_DATA_PATH}")
        
        df = pd.read_csv(PATIENT_DATA_PATH)
        logger.info(f"Loaded patient data with shape: {df.shape}")
        
        # Chargez le modèle depuis MLflow au démarrage
        load_best_model_from_mlflow()
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise e

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "data_shape": df.shape if df is not None else None
    }

@app.post("/reload-model", summary="Recharge le meilleur modèle depuis MLflow")
def reload_model(username: str = Depends(verify_token)):
    """Recharge manuellement le modèle depuis MLflow"""
    logger.info(f"Model reload requested by: {username}")
    load_best_model_from_mlflow()
    if model is None:
        raise HTTPException(status_code=500, detail="Échec du rechargement du modèle.")
    return {"message": "Modèle rechargé avec succès depuis MLflow."}

@app.get("/predict/{patient_id}")
def predict(patient_id: int, username: str = Depends(verify_token)):
    global df, model
    
    if df is None or model is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded")
    
    df_patient = df[df["patient_id"] == patient_id]
    if df_patient.empty:
        raise HTTPException(status_code=404, detail=f"No data found for patient {patient_id}")
    
    try:
        with PREDICTION_LATENCY.time():

            X = df_patient.drop(columns=["patient_id"]).values
            logger.info(f"Input shape before reshape: {X.shape}")
            
            X = X.reshape(X.shape[0], X.shape[1], 1)
            logger.info(f"Input shape after reshape: {X.shape}")
            
            # Make predictions
            preds = model.predict(X)
            pred_classes = np.argmax(preds, axis=1)

             # Log the increment operation
            logger.info(f"Incrementing prediction count by {len(pred_classes)}")
            PREDICTION_COUNT.inc()         
            
        
        # Get epileptic recording indices (1-based)
        epileptic_idxs = [i+1 for i, c in enumerate(pred_classes) if c == 1]
        
        message = (
            f" Patient {patient_id} predicted epileptic recordings at rows: {epileptic_idxs}"
            if epileptic_idxs else
            f" Patient {patient_id} predicted as non-epileptic in all recordings."
        )

        num_epileptic = len(epileptic_idxs)
            
            # Track epilepsy-specific metrics
        if num_epileptic > 0:
                PATIENTS_WITH_EPILEPSY.inc()
                EPILEPTIC_RECORDINGS.inc(num_epileptic)
        else:
                PATIENTS_WITHOUT_EPILEPSY.inc()
        
        logger.info(f"Prediction completed for patient {patient_id} by user {username}")
        
        return {
            "patient_id": patient_id,
            "total_recordings": len(pred_classes),
            "epileptic_recordings": epileptic_idxs,
            "predictions": pred_classes.tolist(),
            "confidence_scores": preds.max(axis=1).tolist(),
            "message": message,
            "processed_by": username
        }
    except Exception as e:
        logger.error(f"Prediction failed for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Add a prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    # Retournez la réponse Prometheus avec le bon Content-Type
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
