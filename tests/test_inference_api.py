import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import mlflow.pyfunc
from unittest.mock import patch, MagicMock
from mlflow.tracking import MlflowClient


# Import the app from your main module
from services.inference.inference_api import app

# Create a test client for your FastAPI application
client = TestClient(app)

# --- Fixtures and Mocks ---

@pytest.fixture
def mock_model():
    """Mock to simulate an ML model."""
    mock = MagicMock()
    # Mock predictions to match your actual model output (2 classes)
    mock.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])  # 2 predictions
    return mock

@pytest.fixture
def mock_mlflow(mock_model, monkeypatch):
    """Mock for MLflow."""
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr("services.inference.inference_api.MlflowClient", MagicMock())

@pytest.fixture
def mock_data(monkeypatch):
    """Mock for patient data."""
    # Create mock data that matches your actual data structure
    data = {'patient_id': [15, 15]}
    # Add 178 features (col_0 to col_177) as in your actual data
    for i in range(178):
        data[f'col_{i}'] = [0.5, 0.5]  # Constant values for reproducibility
    
    mock_df = pd.DataFrame(data)
    monkeypatch.setattr("services.inference.inference_api.pd.read_csv", lambda *args, **kwargs: mock_df)

@pytest.fixture
def mock_auth(monkeypatch):
    """Mock for authentication."""
    monkeypatch.setattr(
        "services.inference.inference_api.verify_token", 
        lambda *args, **kwargs: "testuser"
    )

# --- Endpoint Tests ---

def test_health_check():
    """Teste l'endpoint /health."""

    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

### continue the tests