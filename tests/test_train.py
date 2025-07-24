# test_train.py

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from services.model_training.train import load_data, main
import json
import shutil

# Chemins simulés pour les tests
TEST_DATA_DIR = "./test_data"
TEST_MODEL_PATH = "./test_model/epilepsy_model.keras"
TEST_METRICS_PATH = "./test_metrics/model_metrics.json"

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Crée un environnement de test avec des données simulées."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    # Créer des données simulées pour le test
    X_train = np.random.rand(800, 178, 1)
    Y_train = np.random.randint(2, size=(800, 2))
    X_test = np.random.rand(200, 178, 1)
    Y_test = np.random.randint(2, size=(200, 2))

    # Sauvegarder les données simulées
    np.save(os.path.join(TEST_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(TEST_DATA_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(TEST_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(TEST_DATA_DIR, "Y_test.npy"), Y_test)

    yield

    # Nettoyer après les tests
    for path in [TEST_DATA_DIR, TEST_MODEL_PATH, TEST_METRICS_PATH]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


@patch("services.model_training.train.mlflow")
def test_load_data(mock_mlflow):
    """Teste la fonction `load_data`."""
    X_train, Y_train, X_test, Y_test = load_data(TEST_DATA_DIR)

    assert X_train.shape == (800, 178, 1), "La forme de X_train est incorrecte."
    assert Y_train.shape == (800, 2), "La forme de Y_train est incorrecte."
    assert X_test.shape == (200, 178, 1), "La forme de X_test est incorrecte."
    assert Y_test.shape == (200, 2), "La forme de Y_test est incorrecte."


@patch("services.model_training.train.mlflow")
def test_main_training_process(mock_mlflow):
    """Teste le processus principal d'entraînement."""
    mock_mlflow.start_run.return_value = MagicMock()
    mock_mlflow.log_param = MagicMock()
    mock_mlflow.log_metric = MagicMock()

    # Arguments simulés pour le test
    class MockArgs:
        data_path = TEST_DATA_DIR
        model_path = TEST_MODEL_PATH
        metrics_path = TEST_METRICS_PATH
        epochs = 2
        batch_size = 16

    args = MockArgs()

    # Exécuter la fonction principale
    main(args)

    # Vérifier que le modèle a été sauvegardé
    assert os.path.exists(TEST_MODEL_PATH), "Le fichier du modèle n'a pas été créé."

    # Vérifier que les métriques ont été sauvegardées
    assert os.path.exists(TEST_METRICS_PATH), "Le fichier des métriques n'a pas été créé."

    with open(TEST_METRICS_PATH, "r") as f:
        metrics = json.load(f)
        assert "train_accuracy" in metrics, "Les métriques ne contiennent pas 'train_accuracy'."
        assert "val_accuracy" in metrics, "Les métriques ne contiennent pas 'val_accuracy'."
        assert "train_loss" in metrics, "Les métriques ne contiennent pas 'train_loss'."
        assert "val_loss" in metrics, "Les métriques ne contiennent pas 'val_loss'."

    # Vérifier que MLflow a été appelé correctement
    mock_mlflow.set_experiment.assert_called_once_with("epilepsy_training")
    mock_mlflow.tensorflow.autolog.assert_called_once()
    mock_mlflow.log_param.assert_any_call("architecture", "2xLSTM+Dense")
    mock_mlflow.log_metric.assert_any_call("final_train_accuracy", pytest.approx(0.5, abs=0.2))





