import os
import pytest
from unittest.mock import patch, MagicMock
import mlflow
from mlflow.exceptions import RestException
from services.evaluate.evaluate import clear_folder, register_model, promote_model_to_production

# Test de la fonction clear_folder
def test_clear_folder_create_new_directory():
    test_path = "test_folder"
    
    # Supprimer le répertoire s'il existe déjà
    if os.path.exists(test_path):
        os.rmdir(test_path)
    
    clear_folder(test_path)
    
    # Vérifier que le répertoire a bien été créé
    assert os.path.exists(test_path), f"Folder {test_path} should be created"
    os.rmdir(test_path)  # Nettoyage après test

def test_clear_folder_remove_files():
    test_path = "test_folder_to_remove"
    
    # Créer un fichier pour tester la suppression
    os.makedirs(test_path, exist_ok=True)
    with open(os.path.join(test_path, "test_file.txt"), "w") as f:
        f.write("Test content")
    
    # Tester la suppression
    clear_folder(test_path)
    
    # Vérifier que le répertoire est vide après suppression
    assert not os.listdir(test_path), f"Folder {test_path} should be empty"
    os.rmdir(test_path)  # Nettoyage après test


# Test de la fonction register_model
@patch("mlflow.register_model")
def test_register_model(mock_register_model):
    mock_model_version = MagicMock()
    mock_model_version.version = "2"
    mock_register_model.return_value = mock_model_version
    
    run_id = "test_run_id"
    model_version = register_model(run_id)
    
    assert model_version.version == "2"
    mock_register_model.assert_called_once()


def test_promote_model_to_production():
    """Test that model promotion to production works correctly"""
    with patch('mlflow.tracking.MlflowClient') as mock_client:
        # Setup
        model_version = "1"
        MODEL_NAME = "test_model"
        
        # Create mock client and method
        mock_instance = mock_client.return_value
        mock_instance.transition_model_version_stage.return_value = MagicMock()
        
        # Call the function under test
        promote_model_to_production(mock_instance, MODEL_NAME, model_version)
        
        # Verify
        mock_instance.transition_model_version_stage.assert_called_once_with(
            name=MODEL_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )





