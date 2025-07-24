import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical 
import logging

# Mock environment variables to avoid dependency on actual file system
@pytest.fixture(scope="module")
def mock_env_vars():
    os.environ["RAW_DATA_PATH"] = "data/raw/Epileptic Seizure Recognition.csv"
    os.environ["PROCESSED_X_TRAIN_PATH"] = "data/processed/X_train.npy"
    os.environ["PROCESSED_Y_TRAIN_PATH"] = "data/processed/Y_train.npy"
    os.environ["PROCESSED_X_TEST_PATH"] = "data/processed/X_test.npy"
    os.environ["PROCESSED_Y_TEST_PATH"] = "data/processed/Y_test.npy"
    os.environ["PATIENT_DATA_PATH"] = "data/patients/patients_data.csv"
    yield
    # Clean up after test
    del os.environ["RAW_DATA_PATH"]
    del os.environ["PROCESSED_X_TRAIN_PATH"]
    del os.environ["PROCESSED_Y_TRAIN_PATH"]
    del os.environ["PROCESSED_X_TEST_PATH"]
    del os.environ["PROCESSED_Y_TEST_PATH"]
    del os.environ["PATIENT_DATA_PATH"]

# Test to check if data is loaded correctly
def test_load_raw_data(mock_env_vars):
    raw_data_path = os.getenv("RAW_DATA_PATH")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        pytest.fail(f"File not found: {raw_data_path}")
    
    # Check if dataframe is not empty
    assert not df.empty, "Raw data should not be empty"
    assert df.shape[1] > 1, "Data should have multiple columns"


# Test to check if processed data is saved correctly
def test_save_processed_data(mock_env_vars):
    processed_x_train_path = os.getenv("PROCESSED_X_TRAIN_PATH")
    processed_y_train_path = os.getenv("PROCESSED_Y_TRAIN_PATH")
    processed_x_test_path = os.getenv("PROCESSED_X_TEST_PATH")
    processed_y_test_path = os.getenv("PROCESSED_Y_TEST_PATH")
    
    # Mock the data to be saved
    X_train = np.random.rand(100, 178, 1)
    Y_train = np.random.randint(0, 2, (100, 2))
    X_test = np.random.rand(25, 178, 1)
    Y_test = np.random.randint(0, 2, (25, 2))

    # Save the data
    np.save(processed_x_train_path, X_train)
    np.save(processed_y_train_path, Y_train)
    np.save(processed_x_test_path, X_test)
    np.save(processed_y_test_path, Y_test)

    # Check if the files are saved and not empty
    assert os.path.exists(processed_x_train_path), f"{processed_x_train_path} does not exist"
    assert os.path.exists(processed_y_train_path), f"{processed_y_train_path} does not exist"
    assert os.path.exists(processed_x_test_path), f"{processed_x_test_path} does not exist"
    assert os.path.exists(processed_y_test_path), f"{processed_y_test_path} does not exist"
    
    # Load and check if the data is saved correctly
    loaded_X_train = np.load(processed_x_train_path)
    loaded_Y_train = np.load(processed_y_train_path)
    loaded_X_test = np.load(processed_x_test_path)
    loaded_Y_test = np.load(processed_y_test_path)

    assert loaded_X_train.shape == X_train.shape, "Loaded X_train shape mismatch"
    assert loaded_Y_train.shape == Y_train.shape, "Loaded Y_train shape mismatch"
    assert loaded_X_test.shape == X_test.shape, "Loaded X_test shape mismatch"
    assert loaded_Y_test.shape == Y_test.shape, "Loaded Y_test shape mismatch"

# Test to check the patient data saved
def test_patient_data_saved(mock_env_vars):
    patient_data_path = os.getenv("PATIENT_DATA_PATH")
    try:
        df_remaining = pd.read_csv(patient_data_path)
        logging.info(f"Patient data loaded successfully")
    except FileNotFoundError:
        pytest.fail(f"Patient data file not found: {patient_data_path}")
    
    # Check if the patient data file is not empty
    assert not df_remaining.empty, "Patient data should not be empty"
