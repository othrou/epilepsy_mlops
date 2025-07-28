import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


import os
from pathlib import Path

""""
python based execution

# Chemin relatif au script
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# Debugging output
print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", str(DATA_DIR / "raw/Epileptic Seizure Recognition.csv"))
PROCESSED_X_TRAIN_PATH = os.getenv("PROCESSED_X_TRAIN_PATH", str(DATA_DIR / "processed/X_train.npy"))
PROCESSED_Y_TRAIN_PATH = os.getenv("PROCESSED_Y_TRAIN_PATH", str(DATA_DIR / "processed/Y_train.npy"))
PROCESSED_X_TEST_PATH = os.getenv("PROCESSED_X_TEST_PATH", str(DATA_DIR / "processed/X_test.npy"))
PROCESSED_Y_TEST_PATH = os.getenv("PROCESSED_Y_TEST_PATH", str(DATA_DIR / "processed/Y_test.npy"))
PATIENT_DATA_PATH = os.getenv("PATIENT_DATA_PATH", str(DATA_DIR / "patients/patients_data.csv")) """


CONTAINER_DATA_BASE_PATH = "/app/data"

RAW_DATA_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "raw", "Epileptic Seizure Recognition.csv")
PROCESSED_X_TRAIN_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "processed", "X_train.npy")
PROCESSED_Y_TRAIN_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "processed", "Y_train.npy")
PROCESSED_X_TEST_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "processed", "X_test.npy")
PROCESSED_Y_TEST_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "processed", "Y_test.npy")
PATIENT_DATA_PATH = os.path.join(CONTAINER_DATA_BASE_PATH, "patients", "patients_data.csv")


print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")

"""
# Debugging output
print(f"PROCESSED_X_TRAIN_PATH: {PROCESSED_X_TRAIN_PATH}")
print(f"PROCESSED_Y_TRAIN_PATH: {PROCESSED_Y_TRAIN_PATH}")
print(f"PROCESSED_X_TEST_PATH: {PROCESSED_X_TEST_PATH}")
print(f"PROCESSED_Y_TEST_PATH: {PROCESSED_Y_TEST_PATH}")
print(f"PATIENT_DATA_PATH: {PATIENT_DATA_PATH}") """

print(f"📂 Reading raw data from: {RAW_DATA_PATH}")

try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Data loaded successfully from {RAW_DATA_PATH}")
except FileNotFoundError:
    print(f"File not found at {RAW_DATA_PATH}")


# Read the raw dataset
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f" Data loaded with shape: {df.shape}")
except Exception as e:
    print(f" Error loading data: {str(e)}")
    raise

df = df.iloc[:, 1:]

df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 1 else 0)

df_label_1 = df[df.iloc[:, -1] == 1]  
df_label_0 = df[df.iloc[:, -1] == 0]  

print(f"📊 Class distribution: Epilepsy: {len(df_label_1)}, Non-epilepsy: {len(df_label_0)}")

df_label_1_sample = df_label_1.sample(n=400, random_state=42)
df_label_0_sample = df_label_0.sample(n=400, random_state=42)

df_train = pd.concat([df_label_1_sample, df_label_0_sample])

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_full = df_train.iloc[:, :-1].values  
y_train_full = df_train.iloc[:, -1].values   

Y_train_full = to_categorical(y_train_full, num_classes=2)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_train_full, Y_train_full, test_size=0.20, random_state=42
)

X_train = X_train.reshape(-1, 178, 1)
X_test = X_test.reshape(-1, 178, 1)

print(f"Saving processed data to {os.path.dirname(PROCESSED_X_TRAIN_PATH)}")
np.save(PROCESSED_X_TRAIN_PATH, X_train)
np.save(PROCESSED_Y_TRAIN_PATH, Y_train)
np.save(PROCESSED_X_TEST_PATH, X_test)
np.save(PROCESSED_Y_TEST_PATH, Y_test)

df_remaining = df.drop(df_train.index)
df_remaining.to_csv(PATIENT_DATA_PATH, index=False)

# Print the shapes for verification
print(f" Preprocessing completed!")
print(f" X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f" X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
print(f" Patient dataset saved at {PATIENT_DATA_PATH}")
print(f"Class distribution in training data: {pd.Series(y_train_full).value_counts()}")
