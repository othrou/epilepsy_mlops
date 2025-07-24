import os
import pandas as pd

# Set paths
BASE_DATA_PATH = os.getenv("DATA_PATH", "/app/data")
PATIENT_DATA_PATH = os.getenv("PATIENT_DATA_PATH", "/app/data/patients/patients_data.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/app/data/patients_inference/patients_data_updated.csv")

print(f" Reading patient data from: {PATIENT_DATA_PATH}")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load patient data
df_patients = pd.read_csv(PATIENT_DATA_PATH)

# Drop the label column (assumed to be the last one)
df_patients = df_patients.iloc[:, :-1]

# Assign patient_id: every 8 rows = 1 patient
df_patients = df_patients.reset_index(drop=True)
df_patients["patient_id"] = df_patients.index // 8

# Reorder columns to have patient_id first
cols = ["patient_id"] + [col for col in df_patients.columns if col != "patient_id"]
df_patients = df_patients[cols]

# Save updated file
df_patients.to_csv(OUTPUT_PATH, index=False)

print(f" Patient data prepared and saved to: {OUTPUT_PATH}")
print(f" Total rows: {len(df_patients)}")
print(f" Number of patients: {df_patients['patient_id'].nunique()}")