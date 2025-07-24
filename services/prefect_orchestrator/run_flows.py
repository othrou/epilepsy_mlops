#!/usr/bin/env python3

from prefect import flow, task, serve
import subprocess
import os

# Your existing tasks (copy them here exactly as they are)
@task
def run_preprocessing():
    result = subprocess.run(
        ["python", "/app/services/preprocessing/preprocessing.py"],
        capture_output=True,
        text=True
    )
    print("📊 Preprocessing STDOUT:\n", result.stdout)
    print("📊 Preprocessing STDERR:\n", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError("❌ Preprocessing script failed!")

@task
def run_training():
    result = subprocess.run(
        [
            "python", "/app/services/model_training/train.py",
            "--data_path", "/app/data/processed",
            "--model_path", "/app/models/epilepsy_model.keras",
            "--metrics_path", "/app/metrics/model_metrics.json",
            "--epochs", "10",
            "--batch_size", "15"
        ],
        capture_output=True,
        text=True
    )
    print("🧠 Training STDOUT:\n", result.stdout)
    print("🧠 Training STDERR:\n", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError("❌ Training script failed!")

@task
def run_evaluate():
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    process = subprocess.Popen(
        ["python", "/app/services/evaluate/evaluate.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": mlflow_uri}
    )
    
    for line in process.stdout:
        print("⚙️ Evaluate STDOUT:", line, end='')
    
    for line in process.stderr:
        print("⚙️ Evaluate STDERR:", line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("❌ Evaluate script failed!")

@task
def dvc_fetch_and_checkout(path):
    fetch_result = subprocess.run(
        ["dvc", "fetch", path],
        capture_output=True,
        text=True
    )
    print(f"⬇️ DVC Fetch '{path}' STDOUT:\n{fetch_result.stdout}")
    print(f"⬇️ DVC Fetch '{path}' STDERR:\n{fetch_result.stderr}")
    if fetch_result.returncode != 0:
        raise RuntimeError(f"❌ DVC fetch failed for {path}")
    
    checkout_result = subprocess.run(
        ["dvc", "checkout", path],
        capture_output=True,
        text=True
    )
    print(f"🔄 DVC Checkout '{path}' STDOUT:\n{checkout_result.stdout}")
    print(f"🔄 DVC Checkout '{path}' STDERR:\n{checkout_result.stderr}")
    if checkout_result.returncode != 0:
        raise RuntimeError(f"❌ DVC checkout failed for {path}")

@task
def dvc_add_and_push(path):
    add_result = subprocess.run(
        ["dvc", "add", path],
        capture_output=True,
        text=True
    )
    print(f"➕ DVC Add '{path}' STDOUT:\n{add_result.stdout}")
    print(f"➕ DVC Add '{path}' STDERR:\n{add_result.stderr}")
    if add_result.returncode != 0:
        raise RuntimeError(f"❌ DVC add failed for {path}")
    
    push_result = subprocess.run(
        ["dvc", "push"],
        capture_output=True,
        text=True
    )
    print(f"⬆️ DVC Push STDOUT:\n{push_result.stdout}")
    print(f"⬆️ DVC Push STDERR:\n{push_result.stderr}")
    if push_result.returncode != 0:
        raise RuntimeError(f"❌ DVC push failed for {path}")

@flow(name="Monthly ML Pipeline")
def monthly_pipeline():
    # Fetch and checkout raw data so you have actual raw files locally
    dvc_fetch_and_checkout("data/raw")
    
    # Run preprocessing on raw data
    run_preprocessing()
    
    # Add & push processed data versioned by DVC
    dvc_add_and_push("data/processed")
    
    # Run training on processed data
    run_training()
    
    # Add & push models and metrics after training
    dvc_add_and_push("models")
    dvc_add_and_push("metrics")
    
    # Run evaluation
    run_evaluate()
    
    # Add & push production artifacts
    dvc_add_and_push("production")

if __name__ == "__main__":
    # Deploy with monthly schedule (1st day of every month at 2 AM)
    monthly_pipeline.serve(
        name="monthly-ml-deployment",
        cron="0 2 1 * *"  # minute hour day month weekday
    )