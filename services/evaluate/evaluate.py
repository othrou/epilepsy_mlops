import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

EXPERIMENT_NAME = "epilepsy_training"
MODEL_NAME = "epilepsy_model"
METRIC_NAME = "val_accuracy"
PRODUCTION_DIR = "production"

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_best_run(experiment_id, client):
    runs = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )
    return runs[0] if runs else None

def get_latest_registered_model(client):
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Production", "Staging"])
        if versions:
            return sorted(versions, key=lambda v: int(v.version))[-1]
        return None
    except RestException:
        return None

def get_metric_from_run(client, run_id, metric_name):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_name, None)

def register_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"✅ Registered new model version: {result.version}")
    return result

def promote_model_to_production(client, model_name, version):
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True  # Auto-archive previous ones
        )
        print(f"🚀 Model version {version} promoted to 'Production'")
    except Exception as e:
        print(f"❌ Failed to promote model to Production: {e}")

def save_model_artifact(version_obj, client, dest_path=PRODUCTION_DIR):
    clear_folder(dest_path)
    print(f"⬇ Downloading model version {version_obj.version} to '{dest_path}'")
    client.download_artifacts(run_id=version_obj.run_id, path="models", dst_path=dest_path)
    print(f"📁 Model saved to '{dest_path}'")

if __name__ == "__main__":
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"📡 Using MLflow Tracking URI: {mlflow_tracking_uri}")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
        exit(1)

    best_run = get_best_run(experiment.experiment_id, client)
    if not best_run:
        print("❌ No successful runs found.")
        exit(1)

    best_val_accuracy = best_run.data.metrics.get(METRIC_NAME)
    print(f"🔍 Best run: {best_run.info.run_id} with {METRIC_NAME}: {best_val_accuracy:.4f}")

    current_registered_model = get_latest_registered_model(client)

    if current_registered_model is None:
        print("📦 No registered model found. Registering best run.")
        new_version = register_model(best_run.info.run_id)
        promote_model_to_production(client, MODEL_NAME, new_version.version)
        save_model_artifact(new_version, client)

    else:
        current_val_accuracy = get_metric_from_run(client, current_registered_model.run_id, METRIC_NAME)
        print(f"📌 Current registered model version: {current_registered_model.version} with {METRIC_NAME}: {current_val_accuracy:.4f}")

        if best_val_accuracy > current_val_accuracy:
            print("✅ New model is better! Registering and promoting as new version.")
            new_version = register_model(best_run.info.run_id)
            promote_model_to_production(client, MODEL_NAME, new_version.version)
            save_model_artifact(new_version, client)
        else:
            print("👍 Registered model is still the best. No update made.")
            save_model_artifact(current_registered_model, client)
