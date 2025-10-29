from joblib import dump, load
from google.cloud import storage
import os
import time
import glob

from dotenv import load_dotenv
load_dotenv()
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")


def save_model(model, local_registry_path, model_name, bucket_name=None):
    filename = f"{model_name}.pkl"
    local_path = os.path.join(local_registry_path, "models", filename)

    if not bucket_name:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        dump(model, local_path)
        print(f"✅ Model saved locally at {local_path}")

    if bucket_name:
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"models/{filename}")
        blob.upload_from_filename(local_path)
        print(f"✅ Model uploaded to GCS at gs://{bucket_name}/models/{filename}")



def load_model(local_registry_path, model_name=None, bucket_name=None):
    model_dir = os.path.join(local_registry_path, "models")

    if model_name:
        filename = f"{model_name}.pkl"
        local_path = os.path.join(model_dir, filename)

        if os.path.exists(local_path):
            print(f"✅ Loaded local model: {local_path}")
            return load(local_path)
        elif bucket_name:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"models/{filename}")
            if blob.exists():
                os.makedirs(model_dir, exist_ok=True)
                blob.download_to_filename(local_path)
                print(f"✅ Downloaded model from GCS: {blob.name}")
                return load(local_path)
            else:
                print("❌ Model not found in GCS")
                return None
        else:
            print("❌ Model not found locally")
            return None

    else:
        # Fallback: load latest model
        model_paths = sorted(glob.glob(f"{model_dir}/*.pkl"))
        if not model_paths:
            print("❌ No local model found")
            return None
        latest_path = model_paths[-1]
        print(f"✅ Loaded latest local model: {latest_path}")
        return load(latest_path)
