import os
import argparse
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load env
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
MODEL_NAME = os.getenv("MODEL_ID", "text-bison@001")

# Init
 aiplatform.init(project=PROJECT_ID, location=REGION)
model = aiplatform.Model(model_name=MODEL_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    )
    # Emit parsable log
    print(f"DEPLOY;ENDPOINT={endpoint.resource_name};JOB_ID={args.job_id};RUN_ID={args.run_id}")