import os
import csv
import argparse
import psycopg
from dotenv import load_dotenv
from google.cloud import aiplatform

# Load config
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
CSV_PATH = os.getenv("CSV_PATH", "./articles.csv")

# Init Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION)
client = aiplatform.gapic.PredictionServiceClient()

# DB connection factory
def get_db_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"))

# Run-state helpers
def get_status(cur, run_id):
    cur.execute("SELECT get_run_status(%s)", (run_id,))
    row = cur.fetchone()
    return row[0] if row else None

def set_result(cur, run_id, result, message=""):
    cur.execute("SELECT set_run_result(%s, %s, %s)", (run_id, result, message))
    cur.connection.commit()

# Summarization call
def summarize_text(text: str) -> str:
    instance = {"content": text}
    request = {"endpoint": ENDPOINT_ID, "instances": [instance]}
    response = client.predict(request=request)
    pred = response.predictions[0]
    return pred.get("summary", pred.get("content", str(pred)))

# CSV mode
def cmd_csv(args):
    # connect metadata DB for run state
    meta_conn = get_db_conn()
    meta_cur = meta_conn.cursor()
    # mark run as Started if new
    if not get_status(meta_cur, args.run_id):
        set_result(meta_cur, args.run_id, 'Started')

    out_file = args.output or "summaries.csv"
    with open(CSV_PATH, newline="", encoding="utf-8") as rf, \
         open(out_file, "w", newline="", encoding="utf-8") as wf:
        reader = csv.DictReader(rf)
        writer = csv.DictWriter(wf, fieldnames=reader.fieldnames + ["summary"])
        writer.writeheader()
        for row in reader:
            status = get_status(meta_cur, args.run_id)
            if status == 'Stopped':
                print(f"STOPPED;JOB_ID={args.job_id};RUN_ID={args.run_id}")
                return
            summary = summarize_text(row.get("content", ""))
            row["summary"] = summary
            writer.writerow(row)
            print(f"CSV_SUMMARY;JOB_ID={args.job_id};RUN_ID={args.run_id};ARTICLE_ID={row.get('article_id')}")
    set_result(meta_cur, args.run_id, 'Success', 'CSV mode complete')
    meta_cur.close(); meta_conn.close()

# DB mode
def cmd_db(args):
    conn = get_db_conn()
    cur = conn.cursor()
    # mark run started
    if not get_status(cur, args.run_id):
        set_result(cur, args.run_id, 'Started')

    cur.execute("SELECT article_id, content FROM articles;")
    for article_id, content in cur.fetchall():
        status = get_status(cur, args.run_id)
        if status == 'Stopped':
            print(f"STOPPED;JOB_ID={args.job_id};RUN_ID={args.run_id}")
            return
        summary = summarize_text(content)
        cur.execute("UPDATE articles SET summary=%s WHERE article_id=%s;", (summary, article_id))
        conn.commit()
        print(f"DB_SUMMARY;JOB_ID={args.job_id};RUN_ID={args.run_id};ARTICLE_ID={article_id}")
    set_result(cur, args.run_id, 'Success', 'DB mode complete')
    cur.close(); conn.close()

# Deploy subcommand to reuse deployment logic
from deployment import model, aiplatform as _a, PROJECT_ID as _P, REGION as _R

def cmd_deploy(args):
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    )
    # mark run complete
    conn = get_db_conn()
    cur = conn.cursor()
    set_result(cur, args.run_id, 'Success', f"Deployed {endpoint.resource_name}")
    cur.close(); conn.close()
    print(f"DEPLOY;ENDPOINT={endpoint.resource_name};JOB_ID={args.job_id};RUN_ID={args.run_id}")

# Main CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--run_id", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_csv = sub.add_parser("csv")
    p_csv.add_argument("-o", "--output")

    sub.add_parser("db")
    sub.add_parser("deploy")

    args = parser.parse_args()
    if args.command == "csv":
        cmd_csv(args)
    elif args.command == "db":
        cmd_db(args)
    elif args.command == "deploy":
        cmd_deploy(args)