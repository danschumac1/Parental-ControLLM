"""
Send a Slack ping using an Incoming Webhook.

Requires:
  - ./resources/.env with SLACK_WEBHOOK_URL set

How to run:
  python ./src/_ping_slack.py \
    --message "hello there!"
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import argparse

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset cleaning."""
    parser = argparse.ArgumentParser(description="Convert .ts time-series dataset to clean NumPy format.")
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="What message do you want to send yourself?",
    )
    return parser.parse_args()

# Load ./resources/.env explicitly
def send_slack_ping(message: str):
    load_dotenv("./resources/.env")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL:
        raise RuntimeError("SLACK_WEBHOOK_URL not found in environment")


    payload = {"text": message}

    r = requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Slack webhook failed: {r.status_code} {r.text}")

if __name__ == "__main__":
    args = parse_args()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_slack_ping(args.message)