import json
import os
from datetime import datetime

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "rag_eval_runs.jsonl",  # file in project root
)

def log_rag_example(question: str, answer: str, contexts: list[str]):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
