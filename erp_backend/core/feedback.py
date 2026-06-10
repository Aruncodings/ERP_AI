import json
from datetime import datetime
from pathlib import Path

from erp_backend.core.config import BASE_DATASET_PATH, FEEDBACK_PATH, LOCAL_FEEDBACK_PATH, RETRAIN_DATASET_PATH


def load_feedback():
    rows = []
    for path in (FEEDBACK_PATH, LOCAL_FEEDBACK_PATH):
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
    return rows


def save_feedback(data):
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_feedback(question, answer, positive=False):
    feedback_data = load_feedback()
    feedback_data.append(
        {
            "question": question,
            "best_answer": answer,
            "positive": positive,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_feedback(feedback_data)


def feedback_to_training_row(row):
    question = (row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
    answer = (
        row.get("best_answer")
        or row.get("answer")
        or row.get("response")
        or row.get("output")
        or ""
    ).strip()
    if not question or not answer:
        return None
    return {"question": question, "best_answer": answer}


def build_retrain_dataset():
    feedback_rows = []
    for row in load_feedback():
        training_row = feedback_to_training_row(row)
        if training_row is not None:
            feedback_rows.append(training_row)

    if feedback_rows:
        retrain_rows = feedback_rows
    elif BASE_DATASET_PATH.exists():
        with open(BASE_DATASET_PATH, "r", encoding="utf-8") as f:
            base_rows = json.load(f)
        retrain_rows = base_rows if isinstance(base_rows, list) else []
    else:
        retrain_rows = []

    with open(RETRAIN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(retrain_rows, f, indent=2, ensure_ascii=False)
    return RETRAIN_DATASET_PATH, len(retrain_rows)
