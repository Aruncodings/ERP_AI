import json
import threading
import time
from datetime import datetime
from pathlib import Path

from erp_backend.core.config import (
    BASE_DATASET_PATH,
    FEEDBACK_COUNTER_PATH,
    FEEDBACK_PATH,
    FEEDBACK_RETRAIN_MIN_SAMPLES,
    FEEDBACK_RETRAIN_THRESHOLD,
    LOCAL_FEEDBACK_PATH,
    RETRAIN_DATASET_PATH,
)

_COUNTER_LOCK = threading.Lock()


# ── Feedback I/O ──────────────────────────────────────────────────────────────

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


def add_feedback(
    question,
    answer=None,
    positive=False,
    wrong_collection=None,
    correct_collection=None,
    wrong_fields=None,
    correct_fields=None,
    plan=None,
):
    feedback_data = load_feedback()
    entry = {
        "question": question,
        "best_answer": answer or "",
        "positive": bool(positive),
        "timestamp": datetime.now().isoformat(),
    }
    if wrong_collection:
        entry["wrong_collection"] = str(wrong_collection).strip()
    if correct_collection:
        entry["correct_collection"] = str(correct_collection).strip()
    if wrong_fields:
        entry["wrong_fields"] = list(wrong_fields) if isinstance(wrong_fields, list) else [str(wrong_fields)]
    if correct_fields:
        entry["correct_fields"] = (
            list(correct_fields) if isinstance(correct_fields, list) else [str(correct_fields)]
        )
    if plan:
        entry["plan"] = plan
    feedback_data.append(entry)
    save_feedback(feedback_data)
    _increment_feedback_counter()
    return entry


# ── Feedback Counter (persistent) ─────────────────────────────────────────────

def _load_counter():
    path = Path(FEEDBACK_COUNTER_PATH)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"count": 0, "last_trained_at": None, "last_timestamp": None}


def _save_counter(data):
    path = Path(FEEDBACK_COUNTER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _increment_feedback_counter():
    with _COUNTER_LOCK:
        counter = _load_counter()
        counter["count"] = int(counter.get("count", 0)) + 1
        counter["last_timestamp"] = datetime.now().isoformat()
        _save_counter(counter)
    return counter["count"]


def get_feedback_count():
    counter = _load_counter()
    return int(counter.get("count", 0))


def should_trigger_retrain():
    count = get_feedback_count()
    return count >= max(1, FEEDBACK_RETRAIN_THRESHOLD) and count >= max(1, FEEDBACK_RETRAIN_MIN_SAMPLES)


def mark_trained():
    with _COUNTER_LOCK:
        counter = _load_counter()
        counter["count"] = 0
        counter["last_trained_at"] = datetime.now().isoformat()
        _save_counter(counter)


# ── Feedback-to-training-data conversion ──────────────────────────────────────

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

    result = {"question": question, "best_answer": answer}

    wrong_col = str(row.get("wrong_collection") or "").strip()
    correct_col = str(row.get("correct_collection") or "").strip()
    if wrong_col and correct_col:
        result["collection_correction"] = {"wrong": wrong_col, "correct": correct_col}

    wrong_flds = row.get("wrong_fields") or []
    correct_flds = row.get("correct_fields") or []
    if wrong_flds and correct_flds:
        result["field_corrections"] = list(zip(wrong_flds, correct_flds))

    return result


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


def collection_correction_rows():
    rows = []
    for row in load_feedback():
        wrong = str(row.get("wrong_collection") or "").strip()
        correct = str(row.get("correct_collection") or "").strip()
        question = str(row.get("question") or "").strip()
        if wrong and correct and question:
            rows.append({"question": question, "wrong": wrong, "correct": correct})
    return rows


def field_correction_rows():
    rows = []
    for row in load_feedback():
        question = str(row.get("question") or "").strip()
        wrong_flds = row.get("wrong_fields") or []
        correct_flds = row.get("correct_fields") or []
        if not question or not wrong_flds or not correct_flds:
            continue
        rows.append({
            "question": question,
            "wrong_fields": list(wrong_flds),
            "correct_fields": list(correct_flds),
        })
    return rows
