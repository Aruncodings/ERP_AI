import asyncio
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from erp_backend.core.config import ADAPTER_DIR, CONTINUOUS_TRAINING_ENABLED, RETRAIN_DATASET_PATH
from erp_backend.core.feedback import (
    build_retrain_dataset,
    collection_correction_rows,
    field_correction_rows,
    get_feedback_count,
    load_feedback,
    mark_trained,
)
from erp_backend.core.metrics import observe_error

logger = logging.getLogger(__name__)

_TRAINER_LOCK = asyncio.Lock()
_TRAINING_IN_PROGRESS = False


def get_training_info():
    feedback_count = get_feedback_count()
    all_feedback = load_feedback()
    col_corrections = len(collection_correction_rows())
    fld_corrections = len(field_correction_rows())
    retrain_dataset_path = Path(RETRAIN_DATASET_PATH)
    retrain_rows = 0
    if retrain_dataset_path.exists():
        try:
            with open(retrain_dataset_path, "r", encoding="utf-8") as f:
                retrain_rows = len(json.load(f))
        except Exception:
            pass
    return {
        "feedback_count": feedback_count,
        "total_feedback_records": len(all_feedback),
        "collection_corrections": col_corrections,
        "field_corrections": fld_corrections,
        "retrain_dataset_size": retrain_rows,
        "training_in_progress": _TRAINING_IN_PROGRESS,
    }


def _build_collection_selection_dataset(corrections, output_path):
    rows = []
    for item in corrections:
        question = str(item.get("question") or "").strip()
        wrong = str(item.get("wrong") or "").strip()
        correct = str(item.get("correct") or "").strip()
        if not question or not correct:
            continue
        rows.append({
            "instruction": "Identify the correct ERP table for this query.",
            "input": f"Query: {question}",
            "output": f"The correct table is `{correct}`.",
            "metadata": {
                "type": "collection_selection",
                "wrong_collection": wrong,
                "correct_collection": correct,
            },
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def _build_field_mapping_dataset(corrections, output_path):
    rows = []
    for item in corrections:
        question = str(item.get("question") or "").strip()
        wrong_flds = item.get("wrong_fields") or []
        correct_flds = item.get("correct_fields") or []
        if not question or not correct_flds:
            continue
        pairs = []
        for w, c in zip(wrong_flds, correct_flds):
            pairs.append(f"{w} -> {c}")
        rows.append({
            "instruction": "Map incorrect field names to the correct ERP field names.",
            "input": f"Query: {question}\nIncorrect fields: {', '.join(wrong_flds)}",
            "output": f"Correct mappings: {'; '.join(pairs)}",
            "metadata": {
                "type": "field_mapping",
                "wrong_fields": list(wrong_flds),
                "correct_fields": list(correct_flds),
            },
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return len(rows)


def run_training_pipeline():
    logger.info("Training pipeline started.")

    try:
        dataset_path, row_count = build_retrain_dataset()
        logger.info("Retrain dataset built at %s (%d rows)", dataset_path, row_count)

        col_corrections = collection_correction_rows()
        if col_corrections:
            col_path = Path(RETRAIN_DATASET_PATH).parent / "collection_corrections.json"
            n = _build_collection_selection_dataset(col_corrections, col_path)
            logger.info("Collection-correction dataset: %d rows at %s", n, col_path)

        fld_corrections = field_correction_rows()
        if fld_corrections:
            fld_path = Path(RETRAIN_DATASET_PATH).parent / "field_corrections.json"
            n = _build_field_mapping_dataset(fld_corrections, fld_path)
            logger.info("Field-correction dataset: %d rows at %s", n, fld_path)

        if row_count == 0 and not col_corrections and not fld_corrections:
            logger.info("No training data available. Skipping model update.")
            return {"trained": False, "reason": "No training data available.", "rows": 0}

        _maybe_run_fine_tuning(dataset_path)

        mark_trained()
        logger.info("Training pipeline complete — counter reset.")
        return {"trained": True, "rows": row_count, "dataset": str(dataset_path)}

    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        observe_error("/train", str(exc))
        return {"trained": False, "error": str(exc)}


def _maybe_run_fine_tuning(dataset_path):
    try:
        from train import train as run_train
    except ImportError:
        logger.info("train.py not available — skipping model fine-tuning.")
        return

    try:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Starting fine-tuning with dataset %s", dataset_path)
        run_train(dataset_path=str(dataset_path), output_dir=ADAPTER_DIR)
        torch.cuda.empty_cache()
        gc.collect()
        from erp_backend.llm.runtime import clear_model_cache
        clear_model_cache()
        logger.info("Fine-tuning complete. Model cache cleared.")
    except Exception as exc:
        logger.exception("Fine-tuning step failed: %s", exc)
        observe_error("/train/fine_tune", str(exc))


async def trigger_training():
    global _TRAINING_IN_PROGRESS
    if _TRAINING_IN_PROGRESS:
        return {"trained": False, "reason": "Training already in progress."}

    async with _TRAINER_LOCK:
        if _TRAINING_IN_PROGRESS:
            return {"trained": False, "reason": "Training already in progress."}
        _TRAINING_IN_PROGRESS = True

    try:
        info_before = get_training_info()
        logger.info("Manual training triggered. Feedback count: %d", info_before["feedback_count"])
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_training_pipeline)
        result["info_before"] = info_before
        result["info_after"] = get_training_info()
        logger.info("Training result: %s", result)
        return result
    finally:
        _TRAINING_IN_PROGRESS = False


async def background_training_loop(interval_seconds=300):
    if not CONTINUOUS_TRAINING_ENABLED:
        return
    logger.info("Background trainer active (manual trigger only — threshold disabled).")
    while True:
        await asyncio.sleep(interval_seconds)
