from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments

from reader.config import DEFAULT_DOC_STRIDE, DEFAULT_MAX_LENGTH, validate_window_config
from reader.data_utils import (
    build_text_preprocessor,
    get_tokenizer,
    load_qa_dataset,
    model_input_features,
    prepare_train_features,
    prepare_validation_features,
)
from reader.evaluate import (
    evaluate,
    flatten_metrics,
    postprocess_validation_logits,
    sweep_thresholds,
)


def _run_name(model_name: str, max_length: int, stride: int, lr: float, epochs: float) -> str:
    family = "phobert_qa" if "phobert" in model_name.lower() else Path(model_name).name.replace("-", "_")
    learning_rate = f"{lr:.0e}".replace("-0", "-").replace("+0", "+")
    epoch_text = str(int(epochs)) if float(epochs).is_integer() else str(epochs).replace(".", "p")
    return f"{family}_ml{max_length}_stride{stride}_lr{learning_rate}_ep{epoch_text}"


def _build_compute_metrics(examples, validation_features, preprocessor):
    """Return full decoded QA metrics for Hugging Face checkpoint selection."""

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple) and len(predictions) > 2:
            predictions = predictions[:2]
        raw = postprocess_validation_logits(
            examples,
            validation_features,
            predictions,
            preprocessor,
        )
        _, best, _, metrics = sweep_thresholds(raw, selection_metric="reader_priority_score")
        result = flatten_metrics(metrics, best_threshold=best["threshold"])
        result["threshold_reader_priority_score"] = best["reader_priority_score"]
        answerable = metrics["answerable"]
        print(
            "QA validation: "
            f"answerable_f1={answerable['f1']:.2f}, "
            f"answerable predicted empty={answerable['predicted_empty']}/{answerable['count']} "
            f"({answerable['predicted_empty_rate']:.2f}%), "
            f"threshold={best['threshold']:.4f}"
        )
        return result

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clean extractive-QA PhoBERT baseline")
    parser.add_argument("--model_name", default="vinai/phobert-base-v2")
    parser.add_argument("--data_variant", default="clean", choices=["clean", "segmented", "auto"])
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc_stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--subset_size", type=int, default=-1, help="Smoke tests only")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_dir", default="models/reader", help="Root for final model artifacts")
    parser.add_argument("--results_root", default="results/reader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Point qa_pipeline.json at this fully evaluated checkpoint",
    )
    args = parser.parse_args()

    if args.data_variant == "segmented":
        print(
            "NOTE: legacy segmented parquet columns are intentionally ignored. "
            "Raw text is segmented exactly once by the shared preprocessor."
        )
    if not 64 <= args.doc_stride <= 96:
        raise ValueError("PhoBERT baseline doc_stride must be in the audited 64-96 range")
    if args.max_seq_len != DEFAULT_MAX_LENGTH:
        raise ValueError("PhoBERT baseline max_seq_len must be 256")

    run_name = args.run_name or _run_name(
        args.model_name,
        args.max_seq_len,
        args.doc_stride,
        args.lr,
        args.epochs,
    )
    model_output_dir = ROOT / args.output_dir / run_name
    result_dir = ROOT / args.results_root / run_name
    checkpoint_dir = result_dir / "checkpoints"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_file = ROOT / "data" / "processed" / "viquad_train_clean.parquet"
    validation_file = ROOT / "data" / "processed" / "viquad_val_clean.parquet"
    if not train_file.exists() or not validation_file.exists():
        raise FileNotFoundError("Processed clean train/validation parquet files are required")

    print(f"Run: {run_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(
        f"Config: model={args.model_name}, max_length={args.max_seq_len}, "
        f"stride={args.doc_stride}, lr={args.lr}, epochs={args.epochs}, loss=standard CE"
    )
    tokenizer = get_tokenizer(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    validate_window_config(args.max_seq_len, args.doc_stride, model, tokenizer)
    preprocessor = build_text_preprocessor(
        args.model_name,
        tokenizer=tokenizer,
        model_config=model.config,
    )

    train_dataset = load_qa_dataset(str(train_file), subset_size=args.subset_size)
    validation_dataset = load_qa_dataset(str(validation_file), subset_size=args.subset_size)
    if args.subset_size <= 0 and len(validation_dataset) != 3814:
        raise ValueError(f"Final model selection requires all 3,814 validation examples, found {len(validation_dataset)}")

    tokenized_train = train_dataset.map(
        lambda batch: prepare_train_features(
            batch,
            tokenizer,
            args.max_seq_len,
            args.doc_stride,
            preprocessor,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training examples with answer_start",
    )
    validation_features = validation_dataset.map(
        lambda batch: prepare_validation_features(
            batch,
            tokenizer,
            args.max_seq_len,
            args.doc_stride,
            preprocessor,
        ),
        batched=True,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing validation examples with answer_start",
    )
    trainer_validation = model_input_features(validation_features)
    print(f"Train features: {len(tokenized_train)}; validation features: {len(validation_features)}")

    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        save_total_limit=2,
        logging_steps=100 if args.subset_size <= 0 else 10,
        load_best_model_at_end=True,
        metric_for_best_model="answerable_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=trainer_validation,
        processing_class=tokenizer,
        compute_metrics=_build_compute_metrics(
            validation_dataset,
            validation_features,
            preprocessor,
        ),
    )
    trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    training_manifest = {
        **vars(args),
        "run_name": run_name,
        "model_output_dir": str(model_output_dir),
        "checkpoint_selection_metric": "answerable_f1",
        "checkpoint_metrics_are_postprocessed": True,
        "threshold_selection_metric": "reader_priority_score_0.7_answerable_f1_0.3_unanswerable_accuracy",
        "loss": "standard_cross_entropy_start_end",
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "train_features": len(tokenized_train),
        "validation_features": len(validation_features),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    (result_dir / "training_args.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # Re-run the final selected model through exactly the production Predictor,
    # then persist threshold sweep, predictions, metrics, and error analysis.
    evaluate(
        str(model_output_dir),
        subset_size=args.subset_size,
        use_cpu=not torch.cuda.is_available(),
        output_dir=str(result_dir),
        batch_size=args.eval_batch_size,
        max_seq_len=args.max_seq_len,
        doc_stride=args.doc_stride,
        write_profile_to_checkpoint=args.subset_size <= 0,
    )
    if args.promote:
        if args.subset_size > 0:
            raise ValueError("Smoke/subset checkpoints cannot be promoted")
        pipeline_path = ROOT / "config" / "qa_pipeline.json"
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        pipeline["reader_checkpoint"] = str(model_output_dir.relative_to(ROOT)).replace("\\", "/")
        pipeline["reader_score_margin_threshold"] = None
        pipeline["require_calibrated_reader_profile"] = True
        pipeline_path.write_text(
            json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Promoted checkpoint pointer in {pipeline_path}")
    print(f"Training complete. Model: {model_output_dir}; results: {result_dir}")


if __name__ == "__main__":
    main()
