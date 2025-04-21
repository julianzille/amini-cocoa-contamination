import argparse

from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from src.constants import BASE_MODEL, DATASET_REPO, ID2LABEL, LABEL2ID
from src.data_utils import CocoaDatasetManager
from src.evaluation import MAPEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_REPO,
        help="Dataset repository or path",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default="main",
        help="Dataset revision or branch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL,
        help="Base model to use for training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_manager = CocoaDatasetManager(
        args.base_model, args.dataset, revision=args.dataset_revision
    )
    config = AutoConfig.from_pretrained(
        args.base_model, label2id=LABEL2ID, id2label=ID2LABEL, decoder_method="discrete"
    )

    model = AutoModelForObjectDetection.from_pretrained(
        args.base_model,
        config=config,
        ignore_mismatched_sizes=True,
    )

    train_dataset = dataset_manager.get_train_dataset()
    val_dataset = dataset_manager.get_val_dataset()

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=dataset_manager.image_processor,
        threshold=0.01,
        id2label=ID2LABEL,
    )

    training_args = TrainingArguments(
        output_dir="models/disease-detection",
        num_train_epochs=10,
        max_grad_norm=5.0,
        learning_rate=5e-5,
        warmup_steps=300,
        save_steps=500,
        eval_steps=500,
        per_device_train_batch_size=args.batch_size,
        report_to=["tensorboard"],
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        eval_on_start=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=dataset_manager.image_processor,
        data_collator=dataset_manager.collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    eval_results = trainer.evaluate(val_dataset)  # type: ignore
    trainer.save_metrics("eval", eval_results)
    trainer.save_state()


if __name__ == "__main__":
    main()
