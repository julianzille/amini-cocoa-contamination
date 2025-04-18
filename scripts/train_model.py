from transformers import (
    AutoConfig,
    RTDetrV2ForObjectDetection,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from constants import BASE_MODEL, DATASET_REPO, ID2LABEL, LABEL2ID
from src.data_utils import CocoaDatasetManager
from src.evaluation import MAPEvaluator


def main():
    dataset_manager = CocoaDatasetManager(BASE_MODEL, DATASET_REPO)
    config = AutoConfig.from_pretrained(
        BASE_MODEL, label2id=LABEL2ID, id2label=ID2LABEL
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(
        BASE_MODEL, config=config, ignore_mismatched_sizes=True
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
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=2,
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
