from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    AdamW,
    get_scheduler,
)
import torch
from src.utils import get_datasets, DataCollator
import os
import evaluate
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Constants
data_dir = "/app/training_data"
metadata_file = os.path.join(data_dir, "metadata.csv")
models_dir = "/app/models"
runs_dir = "/app/runs" # For models checkpoints
logs_dir = "/app/logs" # For tensorboard logs
num_classes = 8
id2label = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}
label2id = {v: k for k, v in id2label.items()}

# Initialize processor and model
model_names = [
    "ntu-spml/distilhubert",
    "facebook/wav2vec2-base"
]
for model_name in model_names:
    model_id = model_name.split("/")[-1]

    model_runs_dir = os.path.join(runs_dir, model_id)
    model_logs_dir = os.path.join(logs_dir, model_id)
    model_output_dir = os.path.join(models_dir, model_id)

    os.makedirs(model_runs_dir, exist_ok=True)
    os.makedirs(model_logs_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    processor = AutoFeatureExtractor.from_pretrained(model_name)

    # Get datasets
    train_dataset, val_dataset = get_datasets(data_dir, metadata_file)
    data_collator = DataCollator(processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_runs_dir,      # Checkpoints for this model
        logging_dir=model_logs_dir,     # Logs for this model
        num_train_epochs=10,                   # num_epochs
        per_device_train_batch_size=8,         # train_batch_size
        per_device_eval_batch_size=8,          # eval_batch_size
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["tensorboard"],      # Enable TensorBoard logging
        dataloader_num_workers=4,
        learning_rate=5e-5,                    # learning_rate
        seed=seed,                             # seed
        lr_scheduler_type="linear",            # lr_scheduler_type
        warmup_ratio=0.1,                      # lr_scheduler_warmup_ratio
    )

    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),  # optimizer betas
        eps=1e-08,           # optimizer epsilon
    )
    num_update_steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    if len(train_dataset) % training_args.per_device_train_batch_size != 0:
        num_update_steps_per_epoch += 1
    total_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * total_training_steps),
        num_training_steps=total_training_steps,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
    )

    # Train
    trainer.train()

    model.save_pretrained(model_output_dir)
    processor.save_pretrained(model_output_dir)
