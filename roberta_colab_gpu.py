
# ===============================
# RoBERTa Text Classification - Google Colab GPU Ready
# ===============================

# Install dependencies (uncomment if running first time in Colab)
# !pip install transformers datasets accelerate -q

import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ===============================
# 1. Check for GPU
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# 2. Load Dataset (Example: IMDB)
# ===============================

dataset = load_dataset("imdb")

num_labels = 2
model_name = "roberta-base"

# ===============================
# 3. Load Tokenizer
# ===============================

tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_dataset = dataset.map(tokenize_function, batched=True)

encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch")

# ===============================
# 4. Load Model
# ===============================

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

model.to(device)

# ===============================
# 5. Metrics
# ===============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }

# ===============================
# 6. Training Arguments
# ===============================

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
)

# ===============================
# 7. Trainer
# ===============================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ===============================
# 8. Train
# ===============================

trainer.train()

# ===============================
# 9. Evaluate
# ===============================

results = trainer.evaluate()
print(results)
