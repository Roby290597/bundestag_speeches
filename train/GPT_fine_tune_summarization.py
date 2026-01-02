# fine_tune_summarization_fixed.py

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import numpy as np
import os

# --- Checkpoint / Tokenizer / Model ---
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# --- Data collator: model muss das Model-Objekt sein ---
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # labels tokenizen — nutzen text_target für Seq2Seq-tokenizer
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # optional: convert padding token id -> -100 falls vorhanden, hier nicht nötig solange collator das macht
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Dataset ---
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.9)
tokenized_billsum = billsum.map(preprocess_function, batched=True)

# --- Rouge (lade nur falls du es nutzt) ---
#from datasets import load_metric


from rouge_score import rouge_scorer


senetence1 = "The quick brown fox jumps on the cat."
senetence2 = "The quick brown fox jumps on the dog."

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
scores = scorer.score(senetence1, senetence2)


#rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Wenn predictions Logits sind: argmax -> token ids
    if isinstance(predictions, tuple) or predictions.ndim == 3:
        # some trainers return (logits, ...) or logits with shape (batch, seq, vocab)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=-1)

    # replace -100 in labels for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = scorer.score(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="first_trained_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # evtl. zu groß, an GPU/RAM anpassen
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,  # nur wenn CUDA + passende accelerate Version
    push_to_hub=True,
    # WORKAROUND: falls du die "unwrap_model" TypeError siehst, setze torch_compile=False
    # torch_compile=False,
)

# --- Trainer: tokenizer statt processing_class ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,            # <-- fix: korrekter Param-Name
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Train! ---
trainer.train()

# optional: push manually (falls Probleme mit env vars)
# trainer.push_to_hub()
