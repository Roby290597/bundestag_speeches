########################################################
#####  Imports and tokenizer setup  ####################
########################################################



from transformers import AutoTokenizer

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# import torch
# from transformers import BertTokenizerFast, EncoderDecoderModel
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ckpt = 'mrm8488/bert2bert_shared-german-finetuned-summarization'
# tokenizer = BertTokenizerFast.from_pretrained(ckpt)
# model = EncoderDecoderModel.from_pretrained(ckpt).to(device)
# def generate_summary(text):
#    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#    input_ids = inputs.input_ids.to(device)
#    attention_mask = inputs.attention_mask.to(device)
#    output = model.generate(input_ids, attention_mask=attention_mask)
#    return tokenizer.decode(output[0], skip_special_tokens=True)
   
# text = "Your text here..."

# generate_summary(text)


################################################################################
##### Model creation using the checkpoint ######################################
################################################################################


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

################################################################################
##### Load the dataset for fine-tuning #########################################
################################################################################



from datasets import load_dataset

billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.999)
tokenized_billsum = billsum.map(preprocess_function, batched=True)
#### Just a subset 








import numpy as np
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=False)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = scorer.score(decoded_preds, decoded_labels)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir="first_trained_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    #eval_dataset=tokenized_billsum["test"],
    eval_dataset=tokenized_billsum["train"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()



#trainer.push_to_hub()
