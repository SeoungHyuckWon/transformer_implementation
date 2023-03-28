import numpy as np
import pandas as pd

from transformers import AutoTokenizer,DataCollatorForSeq2Seq,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
import evaluate

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

train_data = pd.read_csv('/Users/wonseonghyuck/Desktop/python/AI/Data/train.csv')
train_list  = list()
for v in train_data.values:
    temp = dict()
    temp['id'] = str(v[0])
    temp['translation'] = dict()
    temp['translation']['ko'] = v[1]
    temp['translation']['en'] = v[2]
    train_list.append(temp)
train = datasets.Dataset.from_list(train_list)

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
source_lang = "ko"
target_lang = "en"
prefix = "translate English to Korean: "

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
books = train.train_test_split(test_size=0.2)
metric = evaluate.load("sacrebleu")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=books["train"],
    eval_dataset=books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
text = "오늘 나는 아침밥을 먹지 않아서 배고파."
tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
inputs = tokenizer(text, return_tensors="pt").input_ids
model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
final = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(final)