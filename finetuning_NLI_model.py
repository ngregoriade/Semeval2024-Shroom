import torch
from transformers import TrainingArguments, Trainer
from sentence_transformers import CrossEncoder
import tqdm
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import evaluate
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
def label2id(example):
  if example["label"] == "Hallucination":
    example["label"] = 1
  else:
    example["label"] = 0
  return example
def tokenize_function(examples):
    if examples['tgt']=="":
        premise=examples['src']
    else:
        premise=examples['tgt']
    hypothesis=examples['hyp']

    input = tokenizer(premise, hypothesis, truncation=True,padding="max_length", return_tensors="pt")
    return input


with open ('path_to_validation_dataset/val.model-agnostic.json', 'r') as f:
    val_agnostic = json.load(f)
with open ('path_to_validation_dataset/val.model-aware.v2.json', 'r') as f:
    val_aware = json.load(f)
with open ('path_to_trial_set/trial-v1.json', 'r') as f:
    trial_dataset = json.load(f)

val_dataset=val_agnostic+val_aware

training_args = TrainingArguments(
    num_train_epochs=1,              # total number of training epochs
    learning_rate=2e-05,
    per_device_train_batch_size=32,   # batch size per device during training
    gradient_accumulation_steps=2,   # to double the effective batch size for
    warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    fp16=False,

    output_dir="path_to_save_model",          # output directory
    evaluation_strategy="epoch"
)

import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3,ignore_mismatched_sizes=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
metric = evaluate.load("accuracy")
path_val_model_agnostic='path_dataset/val_all_datas.json'
train = load_dataset('json', data_files={"val": path_val_model_agnostic})
train=train['val']
train = train.map(label2id)
train_dataset = train.map(tokenize_function, batched=True)

path_val_model_agnostic='path_dataset/trial-v1.json'
val = load_dataset('json', data_files={"val": path_val_model_agnostic})
val=val['val']
val=val.map(label2id)
val_dataset = val.map(tokenize_function, batched=True)

for name, param in model.named_parameters():
     if name.startswith("deberta"):
        param.requires_grad = False
        print(name)
     else:
       print("NO", name)
val_dataset = val_dataset.remove_columns(['labels', 'ref','tgt', 'src', 'task', 'p(Hallucination)', 'model', 'hyp',])
train_dataset = train_dataset.remove_columns(['labels', 'ref','tgt', 'src', 'task', 'p(Hallucination)', 'model', 'hyp',])
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "path_to_saved_model/test_trainer_check1"
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3,ignore_mismatched_sizes=True)

model.to(device)
training_args = TrainingArguments(
    num_train_epochs=5,              # total number of training epochs
    learning_rate=2e-05,
    per_device_train_batch_size=2,   # batch size per device during training
    gradient_accumulation_steps=2,   # to double the effective batch size for
    warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    fp16=False,

    output_dir="test_trainer_new",
    evaluation_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
for name, param in model.named_parameters():
  param.requires_grad = True
  print(name)
trainer.train()
trainer.save_model("finetuned_model_final")

cluste_list=[]
submit=[]
import tqdm
with open ('path_to_test_set/test.model-aware.json', 'r') as f:
    test_aware = json.load(f)
for i in tqdm.tqdm(range(len(test_aware))):
    if test_aware[i]['ref']=='src':
        premise = test_aware[i]['src']
    else:
        premise = test_aware[i]['tgt']
    hypothesis = test_aware[i]['hyp']
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    cluste_list.append([prediction['entailment'],prediction['neutral'],prediction['contradiction']])
    if prediction['entailment']>80:
        test_aware[i]['pred_label']='Not Hallucination'
    else:
        test_aware[i]['pred_label']='Hallucination'

y_true=[]
y_pred=[]
for i in range(len(test_aware)):
    y_true.append(test_aware[i]['label'])
    y_pred.append(test_aware[i]['pred_label'])
print(accuracy_score(y_true, y_pred))
