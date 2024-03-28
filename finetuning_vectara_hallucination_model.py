import json
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import math
from sklearn.metrics import accuracy_score

model = CrossEncoder('vectara/hallucination_evaluation_model')
with open ('path_to_validation_dataset/val.model-agnostic.json', 'r') as f:
    val_agnostic = json.load(f)
with open ('path_to_validation_dataset/val.model-aware.v2.json', 'r') as f:
    val_aware = json.load(f)
with open ('path_to_trial_set/trial-v1.json', 'r') as f:
    trial_dataset = json.load(f)

val_dataset=val_agnostic+val_aware
train_examples, test_examples = [], []
for i in range(len(val_dataset)):
    if val_dataset['tgt']=="":
        premise=val_dataset['src']
    else:
        premise=val_dataset['tgt']
    train_examples.append(InputExample(texts=[premise, val_dataset['hyp']], label=(round(1-val_dataset['p(Hallucination)']))))

for i in range(len(trial_dataset)):
    if trial_dataset['tgt']=="":
        premise=trial_dataset['src']
    else:
        premise=trial_dataset['tgt']
    train_examples.append(InputExample(texts=[premise, trial_dataset['hyp']], label=(round(1-trial_dataset['p(Hallucination)']))))
test_evaluator = CECorrelationEvaluator.from_input_examples(test_examples, name='sts-dev')

num_epochs = 5
model_save_path = "./model_dump"

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
model.fit(train_dataloader=train_dataloader,
          evaluator=test_evaluator,
          epochs=5,
          evaluation_steps=10_000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          show_progress_bar=True)

# Calculate accuracy on the model-aware test set (same with agnostic)
with open ('path_to_test_model_aware/test.model-aware.json', 'r') as f:
    test_aware = json.load(f)
for i in range(len(test_aware)):
    if test_aware[i]['tgt']=="":
        premise=[test_aware[i]['src'],test_aware[i]['hyp']]
    else:
        premise=[test_aware[i]['tgt'],test_aware[i]['hyp']]
    score=model.predict(premise)
    if score<0.5:
        test_aware[i]['pred_label']='Hallucination'
    else:
        test_aware[i]['pred_label']='Not Hallucination'
    test_aware[i]['pred_p(Hallucination)']=float(score)
y_true=[]
y_pred=[]
for i in range(len(test_aware)):
    y_true.append(test_aware[i]['label'])
    y_pred.append(test_aware[i]['pred_label'])
print(accuracy_score(y_true, y_pred))