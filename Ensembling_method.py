from collections import Counter
import json
from sklearn.metrics import accuracy_score

with open('path_to_results_from_method_1/test.model-agnostic.json') as f:
    agnostic1 = json.load(f)
with open('path_to_results_from_method_2/test.model-agnostic.json') as f:
    agnostic2 = json.load(f)
with open('path_to_results_from_method_3/test.model-agnostic.json') as f:
    agnostic3 = json.load(f)

with open('path_to_results_from_method_1/test.model-aware.json') as f:
    aware1 = json.load(f)
with open('path_to_results_from_method_2/test.model-aware.json') as f:
    aware2 = json.load(f)
with open('path_to_results_from_method_3/test.model-aware.json') as f:
    aware3 = json.load(f)

with open ('path_to_test_model_aware/test.model-aware.json', 'r') as f:
    test_aware = json.load(f)
y_true=[]
y_pred=[]
for i in range(len(agnostic1)):
    y_pred.append(Counter([agnostic1[i]['label'],agnostic2[i]['label'],agnostic3[i]['label']]).most_common(1)[0][0])
    y_true.append(test_aware[i]['label'])
print(accuracy_score(y_true, y_pred))
