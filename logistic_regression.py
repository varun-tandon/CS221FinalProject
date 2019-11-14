from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import json

def bool_to_int(x):
    return 1 if x else 0

df = pd.read_csv('data_parsed_contains_word_features.csv')
df['y'] = df['IsProfitable'].apply(bool_to_int)

features = ['contains_word_features', 'num_words_features', 'num_chars_features']
examples_list = []
for feature in features:
    examples = df[feature]
    if len(examples_list) == 0:
        for example in examples:
            examples_list.append(json.loads(example))
    else:
        for i in range(len(examples)):
            examples_list[i].update(json.loads(examples[i]))

v = DictVectorizer()
X = v.fit_transform(examples_list)
y = df['y'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=500).fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1_score, roc))
print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))
