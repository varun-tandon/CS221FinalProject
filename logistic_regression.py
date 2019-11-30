from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
from imblearn.over_sampling import SMOTE

def bool_to_int(x):
    return 1 if x else 0

df = pd.read_csv('data_parsed_contains_word_features.csv')
df['y'] = df['IsProfitable'].apply(bool_to_int)

features = ['script_author_features']
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

sm = SMOTE()
X, y = sm.fit_resample(X, y)
X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=500).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc = roc_auc_score(y_test, y_pred)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
# print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))

# clf = LinearSVC(random_state=0, tol=1e-5, max_iter=30000)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))