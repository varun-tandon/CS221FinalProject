from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import json
import random
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

def bool_to_int(x):
    return 1 if x == 'Fresh' or x == 'Certified Fresh' else 0

df = pd.read_csv('data_parsed_contains_word_features.csv')
df['y'] = df['tomatometer_status'].apply(bool_to_int)
df.to_csv('sanity.csv')
features = [
    'contains_director_features', 
    'genre_features', 
    'contains_cast_features', 
    'runtime_features', 
    # 'movie_title_features',
    'movie_rating_features',
    # 'movie_desc_features'
]
examples_list = []
for feature in features:
    examples = df[feature]
    if len(examples_list) == 0:
        for example in examples:
            examples_list.append(json.loads(example))
    else:
        for i in range(len(examples)):
            examples_list[i].update(json.loads(examples[i]))

import pickle
important_features = set(pickle.load(open('important_features.pkl', 'rb')))
print(len(important_features))
input()
for example in examples_list:
    for key in list(example):
        if key not in important_features:
            del example[key]

pickle.dump(examples_list, open('examples_w_important_feats_only.pkl', 'wb'))

v = DictVectorizer()
X = v.fit_transform(examples_list)
y = df['y'].to_numpy()
print(X.shape)

# sm = SMOTE()
# X, y = sm.fit_resample(X, y)
X = X.toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

################## CROSS VALIDATION ##################

# Logistic regression
prev_score = float('-Inf')
final_c = 1
for c in range(1, 11, 1):
    print('testing C value:', c/10)
    model = LogisticRegression(C = (c/10), solver='saga', max_iter=3000, random_state=1, n_jobs=-1, verbose=True)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    the_c = c/10
    score = sum(scores) / len(scores)
    if score > prev_score:
        prev_score = score
        final_c = the_c

#knn
prev_score = float('-Inf')
final_k = 1
potential_ks = [3,4,5, 20, 25]
potential_ks += range(315, 505, 10)
for k in potential_ks:
    print('testing k value:', k)
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    score = sum(scores) / len(scores)
    if score > prev_score:
        prev_score = score
        final_k = k

#random forests
prev_score = float('-Inf')
final_n = 1
for n in range(0, 200, 10):
    print('testing n value:', k)
    model = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=1)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    score = sum(scores) / len(scores)
    if score > prev_score:
        prev_score = score
        final_n = n


################################################################################

classifiers = [
    LogisticRegression(C=final_c, solver='saga', max_iter=3000, random_state=1, n_jobs=-1, verbose=True),
    KNeighborsClassifier(n_neighbors=final_k, n_jobs=-1),
    LinearSVC(max_iter=1000),
    svm.SVC(gamma='scale'),
    BaggingClassifier(LogisticRegression(solver='sag', max_iter=1000, random_state=1, verbose=True), n_jobs=-1),
    RandomForestClassifier(n_estimators=final_n, n_jobs=-1, random_state=1),
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(100, 50, 10), early_stopping=True),
]

for clf in classifiers:
    print("--------------")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
    print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))


#Voting classifier

estimators = []
i = 0
for clf in classifiers:
    estimators.append((str(i), clf))
    i += 1

clf = VotingClassifier(estimators, n_jobs=-1)
print("--------------")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))
print(clf.coef_)
