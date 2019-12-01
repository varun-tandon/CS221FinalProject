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
import json
import random
from imblearn.over_sampling import SMOTE

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

v = DictVectorizer()
X = v.fit_transform(examples_list)
y = df['y'].to_numpy()
print(X.shape)

# sm = SMOTE()
# X, y = sm.fit_resample(X, y)
X = X.toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifiers = [
    LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1, n_jobs=-1, verbose=True),
    KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    LinearSVC(max_iter=1000),
    svm.SVC(gamma='scale'),
    BaggingClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1), n_jobs=-1),
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
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
# print(clf.coef_)