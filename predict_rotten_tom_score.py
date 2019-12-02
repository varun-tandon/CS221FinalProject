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
    LogisticRegression(solver='sag', max_iter=1000, random_state=1, n_jobs=-1, verbose=True),
    # KNeighborsClassifier(n_neighbors=4, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=315, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=325, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=335, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=345, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=355, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=365, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=375, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=385, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=395, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=415, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=425, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=435, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=445, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=455, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=465, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=475, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=485, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=495, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=20, n_jobs=-1),
    # KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
    # LinearSVC(max_iter=1000),
    # svm.SVC(gamma='scale'),
    BaggingClassifier(LogisticRegression(solver='sag', max_iter=1000, random_state=1, verbose=True), n_jobs=-1),
    # RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
    # GaussianNB(),
    # MLPClassifier(hidden_layer_sizes=(100, 50, 10), early_stopping=True),
]

'''
--------------
Accuracy: 0.5032537960954447. F1: 0.5873873873873874. ROC AUC: 0.5169694794893103.
TN: 138. FP: 350. FN: 108. TP: 326
--------------
Accuracy: 0.53470715835141. F1: 0.5994397759103641. ROC AUC: 0.5460451763994862.
TN: 172. FP: 316. FN: 113. TP: 321
--------------
Accuracy: 0.5466377440347071. F1: 0.6172161172161172. ROC AUC: 0.5593554053033165.
TN: 167. FP: 321. FN: 97. TP: 337
--------------
Accuracy: 0.5509761388286334. F1: 0.6049618320610687. ROC AUC: 0.5609040945833648.
TN: 191. FP: 297. FN: 117. TP: 317
--------------
Accuracy: 0.5759219088937093. F1: 0.6109452736318408. ROC AUC: 0.5831948326660119.
TN: 224. FP: 264. FN: 127. TP: 307
--------------
Accuracy: 0.5856832971800434. F1: 0.6069958847736626. ROC AUC: 0.5908863413160081.
TN: 245. FP: 243. FN: 139. TP: 295
--------------
Accuracy: 0.5683297180043384. F1: 0.5871369294605809. ROC AUC: 0.5729630958676437.
TN: 241. FP: 247. FN: 151. TP: 283
--------------
Accuracy: 0.579175704989154. F1: 0.5932914046121593. ROC AUC: 0.5832089975069881.
TN: 251. FP: 237. FN: 151. TP: 283
--------------
Accuracy: 0.5770065075921909. F1: 0.5903361344537816. ROC AUC: 0.5809048500415502.
TN: 251. FP: 237. FN: 153. TP: 281
--------------
--------------115
Accuracy: 0.5802603036876356. F1: 0.5964546402502608. ROC AUC: 0.5846160383772758.
TN: 249. FP: 239. FN: 148. TP: 286
--------------
Accuracy: 0.5813449023861171. F1: 0.5953878406708596. ROC AUC: 0.5853856614036413.
TN: 252. FP: 236. FN: 150. TP: 284
--------------
Accuracy: 0.5726681127982647. F1: 0.5781584582441114. ROC AUC: 0.5754041701291833.
TN: 258. FP: 230. FN: 164. TP: 270
--------------
Accuracy: 0.5780911062906724. F1: 0.5821697099892589. ROC AUC: 0.58065460451764.
TN: 262. FP: 226. FN: 163. TP: 271
--------------
Accuracy: 0.571583514099783. F1: 0.572972972972973. ROC AUC: 0.5737421621213266.
TN: 262. FP: 226. FN: 169. TP: 265
--------------
Accuracy: 0.5650759219088937. F1: 0.5683530678148547. ROC AUC: 0.5674671375689355.
TN: 257. FP: 231. FN: 170. TP: 264
--------------
Accuracy: 0.5748373101952278. F1: 0.5802997858672377. ROC AUC: 0.5775808340258367.
TN: 259. FP: 229. FN: 163. TP: 271
--------------
Accuracy: 0.5802603036876356. F1: 0.5816216216216217. ROC AUC: 0.5824488177079399.
TN: 266. FP: 222. FN: 165. TP: 269
--------------
Accuracy: 0.579175704989154. F1: 0.5773420479302832. ROC AUC: 0.5809142932688676.
TN: 269. FP: 219. FN: 169. TP: 265
--------------
Accuracy: 0.5813449023861171. F1: 0.5849462365591398. ROC AUC: 0.5838558585782276.
TN: 264. FP: 224. FN: 162. TP: 272
--------------
------------- 215
Accuracy: 0.579175704989154. F1: 0.5717439293598233. ROC AUC: 0.5801493918561608.
TN: 275. FP: 213. FN: 175. TP: 259
--------------
Accuracy: 0.5845986984815619. F1: 0.5701459034792369. ROC AUC: 0.5846349248319106.
TN: 285. FP: 203. FN: 180. TP: 254
--------------
Accuracy: 0.5878524945770065. F1: 0.5701357466063349. ROC AUC: 0.5874537281861448.
TN: 290. FP: 198. FN: 182. TP: 252
--------------
Accuracy: 0.586767895878525. F1: 0.5665529010238908. ROC AUC: 0.5860466873158571.
TN: 292. FP: 196. FN: 185. TP: 249
--------------
Accuracy: 0.6008676789587852. F1: 0.5710955710955712. ROC AUC: 0.5988564251718668.
TN: 309. FP: 179. FN: 189. TP: 245
--------------
Accuracy: 0.5900216919739696. F1: 0.5542452830188679. ROC AUC: 0.5873356878446778.
TN: 309. FP: 179. FN: 199. TP: 235
--------------
Accuracy: 0.5976138828633406. F1: 0.5609467455621302. ROC AUC: 0.5947627861297877.
TN: 314. FP: 174. FN: 197. TP: 237
--------------
Accuracy: 0.6030368763557483. F1: 0.5600961538461537. ROC AUC: 0.5993758026743219.
TN: 323. FP: 165. FN: 201. TP: 233
--------------
Accuracy: 0.5878524945770065. F1: 0.5343137254901961. ROC AUC: 0.5831192868474729.
TN: 324. FP: 164. FN: 216. TP: 218
--------------
'''

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

# estimators = []
# i = 0
# for clf in classifiers:
#     estimators.append((str(i), clf))
#     i += 1

# clf = VotingClassifier(estimators, n_jobs=-1)
# print("--------------")
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc = roc_auc_score(y_test, y_pred)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
# print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))
# print(clf.coef_)