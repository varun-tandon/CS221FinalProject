from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def bool_to_int(x):
    return 1 if x else 0

df = pd.read_csv('data_parsed_contains_word_features.csv')
df['y'] = df['IsProfitable'].apply(bool_to_int)

features = ['contains_director_features']
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)

#predicitng
y_est = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_est)