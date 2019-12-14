import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
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

rotten_tom_df = pd.read_csv('rotten_tomatoes_movies_matched.csv')
budgets_df = pd.read_csv('movie_budgets.csv')
distinct_df = pd.read_csv('data_parsed_contains_word_features.csv')

genre_columns = ['genre_action_and_adventure', 'genre_animation',
       'genre_anime_and_manga', 'genre_arthouse_and_international',
       'genre_classics', 'genre_comedy', 'genre_cultmovies',
       'genre_documentary', 'genre_drama', 'genre_faith_and_spirituality',
       'genre_gay_and_lesbian', 'genre_horror', 'genre_kids_and_family',
       'genre_musical_and_performingarts', 'genre_mystery_and_suspense',
       'genre_romance', 'genre_sciencefiction_and_fantasy',
       'genre_specialinterest', 'genre_sports_and_fitness', 'genre_television',
       'genre_western']

for genre_column in genre_columns:
    genre_df = distinct_df[distinct_df[genre_column] == 1].reset_index()
    labels = ['R', 'F']
    profitable_movies = [
        genre_df[(genre_df.tomatometer_status == 'Rotten') & (genre_df.WorldwideGross > genre_df.ProductionBudget)].shape[0],
        genre_df[((genre_df.tomatometer_status == 'Fresh') | (genre_df.tomatometer_status == 'Certified Fresh')) & (genre_df.WorldwideGross > genre_df.ProductionBudget)].shape[0],
    ]
    unprofitable_movies = [
        genre_df[(genre_df.tomatometer_status == 'Rotten') & (genre_df.WorldwideGross <= genre_df.ProductionBudget)].shape[0],
        genre_df[((genre_df.tomatometer_status == 'Fresh') | (genre_df.tomatometer_status == 'Certified Fresh')) & (genre_df.WorldwideGross <= genre_df.ProductionBudget)].shape[0],
    ]

    '''
    Tomatometer Status
    '''
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, unprofitable_movies, width, label='NP')
    rects2 = ax.bar(x + width/2, profitable_movies, width, label='P')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Movies')
    ax.set_title('Number of Movies per Tomatometer Status')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('new_plots/tomatometer_status_movie_counts/' + genre_column)

    def bool_to_int(x):
        return 1 if x == 'Fresh' or x == 'Certified Fresh' else 0

    genre_df['y'] = genre_df['tomatometer_status'].apply(bool_to_int)

    features = [
        'contains_director_features', 
        'genre_features', 
        'contains_cast_features', 
        'runtime_features', 
        'movie_rating_features',
    ]
    examples_list = []
    for feature in features:
        examples = genre_df[feature]
        if len(examples_list) == 0:
            for example in examples:
                examples_list.append(json.loads(example))
        else:
            for i in range(len(examples_list)):
                examples_list[i].update(json.loads(examples[i]))

    v = DictVectorizer()
    X = v.fit_transform(examples_list)
    y = genre_df['y'].to_numpy()
    print(X.shape)

    # sm = SMOTE()
    # X, y = sm.fit_resample(X, y)
    X = X.toarray()

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    except:
        print("THIS CATEGORY IS TOO SMALL", genre_column)
        continue
    classifiers = [
        BernoulliNB(),
        LogisticRegression(max_iter=200, random_state=1, n_jobs=-1),
        RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=1),
        MLPClassifier(hidden_layer_sizes=(100, 10), early_stopping=True)
    ]

    for clf in classifiers:
        print("--------------")
        print(clf.__repr__()[0:10])
        print(genre_column)
        print("--------------")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print("Accuracy: {}. F1: {}. ROC AUC: {}.".format(accuracy, f1, roc))
        print("TN: {}. FP: {}. FN: {}. TP: {}".format(tn, fp, fn, tp))