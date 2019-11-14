import pandas as pd
import json

df = pd.read_csv('data_parsed.csv')

def contains_word_extractor(x):
    contains_word_features = {}
    try:
        with open('movie_scripts/' + x['ScriptLink']) as f:
            text = f.read()
            words = text.split()
            for word in words:
                contains_word_features['contains_word_' + word] = True
        x['contains_word_features'] = json.dumps(contains_word_features)
    except FileNotFoundError:
        print("German, I cannot find this file: ", x['fuzzy_picked_file'])
    return x

df = df.apply(contains_word_extractor, axis=1)
df = df.drop(columns=['Unnamed: 0'])
df.to_csv('data_parsed_contains_word_features.csv')