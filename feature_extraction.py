import pandas as pd
import json

df = pd.read_csv('data_parsed_contains_word_features.csv')

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

def count_num_words(x):
    has_num_words = {}
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        tokens = text.split()
        for i in range(0, len(tokens), 100):
            has_num_words['has_{}_num_words'.format(i)] = 1
    x['num_words_features'] = json.dumps(has_num_words)
    return x

def count_num_chars(x):
    has_num_chars = {}
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        for i in range(0, len(text), 1000):
            has_num_chars['has_{}_num_chars'.format(i)] = 1
    x['num_chars_features'] = json.dumps(has_num_chars)
    return x
df = df.apply(contains_word_extractor, axis=1)
df = df.apply(count_num_words, axis=1)
df = df.apply(count_num_chars, axis=1)
df = df.drop(columns=['Unnamed: 0'])
df.to_csv('data_parsed_contains_word_features.csv')