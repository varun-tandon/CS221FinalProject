import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('data_parsed.csv')

def contains_word_extractor(x):
    contains_word_features = {}
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        words = text.split()
        for word in words:
            contains_word_features['contains_word_' + word] = True
    x['contains_word_features'] = json.dumps(contains_word_features)
    return x

def contains_blank_occ_of_word(x):
    contains_occ_of_word_features = {}
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        words = text.split()
        for word in words:
            key = word + '_num_occ'
            if key in contains_occ_of_word_features:
                contains_occ_of_word_features[key] += 1
            else:
                contains_occ_of_word_features[key] = 1
    x['contains_occ_of_word_features'] = json.dumps(contains_occ_of_word_features)
    return x

def script_author(x):
    script_author_features = {}
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        by_index = text.lower().find('by')
        text = text[by_index + 2:by_index + 100]
        text = re.sub(r'<[^>]*>', '', text)
        text_tok = text.split()
        text_pos = nltk.pos_tag(text_tok)
        script_author_features['author_is_' + ' '.join([nnp[0] for nnp in text_pos if nnp[1] == 'NNP'][0:2])] = 1
    x['script_author_features'] = json.dumps(script_author_features)
    return x

def count_num_words(x):
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        tokens = text.split()
        x['num_words'] = len(tokens)
    return x

def count_num_chars(x):
    with open('movie_scripts/' + x['ScriptLink']) as f:
        text = f.read()
        x['num_chars'] = len(text.split())
    return x


df = df.apply(contains_blank_occ_of_word, axis=1)
df = df.apply(script_author, axis=1)
# df = df.apply(count_num_words, axis=1)
# df = df.apply(count_num_chars, axis=1)
df = df.drop(columns=['Unnamed: 0'])
df.to_csv('data_parsed_contains_word_features.csv')