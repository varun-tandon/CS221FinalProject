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

distinct_df = pd.read_csv('rotten_tom_movies_distinct_matched.csv')

def contains_director(x):
    contains_director_features = {}
    if type(x['directors']) != str:
        x['contains_director_features'] = json.dumps({})
        return x
    directors = x['directors'].split(',')
    for director in directors:
        contains_director_features['contains_director_' + director.strip()] = 1
    x['contains_director_features'] = json.dumps(contains_director_features)
    return x

def contains_cast(x):
    contains_cast_features = {}
    if type(x['cast']) != str:
        x['contains_cast_features'] = json.dumps({})
        return x
    cast = x['cast'].split(',')
    for cast_mem in cast:
        contains_cast_features['contains_cast_' + cast_mem.strip()] = 1
    x['contains_cast_features'] = json.dumps(contains_cast_features)
    return x

def genre_features(x):
    genre_features = {}
    genres = x['genre'].split(',')
    for genre in genres:
        genre_features['is_genre_' + genre] = 1
    x['genre_features'] = json.dumps(genre_features)
    return x

def runtime_features(x):
    runtime_features = {}
    runtime_features['runtime'] = x['runtime_in_minutes']
    x['runtime_features'] = json.dumps(runtime_features)
    return x

def profitable_features(x):
    profitable_features = {}
    x['profitable_features'] = json.dumps({'is_profitable': x['IsProfitable']})
    return x

def movie_title_features(x):
    movie_title_features = {}
    movie_title_words = x['movie_title'].split()
    for movie_title_word in movie_title_words:
        movie_title_features['has_word_in_title_' + movie_title_word.lower()] = 1
    x['movie_title_features'] = json.dumps(movie_title_features)
    return x

def movie_desc_features(x):
    movie_desc_features = {}
    movie_desc_words = x['movie_info'].split()
    for movie_desc_word in movie_desc_words:
        movie_desc_features['has_word_in_desc_' + movie_desc_word.lower()] = 1
    x['movie_desc_features'] = json.dumps(movie_desc_features)
    return x

def movie_rating(x):
    movie_rating_features = {}
    movie_rating_features['movie_rating_is_' + x['rating']] = 1
    x['movie_rating_features'] = json.dumps(movie_rating_features)
    return x

def isProfitable(x):
    x['IsProfitable'] = 1 if x['WorldwideGross'] > x['ProductionBudget'] else 0
    return x

distinct_df['runtime_in_minutes'] = distinct_df['runtime_in_minutes'].fillna(0)
distinct_df['movie_info'] = distinct_df['movie_info'].fillna('')
distinct_df = distinct_df.apply(isProfitable, axis=1)
distinct_df = distinct_df.apply(contains_director, axis=1)
distinct_df = distinct_df.apply(contains_cast, axis=1)
distinct_df = distinct_df.apply(genre_features, axis=1)
distinct_df = distinct_df.apply(runtime_features, axis=1)
distinct_df = distinct_df.apply(profitable_features, axis=1)
distinct_df = distinct_df.apply(movie_title_features, axis=1)
distinct_df = distinct_df.apply(movie_desc_features, axis=1)
distinct_df = distinct_df.apply(movie_rating, axis=1)
# df = df.apply(count_num_words, axis=1)
# df = df.apply(count_num_chars, axis=1)
# distinct_df = distinct_df[df.contains_director_features != 'Tina']
distinct_df = distinct_df.drop(columns=['Unnamed: 0'])
distinct_df.to_csv('data_parsed_contains_word_features.csv')
