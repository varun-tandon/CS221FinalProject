import pandas as pd
import matplotlib
import numpy as np
from fuzzywuzzy import fuzz

rotten_tom_df = pd.read_csv('datasets/rotten_tomatoes_movies.csv')
profit_data_df = pd.read_csv('data_parsed.csv')
movie_budget_df = pd.read_csv('movie_budgets.csv')

print(len(list(rotten_tom_df['movie_title'])))

movie_budget_titles = list(movie_budget_df['Movie'])
i = 0
def map_rotten_tom_to_budget(x):
    global i
    i += 1
    if i % 100 == 0: print(i / 16638)
    selected = sorted(movie_budget_titles, key=lambda y: fuzz.ratio(x['movie_title'][0:50], y[0:50]), reverse=True)[0]
    ratio = fuzz.ratio(x['movie_title'][0:50], selected[0:50])
    x['movie_title_budget'] = selected
    x['ratio'] = ratio
    return x

rotten_tom_df = rotten_tom_df.apply(map_rotten_tom_to_budget, axis=1)
rotten_tom_df.to_csv('rotten_tomatoes_movies_matched.csv')