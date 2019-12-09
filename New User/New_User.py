#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# read the files
movies = pd.read_csv('ml-20m/movies.csv')
ratings = pd.read_csv('ml-20m/ratings.csv')
genome_scores = pd.read_csv('Project91/ml-20m/genome-scores.csv')

######################################
######### Filter and Sort ############
######################################

# merge rating into the movie table 
mean = pd.DataFrame(ratings[['movieId','rating']].groupby(['movieId'])['rating'].mean())
mean['movieId'] = mean.index
mean.index = range(len(mean))
merged_table = pd.merge(movies, mean, on = 'movieId')

# merge rating count into the table
count = pd.DataFrame(ratings[['movieId','userId']].groupby(['movieId'])['userId'].count())
count['movieId'] = count.index
count.index = range(len(count))
count.columns = ['count','movieId']
count = count.sort_values('count', ascending = False)
merged = pd.merge(merged_table, count, on = 'movieId')

result = merged.sort_values('count', ascending = False)

def choose(genre):
    
    inpt = '|'.join(genre)
    final = result[result.genres.str.contains(inpt) & (result.rating > 4)]
    
    if len(final) < 10:
        return final.title.tolist()
    
    return final.title[:10].tolist()

###################################################
######### Content-based Recommendation ############
###################################################

# transform the genome_scores dataframe into relevance matrix
df = genome_scores.groupby(['movieId'])['relevance'].apply(list).apply(pd.Series)
score_matrix = np.matrix(df)

# compute cosine similarity for every movie 
cosine_sim = cosine_similarity(score_matrix, score_matrix)

# get the movieId value and reset the df index
df['movieId']=df.index
df.index = range(len(df))

def recommendation(title):
    
    # link title to movieId and find the row number of the matrix
    ids = movies[movies['title'] == title]['movieId'].values[0]
    i = df[df.movieId==ids].index.values[0]
    
    # find the specific score for the movie and sort the scores
    sim_scores = list(enumerate(cosine_sim[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # store the row numbers and related movieId
    indices = [i[0] for i in sim_scores[1:]]
    movie_ids = df.iloc[indices,:].movieId.values
    
    result = []
    for i in movie_ids:
        result.append(movies[movies.movieId == i].title.values[0])
    
    return result

