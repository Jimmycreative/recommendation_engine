import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("toy_dataset.csv",index_col=0)
df = df.fillna(0)
print(df.head())

def normalization(row):
    new_row = (row-row.mean())/(row.max()-row.min())
    return new_row

ratings_nor = df.apply(normalization)

#item to item collaborative so we need to do the transpose
item_similarity = cosine_similarity(ratings_nor.T)
print(item_similarity)

item_similarity_df = pd.DataFrame(item_similarity,index=df.columns,columns=df.columns)
print(item_similarity_df)

#let's make recommendation
def get_similar_movies(movie_name, user_ratings):
    similar_score = item_similarity_df[movie_name]*(user_ratings-2.5)
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score

print(item_similarity_df["romantic3"])
print(get_similar_movies("romantic3",1))

action_lover = [("action1",5),("romantic2",1),("romantic3",1)]
similar_scores = pd.DataFrame()
for movie,rating in action_lover:
    similar_scores = similar_scores.append(get_similar_movies(movie,rating),ignore_index = True)

print(similar_scores)

similar_scores.sum().sort_values(ascending=False)