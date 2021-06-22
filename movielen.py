import pandas as pd

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

ratings = pd.merge(movies,ratings).drop(["genres","timestamp"],axis=1)


user_ratings = ratings.pivot_table(index =["userId"],columns=["title"],values="rating")


#Let's remove movies which have less than 10 users that have rated, and filled remaining NaN with 0

user_ratings = user_ratings.dropna(thresh=10,axis=1).fillna(0)


#let's build a similarity matrix
#pearson method will adjust the mean so don't need standirize
item_similarity_df = user_ratings.corr(method="pearson")
print(item_similarity_df.head())
def get_similar_movies(movie_name, rating):
    similar_score = item_similarity_df[movie_name]*(rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score

romantic_lover =  [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index = True)

#print(similar_movies.head(10))

print(similar_movies.sum().sort_values(ascending=False))