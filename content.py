import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")


features = ["keywords","genres","cast","director"]

for feature in features:
    df[feature] = df[feature].fillna("")



def combine_features(row):
    return row["keywords"]+" "+row["genres"]+ " "+row["cast"]+" "+row["director"]

df["combined_features"] = df.apply(combine_features,axis=1)

print(df.head())

cv = CountVectorizer()  #important
count_matrix = cv.fit_transform(df["combined_features"])
x = count_matrix.toarray()
#print(x)

cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)

def get_index(movie_name):
    return df[df.title==movie_name]["index"].values[0]
def get_title(index):
    return df[df.index==index]["title"].values[0]
#print(get_index("Man of Steel"))
#print(get_title(9))

movie_name = "Avatar"

def similar_movies(movie):
    list_movie = list(enumerate(cosine_sim[get_index(movie)]))
    sorted_movies = sorted(list_movie,key=lambda x:x[1],reverse=True)[1:]

    count = 0
    print("Top5 Movies")
    for movie in sorted_movies:
        if count==5:
            break
        print(get_title(movie[0]))
        count+=1

similar_movies(movie_name)