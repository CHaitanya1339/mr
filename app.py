from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from imdb import IMDb
import pandas as pd
from tabulate import tabulate
ia = IMDb()
app = Flask(__name__)
df = pd.read_csv('IMDB_10000.csv')
dataset = df[['title','year','certificate','genre','rating']]
dataset.fillna('', inplace=True)
dataset['genre'] = dataset['genre'].apply(lambda a: str(a).replace(',', ' '))
vec = CountVectorizer()
vec_matrix = vec.fit_transform(dataset['genre'])

tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(vec_matrix)

lsa = TruncatedSVD(n_components=20, algorithm='arpack')
lsa.fit(tfidf)

similarity = cosine_similarity(tfidf)
similarity_df = pd.DataFrame(similarity, index=dataset['title'], columns=dataset['title'])
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    user_movie = request.form['movie']
    movie_index = dataset[dataset['title'] == user_movie].index[0]

    similarity_scores = cosine_similarity(tfidf[movie_index], tfidf)

    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]


    movie_data = []
    for i,score in sorted_similar_movies:
        title = dataset.loc[i, 'title']
        year = dataset.loc[i, 'year']
        certificate=dataset.loc[i,'certificate']
        rating = dataset.loc[i, 'rating']
        movie_data.append((title, year, certificate, rating))

    headers = ['Title', 'Year', 'Certificate', 'Rating']
    movie_table = tabulate(movie_data, headers=headers, tablefmt='html')

    return render_template('recommend.html', movie_table=movie_table, user_movie=user_movie)
if __name__ == '__main__':
    app.run(debug=True)

