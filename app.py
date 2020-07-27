from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

C = df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.75)
q_movies = df2.copy().loc[df2['vote_count'] >= m]
def WR(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(WR, axis=1)
q_movies = q_movies.sort_values('score', ascending = False)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

tfidf = TfidfVectorizer(stop_words='english')

df2['overview'] = df2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df2['overview'])

def get_recc(title, cosine_sim):
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    res = []
    for i in sim_scores:
        res.append(df2['title'][i[0]])

    return res

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name']for i in x]
        
        if len(names) > 3:
            names = names[:3]
        return names
    
    return []

df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df2.index, index=df2['title'])

##Run this only once then comment it
##html = q_movies[['title', 'score']].to_html()
##text_file = open("index.html", "w")
##text_file.write(html)
##text_file.close()

app = Flask(__name__)

@app.route("/send", methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        movie = request.form['movie']
        try:
            return render_template('movie.html',movies=get_recc(movie, cosine_sim))
        except KeyError:
            return render_template('err.html')

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
