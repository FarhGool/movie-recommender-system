from flask import Flask, request, render_template, jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import threading

from surprise import KNNBasic

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================
movies = pickle.load(open('model.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

model = pickle.load(open('collab_model.pkl', 'rb'))
ratings_df = pickle.load(open('ratings_df.pkl', 'rb'))

# =========================
# POSTER FUNCTION
# =========================
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=390e76286265f7638bb6b19d86474639&language=en-US"
    data = requests.get(url).json()

    poster_path = data.get('poster_path')
    if not poster_path:
        return None

    return "https://image.tmdb.org/t/p/w500/" + poster_path

# =========================
# CONTENT-BASED
# =========================
def recommend_content(movie_title, top_n=5):

    idx_list = movies[movies['title'] == movie_title].index
    if len(idx_list) == 0:
        return []

    idx = idx_list[0]
    distances = similarity[idx]

    results = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:top_n+1]

    output = []
    for i in results:
        output.append({
            "movie_id": int(movies.iloc[i[0]].id),
            "title": movies.iloc[i[0]].title,
            "score": float(i[1])
        })

    return output

# =========================
# COLLABORATIVE FILTERING
# =========================
def recommend_collab(user_id, top_n=5):

    movie_ids = ratings_df['movieId'].unique()

    preds = []
    for mid in movie_ids:
        pred = model.predict(user_id, mid)
        preds.append((mid, pred.est))

    preds.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in preds[:top_n]:
        match = movies[movies['id'] == mid]['title']
        if not match.empty:
            results.append({
                "title": match.values[0],
                "score": float(score)
            })

    return results

# =========================
# HYBRID MODEL
# =========================
def hybrid_recommend(user_id, movie_title, alpha=0.5, top_n=5):

    content_results = recommend_content(movie_title, top_n=20)

    hybrid = []

    for item in content_results:
        movie_id = item['movie_id']

        pred = model.predict(user_id, movie_id).est
        collab_norm = pred / 10  # normalize to 0–1

        final_score = alpha * item['score'] + (1 - alpha) * collab_norm

        hybrid.append({
            "title": item['title'],
            "score": final_score
        })

    hybrid = sorted(hybrid, key=lambda x: x['score'], reverse=True)

    return hybrid[:top_n]

# =========================
# UI ROUTES
# =========================
@app.route('/')
def home():
    movie_list = movies['title'].tolist()
    return render_template('index.html', movie_list=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['selected_movie']

    titles = [x['title'] for x in recommend_content(movie_title, 10)]

    return render_template(
        'index.html',
        movie_list=movies['title'].tolist(),
        recommended_movie_titles=titles
    )

# =========================
# API ROUTES
# =========================
@app.route('/api/content')
def api_content():
    movie = request.args.get('movie')
    return jsonify(recommend_content(movie))

@app.route('/api/collab')
def api_collab():
    user_id = int(request.args.get('user_id'))
    return jsonify(recommend_collab(user_id))

@app.route('/api/hybrid')
def api_hybrid():
    user_id = int(request.args.get('user_id'))
    movie = request.args.get('movie')
    return jsonify(hybrid_recommend(user_id, movie))

# =========================
# RUN FLASK (COLAB SAFE)
# =========================
if __name__ == '__main__':
    threading.Thread(
        target=lambda: app.run(port=5000, use_reloader=False)
    ).start()