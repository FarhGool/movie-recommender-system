# 🎬 Movie Recommendation System

## 📌 Overview

This project implements a Movie Recommendation System using Machine Learning techniques and Flask. It recommends movies based on content similarity, user preferences, and a hybrid approach combining both methods.

---

## 🚀 Features

* Content-Based Filtering (using genres, cast, keywords)
* Collaborative Filtering (using Surprise library)
* Hybrid Recommendation System
* Flask API for real-time recommendations
* Docker support for deployment

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Scikit-surprise
* Flask
* Docker (optional)

---

## 📂 Project Structure

```
analysis.ipynb
app.py
model.pkl
similarity.pkl
collab_model.pkl
ratings_df.pkl
requirements.txt
Dockerfile
templates/
    index.html
```

---

## ⚙️ How It Works

### 1. Content-Based Filtering

Uses movie metadata (genres, keywords, cast) and cosine similarity to recommend similar movies.

### 2. Collaborative Filtering

Uses user-item interactions and machine learning (KNN/SVD) to predict ratings.

### 3. Hybrid Approach

Combines both methods using a weighted score:
Final Score = α × Content Score + (1 - α) × Collaborative Score

---

## 🌐 API Endpoints

### Content-Based

```
/api/content?movie=Avatar
```

### Collaborative Filtering

```
/api/collab?user_id=1
```

### Hybrid Recommendation

```
/api/hybrid?user_id=1&movie=Avatar
```

---

## 🐳 Docker (Optional)

To build and run the project using Docker:

```
docker build -t movie-recommender .
docker run -p 5000:5000 movie-recommender
```

---

## ⚠️ Note

Model training was performed in Google Colab due to environment constraints. The trained models were saved and used in the Flask application.

---

## 📌 Author

Your Name
