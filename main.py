import os
import re
import numpy as np
import faiss
from datetime import datetime, timedelta
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ------------------ SETUP ------------------
load_dotenv()

app = FastAPI(title="Movie Streaming Platform API")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "movie_stream")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

movies_col = db.movies
users_col = db.users
watch_col = db.watch_history
reviews_col = db.reviews

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast embedding model


# ------------------ MODELS ------------------
class Movie(BaseModel):
    title: str
    release_year: int
    genres: list[str]
    cast: list[str]
    director: str
    rating: float = 0.0
    watch_count: int = 0


class User(BaseModel):
    name: str
    email: EmailStr
    subscription_type: str


class ReviewIn(BaseModel):
    user_id: str
    movie_id: str
    rating: float = Field(ge=0, le=10)
    text: str


class WatchEventIn(BaseModel):
    user_id: str
    movie_id: str
    timestamp: datetime | None = None
    watch_duration_seconds: int = 0


@app.get("/")
def show():
    return {"movie-streaming"}
# ------------------ SEED DATA ------------------
@app.post("/seed")
async def seed_data():
    await movies_col.delete_many({})
    await users_col.delete_many({})
    await watch_col.delete_many({})
    await reviews_col.delete_many({})

    movies = [
        {"title": "The Godfather", "release_year": 1972, "genres": ["Crime", "Drama"],
         "cast": ["Marlon Brando", "Al Pacino"], "director": "Francis Ford Coppola",
         "rating": 9.2, "watch_count": 0},
        {"title": "Inception", "release_year": 2010, "genres": ["Action", "Sci-Fi"],
         "cast": ["Leonardo DiCaprio"], "director": "Christopher Nolan",
         "rating": 8.8, "watch_count": 0},
        {"title": "Titanic", "release_year": 1997, "genres": ["Romance", "Drama"],
         "cast": ["Leonardo DiCaprio", "Kate Winslet"], "director": "James Cameron",
         "rating": 7.8, "watch_count": 0},
        {"title": "Interstellar", "release_year": 2014, "genres": ["Adventure", "Sci-Fi"],
         "cast": ["Matthew McConaughey"], "director": "Christopher Nolan",
         "rating": 8.6, "watch_count": 0},
        {"title": "The Dark Knight", "release_year": 2008, "genres": ["Action", "Crime"],
         "cast": ["Christian Bale", "Heath Ledger"], "director": "Christopher Nolan",
         "rating": 9.0, "watch_count": 0},
    ]

    # Generate and attach embeddings for each movie title
    for movie in movies:
        emb = embedding_model.encode(movie["title"]).tolist()
        movie["embedding"] = emb

    res_movies = await movies_col.insert_many(movies)

    users = [
        {"name": "John Doe", "email": "john@example.com", "subscription_type": "Premium"},
        {"name": "Alice Smith", "email": "alice@example.com", "subscription_type": "Free"},
    ]
    res_users = await users_col.insert_many(users)

    watch_events = []
    now = datetime.utcnow()
    for i, m in enumerate(res_movies.inserted_ids):
        for j, u in enumerate(res_users.inserted_ids):
            watch_events.append({
                "user_id": u,
                "movie_id": m,
                "timestamp": now - timedelta(days=i + j),
                "watch_duration_seconds": 3600
            })
    await watch_col.insert_many(watch_events)

    reviews = [
        {"user_id": res_users.inserted_ids[0], "movie_id": res_movies.inserted_ids[0],
         "rating": 9.0, "text": "Classic masterpiece"},
        {"user_id": res_users.inserted_ids[1], "movie_id": res_movies.inserted_ids[1],
         "rating": 8.5, "text": "Mind-bending!"},
    ]
    await reviews_col.insert_many(reviews)

    try:
        await movies_col.create_index([("title", "text"), ("director", "text"), ("cast", "text")], name="movies_text_idx")
    except Exception as e:
        print("Index creation error:", e)

    return {"seeded": True}


# ------------------ SEMANTIC SEARCH ------------------
@app.get("/movies/semantic_search")
async def semantic_search(query: str = Query(..., min_length=1), top_k: int = 5):
    movies = await movies_col.find({}, {"title": 1, "rating": 1, "watch_count": 1, "embedding": 1}).to_list(None)
    if not movies:
        raise HTTPException(404, "No movies found")

    vectors = np.array([m["embedding"] for m in movies]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    query_emb = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        movie = movies[idx]
        movie["_id"] = str(movie["_id"])
        movie["similarity_score"] = float(1 / (1 + distances[0][i]))
        results.append(movie)

    return {"query": query, "results": results}


# ------------------ HYBRID SEARCH ------------------
@app.get("/movies/hybrid_search")
async def hybrid_search(query: str = Query(..., min_length=1), top_k: int = 5):
    movies = await movies_col.find({}, {"title": 1, "rating": 1, "watch_count": 1, "embedding": 1}).to_list(None)
    if not movies:
        raise HTTPException(404, "No movies found")

    vectors = np.array([m["embedding"] for m in movies]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    query_emb = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        movie = movies[idx]
        similarity = float(1 / (1 + distances[0][i]))
        final_score = 0.5 * similarity + 0.3 * movie["rating"] + 0.2 * movie["watch_count"]
        movie["_id"] = str(movie["_id"])
        movie["hybrid_score"] = final_score
        results.append(movie)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {"query": query, "results": results}


# ------------------ KEYWORD + FUZZY SEARCH ------------------
@app.get("/movies/keyword_search")
async def keyword_search(query: str = Query(..., min_length=1)):
    regex_pattern = re.compile(f".*{re.escape(query)}.*", re.IGNORECASE)
    movies = await movies_col.find({"title": {"$regex": regex_pattern}}).to_list(None)

    if not movies:
        fuzzy_pattern = re.compile(f".*{re.escape(query[:-1])}.*", re.IGNORECASE)
        movies = await movies_col.find({"title": {"$regex": fuzzy_pattern}}).to_list(None)

    for m in movies:
        m["_id"] = str(m["_id"])

    return {"query": query, "results": movies}


@app.get("/users/{user_id}/history")
async def get_user_history(user_id: str):
    try:
        uid = ObjectId(user_id)
    except:
        raise HTTPException(400, "Invalid user ID")
    user = await users_col.find_one({"_id": uid})
    if not user:
        raise HTTPException(404, "User not found")
    cursor = watch_col.find({"user_id": uid}).sort("timestamp", -1)
    history = []
    async for w in cursor:
        movie = await movies_col.find_one({"_id": w["movie_id"]})
        if movie:
            history.append({
                "movie_title": movie["title"],
                "timestamp": w["timestamp"],
                "duration": w["watch_duration_seconds"]
            })
    return {"user": user["name"], "history": history}


@app.get("/movies/{movie_id}/reviews")
async def get_reviews(movie_id: str):
    try:
        mid = ObjectId(movie_id)
    except:
        raise HTTPException(400, "Invalid movie ID")
    movie = await movies_col.find_one({"_id": mid})
    if not movie:
        raise HTTPException(404, "Movie not found")
    cursor = reviews_col.find({"movie_id": mid})
    reviews = []
    async for r in cursor:
        user = await users_col.find_one({"_id": r["user_id"]})
        reviews.append({
            "user": user["name"] if user else "Unknown",
            "rating": r["rating"],
            "text": r["text"]
        })
    return {"movie": movie["title"], "reviews": reviews}


@app.post("/movies/{movie_id}/reviews")
async def add_review(movie_id: str, review: ReviewIn):
    try:
        mid = ObjectId(movie_id)
        uid = ObjectId(review.user_id)
    except:
        raise HTTPException(400, "Invalid ID format")
    movie = await movies_col.find_one({"_id": mid})
    if not movie:
        raise HTTPException(404, "Movie not found")
    user = await users_col.find_one({"_id": uid})
    if not user:
        raise HTTPException(404, "User not found")
    await reviews_col.insert_one({
        "user_id": uid,
        "movie_id": mid,
        "rating": review.rating,
        "text": review.text
    })
    return {"ok": True}


@app.post("/watch")
async def add_watch(event: WatchEventIn):
    try:
        uid = ObjectId(event.user_id)
        mid = ObjectId(event.movie_id)
    except:
        raise HTTPException(400, "Invalid ID format")
    await watch_col.insert_one({
        "user_id": uid,
        "movie_id": mid,
        "timestamp": event.timestamp or datetime.utcnow(),
        "watch_duration_seconds": event.watch_duration_seconds
    })
    await movies_col.update_one({"_id": mid}, {"$inc": {"watch_count": 1}})
    return {"ok": True}


@app.get("/analytics/top5_last_month")
async def top5_last_month():
    last_month = datetime.utcnow() - timedelta(days=30)
    pipeline = [
        {"$match": {"timestamp": {"$gte": last_month}}},
        {"$group": {"_id": "$movie_id", "watch_count": {"$sum": 1}}},
        {"$sort": {"watch_count": -1}},
        {"$limit": 5},
        {"$lookup": {"from": "movies", "localField": "_id", "foreignField": "_id", "as": "movie"}},
        {"$unwind": "$movie"},
        {"$project": {"_id": 0, "title": "$movie.title", "watch_count": 1}}
    ]
    cursor = watch_col.aggregate(pipeline)
    result = [doc async for doc in cursor]
    return result
