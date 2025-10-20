import os
from datetime import datetime, timedelta
from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

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


@app.post("/seed")
async def seed_data():
    await movies_col.delete_many({})
    await users_col.delete_many({})
    await watch_col.delete_many({})
    await reviews_col.delete_many({})

    movies = [
        {"title": "The Godfather", "release_year": 1972, "genres": ["Crime", "Drama"], "cast": ["Marlon Brando", "Al Pacino"], "director": "Francis Ford Coppola", "rating": 9.2, "watch_count": 0},
        {"title": "Inception", "release_year": 2010, "genres": ["Action", "Sci-Fi"], "cast": ["Leonardo DiCaprio"], "director": "Christopher Nolan", "rating": 8.8, "watch_count": 0},
        {"title": "Titanic", "release_year": 1997, "genres": ["Romance", "Drama"], "cast": ["Leonardo DiCaprio", "Kate Winslet"], "director": "James Cameron", "rating": 7.8, "watch_count": 0},
        {"title": "Interstellar", "release_year": 2014, "genres": ["Adventure", "Sci-Fi"], "cast": ["Matthew McConaughey"], "director": "Christopher Nolan", "rating": 8.6, "watch_count": 0},
        {"title": "The Dark Knight", "release_year": 2008, "genres": ["Action", "Crime"], "cast": ["Christian Bale", "Heath Ledger"], "director": "Christopher Nolan", "rating": 9.0, "watch_count": 0},
    ]
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
        {"user_id": res_users.inserted_ids[0], "movie_id": res_movies.inserted_ids[0], "rating": 9.0, "text": "Classic masterpiece"},
        {"user_id": res_users.inserted_ids[1], "movie_id": res_movies.inserted_ids[1], "rating": 8.5, "text": "Mind-bending!"},
    ]
    await reviews_col.insert_many(reviews)

    try:
        await movies_col.create_index([("title", "text"), ("director", "text"), ("cast", "text")], name="movies_text_idx")
    except Exception as e:
        print("Index creation error:", e)

    return {"seeded": True}


@app.get("/movies/search")
async def search_movies(query: str = Query(..., min_length=1)):
    cursor = movies_col.aggregate([
        {"$match": {"$text": {"$search": query}}},
        {"$addFields": {"score": {"$meta": "textScore"}}},
        {"$sort": {"score": -1}},
        {"$limit": 20}
    ])
    results = []
    async for doc in cursor:
        score = 0.5 * doc["score"] + 0.3 * doc["rating"] + 0.2 * doc["watch_count"]
        doc["_id"] = str(doc["_id"])
        doc["final_score"] = score
        results.append(doc)
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


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
