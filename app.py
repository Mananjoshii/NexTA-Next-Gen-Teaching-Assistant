from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ APP INIT ------------------
app = FastAPI(title="Sigma Course AI Backend")

# ------------------ LOAD DATA ------------------
df = joblib.load("embeddings.joblib")

# ------------------ REQUEST / RESPONSE SCHEMAS ------------------
class QuestionRequest(BaseModel):
    question: str

class VideoChunk(BaseModel):
    title: str
    video_number: int
    start_time_seconds: int
    end_time_seconds: int
    text_snippet: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    related_videos: list[VideoChunk]
    meta: dict

# ------------------ OLLAMA HELPERS ------------------
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    return r.json()["embeddings"]

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]

# ------------------ API ENDPOINT ------------------
@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest):
    incoming_query = payload.question

    # 1. Create embedding for user query
    question_embedding = create_embedding([incoming_query])[0]

    # 2. Cosine similarity
    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

    top_k = 5
    top_indices = similarities.argsort()[::-1][:top_k]
    new_df = df.loc[top_indices]

    # 3. Build prompt
    context_json = new_df[
        ["title", "number", "start", "end", "text"]
    ].to_json(orient="records")

    prompt = f"""
I am teaching web development in my Sigma web development course.
Here are video subtitle chunks containing video title, video number,
start time in seconds, end time in seconds, and the text:

{context_json}
---------------------------------
"{incoming_query}"

User asked this question related to the video chunks.
Answer in a human way and clearly mention:
- which video
- what topic
- exact timestamps

If the question is unrelated to the course, say you can only answer
questions related to the course.
"""

    # 4. LLM response
    answer = inference(prompt)

    # 5. Prepare related videos
    related_videos = []
    for _, row in new_df.iterrows():
        related_videos.append({
            "title": row["title"],
            "video_number": int(row["number"]),
            "start_time_seconds": int(row["start"]),
            "end_time_seconds": int(row["end"]),
            "text_snippet": row["text"][:200]
        })

    # 6. Final JSON response
    return {
        "question": incoming_query,
        "answer": answer,
        "related_videos": related_videos,
        "meta": {
            "top_k": top_k,
            "model": "llama3.2",
            "embedding_model": "bge-m3"
        }
    }