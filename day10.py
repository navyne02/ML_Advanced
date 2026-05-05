from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import os
import sqlite3
from datetime import datetime

app = FastAPI()

# 1. Database Setup (Simple SQLite)
def init_db():
    conn = sqlite3.connect("ai_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            emotion TEXT,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.post("/predict_and_save")
async def predict_and_save(file: UploadFile = File(...)):
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. AI Prediction
        results = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']
        score = results[0]['emotion'][dominant_emotion]

        # 3. Save to Database
        conn = sqlite3.connect("ai_results.db")
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO predictions (timestamp, emotion, score) VALUES (?, ?, ?)", 
                       (now, dominant_emotion, score))
        conn.commit()
        conn.close()

        os.remove(temp_path)
        return {"status": "saved", "detected": dominant_emotion, "time": now}

    except Exception as e:
        return {"error": str(e)}

# To run: uvicorn day10:app --reload