from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import cv2
import numpy as np
import os

# 1. Initialize FastAPI App
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to AI Emotion API! Send a POST request to /predict"}

# 2. Prediction Endpoint
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 3. Use DeepFace to analyze emotion (Same as Day 3 logic)
        results = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "status": "success",
            "detected_emotion": dominant_emotion,
            "all_emotions": results[0]['emotion']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run this: uvicorn day8:app --reload