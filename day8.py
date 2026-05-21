from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import cv2
import numpy as np
import os

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to AI Emotion API! Send a POST request to /predict"}


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
       
        results = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']
        
       
        os.remove(temp_path)
        
        return {
            "status": "success",
            "detected_emotion": dominant_emotion,
            "all_emotions": results[0]['emotion']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
