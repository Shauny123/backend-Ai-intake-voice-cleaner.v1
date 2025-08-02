from fastapi import FastAPI, UploadFile, File
from voice_cleaner_integration import clean_voice

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Voice Cleaner API is live"}

@app.post("/clean")
async def clean_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = clean_voice(audio_bytes)
    return {"result": result}

