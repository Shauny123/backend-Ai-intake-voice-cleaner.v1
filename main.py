from fastapi import FastAPI
from voice_cleaner_integration import clean_voice  # Assuming this exists

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Voice Cleaner API is live"}

@app.post("/clean")
def clean_audio():
    # Replace this with actual logic from `voice_cleaner_integration.py`
    result = clean_voice()
    return {"result": result}
