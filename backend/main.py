# Activate venv - mac: source ./venv/bin/activate
# Activate venv - windows: .\venv\Scripts\activate
# Activate fastAPI - uvicorn main:app --reload

from fastapi import FastAPI, UploadFile, File
from models import MatchedPhoto
from face_recognition import process_photo
from config import db
from datetime import datetime

app = FastAPI()

@app.post("/find_friend_in_photo")
async def find_friend_in_photo(file: UploadFile = File(...), user_id: str = ""):
    image = Image.open(io.BytesIO(await file.read()))
    is_match = await process_photo(image)
    if is_match:
        matched_photo = MatchedPhoto(user_id=user_id, photo_name=file.filename, timestamp=datetime.utcnow())
        await db.matched_photos.insert_one(matched_photo.dict())
        return {"status": "match", "message": "Friend found in photo!", "photo_name": file.filename}
    return {"status": "no_match", "message": "No match found."}

@app.get("/")
async def root():
    return {"message": "Backend for Photo App"}
