# Activate venv - mac: source ./venv/bin/activate
# Activate fastAPI - uvicorn main:app --reload

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Backend for Accessibility App"}
