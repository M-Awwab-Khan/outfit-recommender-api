# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from interface import find_best_match

app = FastAPI()


class OutfitRequest(BaseModel):
    shirts: List[str]
    pants: List[str]
    min_similarity: float = 0.7  # Optional threshold


@app.get("/")
async def root():
    return {"message": "Welcome to the Outfit Recommendation API!"}


@app.post("/recommend")
async def recommend_outfit(req: OutfitRequest):
    try:
        result = find_best_match(req.shirts, req.pants, req.min_similarity)
        return {"recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
