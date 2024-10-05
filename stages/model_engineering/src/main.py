from fastapi import FastAPI, HTTPException
from typing import List, Optional

app = FastAPI()


@app.post("/updateDatasets/")
async def update_datasets(datasets: List[dict]):
    try:
        return {"message": "updateDatasets: success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/")
async def train():
    try:
        return {"message": "train: success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates/")
async def candidates():
    try:
        model_candidates = []
        return {"candidates": model_candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/")
async def validate(candidate_name: str):
    try:
        return {"message": f"validate {candidate_name}: success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getLeader/")
async def get_leader():
    try:
        model_name = ""
        return {"leader name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
