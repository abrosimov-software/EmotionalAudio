from fastapi import FastAPI, HTTPException, Body
from typing import Optional, List
import src # all the custom modules should be implemented in src folder

app = FastAPI()

@app.post("/extract_data/")
async def extract_data(source: Optional[str] = None):
    try:
        data = src.extract_data(source)
        return {"message": f"extract_data from: {source or 'default'}", "extracted_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_data/")
async def process_data(user_data: Optional[dict] = Body(None)):

    try:
        if user_data is None:
            extracted_data = []
            return {"message": "process_data from extract_data : success", "processed_data": extracted_data}
        else:
            return {"message": "process_data from user_data : success", "processed_data": user_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/serve/")
async def serve():
    try:
        final_dataset = []
        return {"message": "serve: success", "dataset": final_dataset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

