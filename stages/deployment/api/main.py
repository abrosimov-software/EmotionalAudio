from fastapi import FastAPI, HTTPException
import subprocess 

app = FastAPI()

@app.post("/updateModel/")
async def update_model(new_model: dict):
    try:
        subprocess.run(["systemctl", "restart", "my_fastapi_service"]) #not sure how it should work
        return {"message": "updateModel: success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
