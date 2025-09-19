from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from inference import compute_profiles

class SolveRequest(BaseModel):
    scenario: str = "Baseline"
    n_points: int = 401

origins = ['http://localhost:5173',
           'http://127.0.0.1:5173']  #change it to actual site URL when in production

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_methods = ["GET", "POST", "OPTIONS"],
    allow_headers = ["*"],  #['Content-Type'] in production
    allow_credentials = False
)

@app.post("/solve")
def solve(req: SolveRequest):
    try:
        result = compute_profiles(req.scenario, req.n_points)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing file: {e.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.get("/health")
def health():
    return {'ok':True}