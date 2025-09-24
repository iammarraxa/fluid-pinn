from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from inference import compute_profiles

class SolveRequest(BaseModel):
    scenario: str = "auto"
    n_points: int = 401
    Sq: float | None = None
    S: float | None = None
    lam: float | None = None
    M: float | None = None
    delta: float | None = None

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

def select_model(req: SolveRequest):
    baseline = {
        "Sq": 0.6,
        "S": 0.08,
        "lam": 0.4,
        "M": 0.8,
        "delta": 2.5,
    }

    highs = {
        "Sq": 1.2,
        "S": 0.16,
        "lam": 0.8,
        "M": 1.6,
        "delta": 5.0,
    }

    if req.scenario != "auto":
        return req.scenario, "override"
    
    deviations = {}
    if req.Sq is not None:
        deviations["Sq"] = max(0.0, min(1.0, (req.Sq - baseline["Sq"]) / (highs["Sq"] - baseline["Sq"])))
    if req.S is not None:
        deviations["S"] = max(0.0, min(1.0, (req.S - baseline["S"]) / (highs["S"] - baseline["S"])))
    if req.lam is not None:
        deviations["lam"] = max(0.0, min(1.0, (req.lam - baseline["lam"]) / (highs["lam"] - baseline["lam"])))
    if req.M is not None:
        deviations["M"] = max(0.0, min(1.0, (req.M - baseline["M"]) / (highs["M"] - baseline["M"])))
    if req.delta is not None:
        deviations["delta"] = max(0.0, min(1.0, (req.delta - baseline["delta"]) / (highs["delta"] - baseline["delta"])))

    if not deviations:
        return "Baseline", "none"
    
    driver_param, driver_val = max(deviations.items(), key=lambda kv: kv[1])

    if driver_param == "Sq":
        selected = "Higher_Sq" if req.Sq >= 0.9 else "Baseline"
    elif driver_param == "S":
        selected = "Higher_S" if req.S >= 0.12 else "Baseline"
    elif driver_param == "lam":
        selected = "Higher_lambda" if req.lam >= 0.6 else "Baseline"
    elif driver_param == "M":
        selected = "Stronger_M" if req.M >= 1.2 else "Baseline"
    elif driver_param == "delta":
        selected = "Higher_delta" if req.delta >= 3.75 else "Baseline"
    else:
        selected = "Baseline"

    return selected, driver_param



@app.post("/solve")
def solve(req: SolveRequest):
    try:
        baseline_defaults = {
            "Sq": 0.6,
            "S": 0.08,
            "lam": 0.4,
            "M": 0.8,
            "delta": 2.5,
        }

        if not (101 <= req.n_points <= 2001):
            raise HTTPException(
                status_code=400,
                detail="n_points must be between 101 and 2001"
            )
        if req.Sq is not None and not (0.2 <= req.Sq <= 2.0):
            raise HTTPException(
                status_code=400,
                detail="Sq must be between 0.2 and 2.0"
            )
        if req.S is not None and not (0.00 <= req.S <= 0.32):
            raise HTTPException(
                status_code=400,
                detail="S must be between 0.00 and 0.32"
            )
        if req.lam is not None and not (0.1 <= req.lam <= 0.8):
            raise HTTPException(
                status_code=400,
                detail="lam must be between 0.1 and 0.8"
            )
        if req.M is not None and not (0.2 <= req.M <= 3.0):
            raise HTTPException(
                status_code=400,
                detail="M must be between 0.2 and 3.0"
            )
        if req.delta is not None and not (1.2 <= req.delta <= 5.0):
            raise HTTPException(
                status_code=400,
                detail="delta must be between 1.2 and 5.0"
            )
        
        selected_model, driver_param = select_model(req)

        print("[/solve]", 
              "scenario=", req.scenario,
              "n_points=", req.n_points, 
              "selected=", selected_model,
              "driver=", driver_param)
        result = compute_profiles(selected_model, req.n_points)

        result["selected_model"] = selected_model
        result["driver_param"] = driver_param
        result["params"] = {
            "Sq": req.Sq if req.Sq is not None else baseline_defaults["Sq"],
            "S": req.S if req.S is not None else baseline_defaults["S"],
            "lam": req.lam if req.lam is not None else baseline_defaults["lam"],
            "M": req.M if req.M is not None else baseline_defaults["M"],
            "delta": req.delta if req.delta is not None else baseline_defaults["delta"],
        }
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing file: {e.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.get("/health")
async def health():
    return {'ok':True}