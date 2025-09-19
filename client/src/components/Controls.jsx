import { useState } from "react";

export default function Controls({onSolve, loading = false}){
    const[scenario, setScenario] = useState("Baseline");
    const[nPoints, setNpoints] = useState(401);

    return(
        <div>
            <label>
                Scenario:
                <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                    <option value="Baseline">Baseline</option>
                    <option value="Higher_delta">Higher_delta</option>
                    <option value="Higher_lambda">Higher_lambda</option>
                    <option value="Higher_S">Higher_S</option>
                    <option value="Higher_Sq">Higher_Sq</option>
                    <option value="Stronger_M">Stronger_M</option>
                </select>
            </label>
            <label>
                nPoints:
                <input type="number" value={nPoints} min={101} max={801} step={50} onChange={(e) => setNpoints(Number(e.target.value))}/>
            </label>
            <button onClick={() => onSolve({scenario, nPoints})} disabled={loading}>
                SOLVE
            </button>
        </div>
    );
}