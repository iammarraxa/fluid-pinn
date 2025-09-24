import { useState, useEffect, useMemo } from "react";

export default function Controls({onSolve, loading = false}){
    const[scenario, setScenario] = useState("Baseline");
    const[nPoints, setNPoints] = useState(401);

    const R = {
        Sq:   { min: 0.2,  max: 2.0,  step: 0.1,  def: 0.6,  label: "Sq (α)" },
        S:    { min: 0.0,  max: 0.32, step: 0.02, def: 0.08, label: "S (V₀)" },
        lam:  { min: 0.1,  max: 0.8,  step: 0.05, def: 0.4,  label: "λ" },
        M:    { min: 0.2,  max: 3.0,  step: 0.1,  def: 0.8,  label: "M (B₀)" },
        delta:{ min: 1.2,  max: 5.0,  step: 0.1,  def: 2.5,  label: "δ" },
    };

    const [Sq, setSq] = useState(R.Sq.def);
    const [S, setS] = useState(R.S.def);
    const [lam, setLam] = useState(R.lam.def);
    const [M, setM] = useState(R.M.def);
    const [delta, setDelta] = useState(R.delta.def);

    const clamp = (v, { min, max }) => Math.min(max, Math.max(min, v));
    const snap  = (v, { min, step }) => Number((Math.round((v - min) / step) * step + min).toFixed(5));
    const norm  = (v, range) => snap(clamp(v, range), range);

    const invalids = useMemo(() => ({
        nPoints: !(Number.isInteger(nPoints) && nPoints >= 101 && nPoints <= 2001),
        Sq:      !(Sq   >= R.Sq.min    && Sq   <= R.Sq.max),
        S:       !(S    >= R.S.min     && S    <= R.S.max),
        lam:     !(lam  >= R.lam.min   && lam  <= R.lam.max),
        M:       !(M    >= R.M.min     && M    <= R.M.max),
        delta:   !(delta>= R.delta.min && delta<= R.delta.max),
    }), [nPoints, Sq, S, lam, M, delta]);

    const anyInvalid = Object.values(invalids).some(Boolean);

    const handleScenarioChange = (e) => {
        const newScenario = e.target.value;
        setScenario(newScenario);
        if (!loading && anyInvalid) {
            onSolve({ scenario: newScenario, nPoints, Sq, S, lam, M, delta });
        }
    };

    useEffect(() => {
        if(loading || anyInvalid) return;

        const timer = setTimeout(() => {
            onSolve({ scenario, nPoints, Sq, S, lam, M, delta });
        }, 200);

        return () => clearTimeout(timer);
    },[nPoints]);

    const Slider = ({ id, value, setValue, range, label, fmt = (x) => x.toFixed(2) }) => (
        <div style={{ marginBottom: "0.5rem" }}>
            <label htmlFor={id} style={{ color: invalids[id] ? "#b30000" : undefined }}>
                {label}: {fmt(value)}
            </label>
            <input
                id={id}
                type="range"
                min={range.min}
                max={range.max}
                step={range.step}
                value={value}
                onChange={(e) => setValue(norm(Number(e.target.value), range))}
                disabled={loading}
                aria-valuemin={range.min}
                aria-valuemax={range.max}
                aria-valuenow={value}
            />
            {invalids[id] && (
                <div style={{ fontSize: 12, color: "#b30000" }}>
                    Must be between {range.min} and {range.max}
                </div>
            )}
        </div>
    );

    return(
        <div style={{ marginBottom: "1rem" }}>
            <div style={{ marginBottom: "0.5rem" }}>
                <label htmlFor="scenario-select">Scenario:&nbsp;</label>
                <select id="scenario-select" value={scenario} onChange={handleScenarioChange} disabled={loading}>
                    <option value="Baseline">Baseline</option>
                    <option value="Higher_delta">Higher_delta</option>
                    <option value="Higher_lambda">Higher_lambda</option>
                    <option value="Higher_S">Higher_S</option>
                    <option value="Higher_Sq">Higher_Sq</option>
                    <option value="Stronger_M">Stronger_M</option>
                    <option value="auto">auto</option>
                </select>
            </div>

            <div style={{ marginBottom: "0.5rem" }}>
                <label htmlFor="points-slider">Points: {nPoints}</label>
                <input id="points-slider" type="range" min={101} max={801} step={50} value={nPoints} aria-valuemin={101} aria-valuemax={801} aria-valuenow={nPoints} 
                onChange={(e) => setNPoints(Number(e.target.value))} disabled={loading}/>
                {invalids.nPoints && (
                    <div style={{fontSize: 12, color: "#b30000"}}>nPoints must be between 101 and 2001</div>
                )}
            </div>

            <Slider id='Sq' value={Sq} setValue={setSq} range={R.Sq} label={R.Sq.label} />
            <Slider id='S' value={S} setValue={setS} range={R.S} label={R.S.label} />
            <Slider id='lam' value={lam} setValue={setLam} range={R.lam} label={R.lam.label} />
            <Slider id='M' value={M} setValue={setM} range={R.M} label={R.M.label} />
            <Slider id='delta' value={delta} setValue={setDelta} range={R.delta} label={R.delta.label} />
            
            
            <button onClick={() => {if (!loading) onSolve({scenario, nPoints, Sq, S, lam, M, delta});}} disabled={loading}>
                {loading ? "Computing…" : "Solve"}
            </button>
            {anyInvalid && (
                <div style={{ fontSize: 12, color: "#b30000", marginTop: 6 }}>
                    Fix invalid inputs before solving
                </div>
            )}
        </div>
    );
}