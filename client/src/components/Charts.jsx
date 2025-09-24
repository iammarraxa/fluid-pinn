import Plot from "react-plotly.js"

export default function Charts({data}){
    if(!data) return <p>Run a solve func to see plots</p>;

    const eta   = data.eta;
    const theta = data.theta;
    const fp    = data.fp;

    const cf = data?.wall?.cf;
    const Nu = data?.wall?.Nu;

    console.log("lengths:", eta?.length, fp?.length, theta?.length);

    if(
        !Array.isArray(eta) || !Array.isArray(fp) || !Array.isArray(theta) || eta.length !== fp.length || eta.length !== theta.length
    ) {return <p style={{ color: "red" }}>Invalid data from server</p>;}

    const mask = eta.map((_, i) => Number.isFinite(fp[i]) && Number.isFinite(theta[i]));

    const x = eta.filter((_, i) => mask[i]);
    const y1 = fp.filter((_, i) => mask[i]);
    const y2 = theta.filter((_, i) => mask[i]);
            
    return(
        <div>
            <h1>Scenario: {data.scenario}</h1>
            <div style={{ border: "1px solid #ccc", margin: "8px 0", padding: "4px" }}>
                <Plot data={[
                    {
                        x: data.eta,
                        y: data.fp,
                        mode: "lines",
                        name: "f′(η)"
                    }
                ]}
                layout={{
                    title: "Velocity profile f′(η)",
                    xaxis: { title: "η"},
                    yaxis: { title: "f′(η)"},
                    margin: { t: 40, r: 10, b: 45, l: 50 },
                    height: 320
                }}
                useResizeHandler = {true}
                style={{ width: "100%"}}
                />
            </div>
            <div style={{ border: "1px solid #ccc", margin: "8px 0", padding: "4px" }}>
                <Plot data={[
                    {
                        x: data.eta,
                        y: data.theta,
                        mode: "lines",
                        name: "θ(η)"
                    }
                ]}
                layout={{
                    title: "Temperature profile θ(η)",
                    xaxis: { title: "η"},
                    yaxis: { title: "θ(η)"},
                    margin: { t: 40, r: 10, b: 45, l: 50 },
                    height: 320
                }}
                useResizeHandler = {true}
                style={{ width: "100%"}}
                />
            </div>
            <div style={{ marginTop: "1rem", padding: "0.5rem", border: "1px solid #ddd", borderRadius: "6px" }}>
                <p>
                    <strong>cf:</strong>{" "}
                    {Number.isFinite(cf) ? cf.toPrecision(4) : "–"}{" "}
                    <strong>Nu:</strong>{" "}
                    {Number.isFinite(Nu) ? Nu.toPrecision(4) : "–"}
                </p>
                <small style={{ color: "#555" }}>
                    Evaluated at η=0: cf = f″(0), Nu = -θ′(0)
                </small>
            </div>   
        </div>
    );
}