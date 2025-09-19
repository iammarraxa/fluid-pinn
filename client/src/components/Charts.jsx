
export default function Charts({data}){
    if(!data){
        return <p>Run a solve func to see plots</p>;
    }
    return(
        <div>
            <h1>Scenario: {data.Scenario}</h1>
            <div style={{ border: "1px solid #ccc", margin: "8px 0", padding: "4px" }}>
                f′(η) plot here
            </div>
            <div style={{ border: "1px solid #ccc", margin: "8px 0", padding: "4px" }}>
                θ(η) plot here
            </div>
            <div style={{ marginTop: "12px" }}>
                <strong>Metrics:</strong>
                <div>cf: {data?.wall?.cf}</div>
                <div>Nu: {data?.wall?.Nu}</div>
            </div>   
        </div>
    );
}