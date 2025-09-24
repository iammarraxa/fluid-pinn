import { use, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import solve from './api'
import Controls from './components/Controls'
import Charts from './components/Charts'

function App() {

  console.log("API BASE:", import.meta.env.VITE_API_BASE);
  const [result, setResult] = useState(null);
  const [loading, setLoading] =  useState(false);
  const [error, setError] = useState(null);
  const [lastParams, setLastParams] = useState(null);

  async function handleSolve({scenario, nPoints}) {

    if (!Number.isInteger(nPoints) || nPoints < 101 || nPoints > 801) {
      setError("nPoints must be between 101 and 801");
      return;
    }
    
    setLastParams({ scenario, nPoints })
    try{
      setLoading(true);
      setError(null);
      const data = await solve({scenario, nPoints});
      console.log('solve result keys:', Object.keys(data));
      console.log('SOLVE sending:', { scenario, nPoints });
      setResult(data);
    } catch (error) {
      console.error("encountered this error ==> ", error.message);
      setError(error.message);
    } finally {
      setLoading(false);
    }
    
  }

  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "1rem" }}>
      <div style={{ maxWidth: "900px", width: "100%" }}>
        <Controls onSolve={handleSolve} loading = {loading}/>
        {error && (
          <div
            style={{
              background: "#ffe0e0",
              border: "1px solid #ff5c5c",
              borderRadius: "6px",
              padding: "0.75rem",
              marginBottom: "1rem",
              color: "#b30000",
            }}
          >
            <p>{error}</p>
            <button disabled = {loading || !lastParams}
            onClick={() => lastParams && handleSolve(lastParams)}>Retry</button>
          </div>)}

          {loading && (<p style={{ fontStyle: "italic", marginBottom: "0.5rem" }}>Computing...</p>)}

        <Charts data={result} />
      </div>
    </div>
  );
}

export default App
