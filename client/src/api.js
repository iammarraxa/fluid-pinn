import axios from "axios";

// const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL : import.meta.env.VITE_API_BASE,
    timeout : 15000,
    headers : {'Content-Type': 'application/json'}
});

export default async function solve({scenario, nPoints, Sq, S, lam, M, delta }){

    if(!scenario){
        throw new Error ("Missing required parameter: scenario");
    }

    if(nPoints==null){
        throw new Error ("Missing required parameter: nPoints");
    }

    try{
        const response = await api.post("/solve", { scenario, n_points: nPoints, Sq, S, lam, M, delta });
        return response.data;
    } catch(err) {
        let message;

        if (err.response?.data?.detail){
            message = err.response.data.detail;
        } else if (err.code == 'ECONNABORTED'){
            message = "Request timed out";
        } else if (err.message?.includes("Network")){
            message = "Network error (is the server running?)";
        } else {
            message = "Unexpected Error";
        }

        throw new Error(message);
    }
}