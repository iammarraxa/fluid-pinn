import axios from "axios";

// const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL : import.meta.env.VITE_API_BASE,
    timeout : 15000,
    headers : {'Content-Type': 'application/json'}
});

export default function solve({scenario, nPoints}){

    if(!scenario){
        throw new Error ("Missing required parameter: scenario");
    }

    if(nPoints==null){
        throw new Error ("Missing required parameter: nPoints");
    }

    console.log("solve called with: ", {scenario, nPoints});

    return api.post("/solve", {scenario, n_points:nPoints}).then((response) => response.data);
}