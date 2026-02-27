import os
import re
import pandas as pd
import itertools
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from predictor import SupplierRiskPredictor

# -------------------- Load environment --------------------
load_dotenv("Risk.env")

# -------------------- Initialize FastAPI --------------------
app = FastAPI(title="AI Orchestrator: Healthcare Risk Investigation")

PROJECT_NAME = "AI Orchestrator"

# -------------------- Initialize OpenAI --------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(question: str) -> str:
    # Using a standard reliable model name
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional healthcare risk investigator."},
            {"role": "user", "content": question}
        ],
        temperature=0.1 # Low temperature for factual consistency
    )
    return response.choices[0].message.content

# -------------------- Initialize Predictor --------------------
predictor = SupplierRiskPredictor()

# Load dataset once
if not os.path.exists("final_supplier_dataset.csv"):
    raise FileNotFoundError("final_supplier_dataset.csv not found.")

df = pd.read_csv("final_supplier_dataset.csv")
risk_cols = ["Tot_Suplr_Benes", "Tot_Suplr_Srvcs", "Avg_Suplr_Sbmtd_Chrg", 
             "Avg_Suplr_Mdcr_Alowd_Amt", "Avg_Suplr_Mdcr_Pymt_Amt"]

# -------------------- Request Models --------------------
class SupplierInput(BaseModel):
    data: dict  # Raw JSON input for /predict

class QueryRequest(BaseModel):
    user_query: str  # Natural language question for /Ask-AI

# -------------------- /predict endpoint --------------------
@app.post("/predict")
def predict_supplier(input_data: SupplierInput):
    try:
        result = predictor.predict(input_data.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- /ASK_AI endpoint --------------------

def extract_entities(query: str, df: pd.DataFrame):
    q_lower = query.lower()
    found = {
        "Suplr_Prvdr_City": [c for c in df["Suplr_Prvdr_City"].dropna().unique() if c.lower() in q_lower],
        "Suplr_Prvdr_Spclty_Desc": [s for s in df["Suplr_Prvdr_Spclty_Desc"].dropna().unique() if s.lower() in q_lower],
        "Suplr_Prvdr_Last_Name_Org": [o for o in df["Suplr_Prvdr_Last_Name_Org"].dropna().unique() if o.lower() in q_lower],
        "Suplr_Prvdr_RUCA_Cat": []
    }
    if "urban" in q_lower: found["Suplr_Prvdr_RUCA_Cat"].append("Urban")
    if "rural" in q_lower: found["Suplr_Prvdr_RUCA_Cat"].append("Rural")
    return {k: v for k, v in found.items() if v}

@app.post("/Ask-AI")
async def ask_ai_orchestrator(request: QueryRequest):
    try:
        query = request.user_query.lower()
        active_filters = extract_entities(query, df)
        
        # Filter for High Risk base
        df_hr = df[df["High_Risk"] == 1].copy()
        df_hr["risk_score"] = df_hr[risk_cols].sum(axis=1)

        # -------------------- LOGIC 1: COUNT REQUESTS (GENERAL & FILTERED) --------------------
        if any(word in query for word in ["number", "count", "how many"]):
            # General Count Fix: If no city/category filters are found
            if not active_filters:
                total_count = len(df_hr)
                prompt = f"The user asked for the total number of high-risk suppliers. The total count in the dataset is {total_count}. Please state this clearly."
                return {"AI_Response": ask_llm(prompt).split("\n")}
            
            # Filtered Count Breakdown
            intersect_df = df_hr.copy()
            for col, vals in active_filters.items():
                intersect_df = intersect_df[intersect_df[col].astype(str).str.contains('|'.join(vals), na=False, case=False)]
            
            breakdowns = {}
            for col, vals in active_filters.items():
                for val in vals:
                    breakdowns[val] = len(df_hr[df_hr[col].astype(str).str.contains(val, na=False, case=False)])

            prompt = f"Total matching: {len(intersect_df)}. Individual Filter Breakdown: {breakdowns}. Summarize the results clearly."
            return {"AI_Response": ask_llm(prompt).split("\n")}

        # -------------------- LOGIC 2: SELECTION (TOP/BOTTOM) --------------------
        num_match = re.search(r"(\d+)", query)
        top_n = int(num_match.group(1)) if num_match else 1
        is_bottom = any(w in query for w in ["bottom", "least", "lowest"])
        sort_asc = True if is_bottom else False

        evidence = ""
        if not active_filters:
            # Global Rank
            final_df = df_hr.sort_values(by="risk_score", ascending=sort_asc).head(top_n)
            for i, (idx, row) in enumerate(final_df.iterrows(), 1):
                evidence += f"ITEM_{i}: {row['Suplr_Prvdr_Last_Name_Org']} in {row['Suplr_Prvdr_City']}\n"
        else:
            # Grouped Rank
            keys = list(active_filters.keys())
            for combo in itertools.product(*[active_filters[k] for k in keys]):
                group_header = " ".join(combo).upper()
                evidence += f"\n Filter: {group_header}\n"
                
                temp_df = df_hr.copy()
                for i, val in enumerate(combo):
                    temp_df = temp_df[temp_df[keys[i]].astype(str).str.contains(val, na=False, case=False)]
                
                if temp_df.empty:
                    evidence += "NO_RECORDS_FOUND\n"
                else:
                    combo_top = temp_df.sort_values(by="risk_score", ascending=sort_asc).head(top_n)
                    for j, (idx, row) in enumerate(combo_top.iterrows(), 1):
                        evidence += f"ITEM_{j}: {row['Suplr_Prvdr_Last_Name_Org']} in {row['Suplr_Prvdr_City']}\n"

        # -------------------- LOGIC 3: FORMATTED RESPONSE --------------------
        prompt = f"""
        USER QUERY: {query}
        EVIDENCE DATA:
        {evidence}
        
        STRICT INSTRUCTIONS:
        1. Start with '{query}:'.
        2. Maintain 'Filter:' headers if they exist in the evidence.
        3. For every ITEM in the evidence, use this EXACT layout:
           - [Organization Name] ([City])
           - Exaplanation: 1-2 sentences explaining risk using metrics: Tot_Suplr_Benes, Tot_Suplr_Srvcs, Avg_Suplr_Sbmtd_Chrg, Avg_Suplr_Mdcr_Alowd_Amt, and Avg_Suplr_Mdcr_Pymt_Amt.
        4. Do NOT show risk scores. List every single item from the evidence.
        """
        return {"AI_Response": ask_llm(prompt).split("\n")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)