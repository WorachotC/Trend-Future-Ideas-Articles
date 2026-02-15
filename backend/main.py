from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
import requests
import os
from dotenv import load_dotenv
from schemas import TopicRequest

load_dotenv()

app = FastAPI(title="Jenosize AI Writer API")

# Retrieve environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO", "mix8645/jenosize-qwen3-4b")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "default_secret_key")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not found in .env file!")

API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_REPO}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates the provided API key against the environment variable."""
    if api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

@app.get("/")
def read_root():
    return {"status": "🚀 Jenosize AI Backend is running (Secure Mode)!"}

@app.post("/generate-article")
def generate_article(req: TopicRequest, api_key: str = Depends(verify_api_key)):
    # Format the prompt to match the training data instruction
    instruction = "Write a creative trend analysis article in Jenosize's professional style."
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\nTopic: {req.topic}\n\n### Response:\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=HF_HEADERS, json=payload)
        response.raise_for_status()
        
        data = response.json()
        result_text = data[0]['generated_text'].strip()
        
        return {
            "topic": req.topic,
            "article": result_text,
            "status": "success"
        }
        
    except Exception as e:
        if 'response' in locals() and hasattr(response, 'json'):
            try:
                error_data = response.json()
                if 'estimated_time' in error_data:
                    wait_time = error_data['estimated_time']
                    raise HTTPException(
                        status_code=503, 
                        detail=f"Model is waking up... Please wait about {wait_time:.0f} seconds and try again."
                    )
            except ValueError:
                pass # Ignore json parsing errors here
            
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")