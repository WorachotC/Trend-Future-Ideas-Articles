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
    # Dynamic instruction based on tone
    tone_instructions = {
        "Casual": "Write a friendly and conversational trend analysis article",
        "Professional": "Write a professional and insightful trend analysis article",
        "Visionary": "Write an inspiring and forward-thinking trend analysis article",
        "Urgent": "Write a compelling and action-driven trend analysis article"
    }
    
    base_instruction = tone_instructions.get(req.tone, "Write a creative trend analysis article")
    instruction = f"{base_instruction} in Jenosize's style for {req.target_audience} in the {req.industry} industry."
    
    # Build detailed input with all parameters
    input_text = f"Topic: {req.topic}"
    
    # Add source URL if provided
    if req.source_url:
        input_text += f"\nSource Reference: {req.source_url}"
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    # Dynamic parameters based on tone
    tone_params = {
        "Casual": {"temperature": 0.8, "top_p": 0.92, "max_tokens": 700},
        "Professional": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 800},
        "Visionary": {"temperature": 0.85, "top_p": 0.95, "max_tokens": 900},
        "Urgent": {"temperature": 0.75, "top_p": 0.88, "max_tokens": 750}
    }
    
    params = tone_params.get(req.tone, {"temperature": 0.7, "top_p": 0.9, "max_tokens": 800})
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "return_full_text": False
        }
    }