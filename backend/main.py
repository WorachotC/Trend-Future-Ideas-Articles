from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
import requests
import os
import logging
from dotenv import load_dotenv
from schemas import TopicRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"Generating article for topic: {req.topic}")
    
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
    
    logger.info(f"Generated prompt length: {len(prompt)} chars")

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
    
    logger.info(f"Sending request to HuggingFace API: {API_URL}")

    try:
        response = requests.post(API_URL, headers=HF_HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        
        logger.info(f"Received response with status code: {response.status_code}")
        
        data = response.json()
        logger.info(f"Response data type: {type(data)}")
        
        # Handle different response formats from HuggingFace
        if isinstance(data, list) and len(data) > 0:
            result_text = data[0].get('generated_text', '').strip()
        elif isinstance(data, dict):
            result_text = data.get('generated_text', '').strip()
        else:
            logger.error(f"Unexpected response format: {type(data)}, data: {data}")
            raise ValueError(f"Unexpected response format: {type(data)}")
        
        if not result_text:
            logger.error("Generated text is empty")
            raise ValueError("Generated text is empty")
        
        logger.info(f"Successfully generated article of length: {len(result_text)} chars")
        
        return {
            "topic": req.topic,
            "article": result_text,
            "status": "success"
        }
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error from HuggingFace: {str(e)}")
        # Handle HTTP errors from HuggingFace
        if hasattr(e.response, 'json'):
            try:
                error_data = e.response.json()
                logger.error(f"Error data: {error_data}")
                if 'error' in error_data:
                    error_msg = error_data['error']
                    if 'estimated_time' in error_data:
                        wait_time = error_data['estimated_time']
                        raise HTTPException(
                            status_code=503,
                            detail=f"Model is loading. Please wait about {wait_time:.0f} seconds and try again."
                        )
                    raise HTTPException(status_code=502, detail=f"HuggingFace API Error: {error_msg}")
            except (ValueError, KeyError):
                pass
        raise HTTPException(status_code=502, detail=f"HuggingFace API Error: {str(e)}")
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        raise HTTPException(
            status_code=504,
            detail="Request to AI model timed out. The model may be busy."
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to AI model: {str(e)}"
        )
        
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Parse error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model response: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )