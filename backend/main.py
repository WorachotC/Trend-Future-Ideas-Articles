from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
from llama_cpp import Llama
from schemas import TopicRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Jenosize AI Writer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "default_secret_key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

# --- Load Local GGUF Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "chinda-qwen3-4b.Q4_K_M.gguf")

logger.info(f"⏳ Loading Local Model from: {MODEL_PATH}")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    llm = None

# --- SYSTEM PROMPTS (One-time instruction for all cases) ---
SYSTEM_PROMPT = """You are a professional article writer for Jenosize - a company known for authentic, human-centered insights about business transformation.

Your writing style:
- Authentic and human-centered (focus on customer understanding, not technology hype)
- Conversational yet insightful (warm, relatable tone, never robotic)
- Story-driven (use real-world examples and narratives)
- Actionable (provide practical insights readers can implement)
- Grounded in reality (honest about both opportunities and challenges)

CRITICAL OUTPUT RULES:
1. Write ONLY the article content - no thinking, reasoning, or meta-commentary
2. Start directly with a compelling markdown headline (#)
3. Use proper markdown formatting (###, ####, bullet points)
4. Include real-world examples and actionable insights
5. Never explain your approach or process
6. Focus on customer value and business impact
7. Maintain authentic, warm tone throughout"""

SYSTEM_PROMPT_THAI = """คุณเป็นนักเขียนบทความมืออาชีพสำหรับ Jenosize - บริษัทที่เป็นที่รู้จักในด้านข้อมูลเชิงลึกที่มีจุดศูนย์กลางมนุษย์เกี่ยวกับการเปลี่ยนแปลงธุรกิจ

สไตล์การเขียนของคุณ:
- ตัวตนและมีจุดศูนย์กลางมนุษย์ (เน้นความเข้าใจลูกค้า ไม่ใช่การโปรโมตเทคโนโลยี)
- เป็นการสนทนา แต่มีความลึกซึ้ง (โทนเสียงที่อบอุ่น เข้าถึงได้ ไม่เคยเป็นหุ่นยนต์)
- ขับเคลื่อนด้วยเรื่องราว (ใช้ตัวอย่างและการบรรยายจากโลกจริง)
- ใช้ได้จริง (ให้ข้อมูลเชิงลึกปฏิบัติที่ผู้อ่านสามารถนำไปใช้)
- มีพื้นฐานในความเป็นจริง (ซื่อสัตย์เกี่ยวกับโอกาสและความท้าทายทั้งสองด้าน)

กฎการส่งออกที่สำคัญ:
1. เขียนเฉพาะเนื้อหาบทความ - ไม่มีการคิด ให้เหตุผล หรือคิดความเห็นทั่วไป
2. เริ่มต้นโดยตรงด้วยหัวเรื่องที่น่าสนใจ markdown (#)
3. ใช้การจัดรูปแบบ markdown ที่เหมาะสม (###, ####, จุดหลัก)
4. รวมตัวอย่างจากโลกจริงและข้อมูลเชิงลึกที่ใช้ได้จริง
5. อย่าเคยอธิบายวิธีการหรือกระบวนการของคุณ
6. เน้นมูลค่าของลูกค้าและผลกระทบต่อธุรกิจ
7. รักษาโทนเสียงที่ตัวตนและอบอุ่นตลอดเวลา"""

@app.get("/")
def read_root():
    return {"status": "🚀 Jenosize AI Backend is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}

@app.get("/model-status")
def model_status():
    return {
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    }

@app.post("/generate-article")
def generate_article(req: TopicRequest, api_key: str = Depends(verify_api_key)):
    """
    Generate article in both English and Thai.
    Uses unified system prompt approach for consistency.
    """
    if not llm:
        raise HTTPException(status_code=500, detail="AI Model is not loaded properly.")

    logger.info(f"Generating bilingual article - Topic: {req.topic}, Tone: {req.tone}")
    
    # Simple parameter mapping based on tone
    tone_params = {
        "Casual": {"temperature": 0.75, "top_p": 0.88},
        "Professional": {"temperature": 0.65, "top_p": 0.85},
        "Visionary": {"temperature": 0.80, "top_p": 0.90},
        "Urgent": {"temperature": 0.70, "top_p": 0.85},
    }
    
    params = tone_params.get(req.tone, {"temperature": 0.70, "top_p": 0.85})

    try:
        # Generate English article
        logger.info("Generating English article...")
        eng_prompt = f"""{SYSTEM_PROMPT}

Topic: {req.topic}
Industry: {req.industry}
Target Audience: {req.target_audience}
Tone: {req.tone}
{f"Reference: {req.source_url}" if req.source_url else ""}

Write the article:
"""

        eng_output = llm(
            eng_prompt,
            max_tokens=1000,
            temperature=params["temperature"],
            top_p=params["top_p"],
            echo=False
        )
        
        eng_article = eng_output['choices'][0]['text'].strip()
        logger.info("✅ English article generated!")

        # Generate Thai article
        logger.info("Generating Thai article...")
        thai_prompt = f"""{SYSTEM_PROMPT_THAI}

หัวข้อ: {req.topic}
อุตสาหกรรม: {req.industry}
กลุ่มผู้ชมเป้าหมาย: {req.target_audience}
โทน: {req.tone}
{f"อ้างอิง: {req.source_url}" if req.source_url else ""}

เขียนบทความเป็นภาษาไทย:
"""

        thai_output = llm(
            thai_prompt,
            max_tokens=1000,
            temperature=params["temperature"],
            top_p=params["top_p"],
            echo=False
        )
        
        thai_article = thai_output['choices'][0]['text'].strip()
        logger.info("✅ Thai article generated!")
        
        return {
            "topic": req.topic,
            "industry": req.industry,
            "target_audience": req.target_audience,
            "tone": req.tone,
            "articles": {
                "en": eng_article,
                "th": thai_article
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")