from pydantic import BaseModel

class TopicRequest(BaseModel):
    topic: str
    industry: str = "Technology"
    target_audience: str = "Business Owners"
    tone: str = "Professional"
    source_url: str = ""