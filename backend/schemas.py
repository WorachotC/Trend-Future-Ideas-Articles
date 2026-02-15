from pydantic import BaseModel

class TopicRequest(BaseModel):
    topic: str