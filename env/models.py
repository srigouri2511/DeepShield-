from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum


class TaskName(str, Enum):
    PHISHING_TRIAGE = "phishing_triage"       # easy
    URL_REPUTATION = "url_reputation"         # easy
    EMAIL_HEADER_ANALYSIS = "email_header_analysis"  # medium
    DEEPFAKE_DETECTION = "deepfake_detection"  # medium
    IDENTITY_FRAUD = "identity_fraud"          # hard


class Observation(BaseModel):
    task: TaskName
    step: int
    content: str                          # raw input presented to the agent
    metadata: dict[str, Any] = Field(default_factory=dict)
    done: bool = False


class Action(BaseModel):
    task: TaskName
    decision: str                         # agent's label e.g. "phishing", "real", "fraud"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    correct: bool
    feedback: str
