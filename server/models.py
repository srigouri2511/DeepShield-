from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState

from env.models import TaskName


class DeepShieldAction(OpenEnvAction):
    task: TaskName
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class DeepShieldObservation(OpenEnvObservation):
    task: TaskName
    step: int
    content: str


class DeepShieldState(OpenEnvState):
    active_task: str = TaskName.PHISHING_TRIAGE.value
    total_reward: float = 0.0
    avg_reward: float = 0.0
    task_state: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    last_action_error: str | None = None
    closed: bool = False
