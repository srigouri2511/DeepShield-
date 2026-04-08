from __future__ import annotations

from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from env.environment import DeepfakePhishingEnv
from env.models import Action as CoreAction
from env.models import TaskName

from .models import DeepShieldAction, DeepShieldObservation, DeepShieldState


DIFFICULTY_DEFAULTS = {
    "easy": TaskName.PHISHING_TRIAGE,
    "medium": TaskName.EMAIL_HEADER_ANALYSIS,
    "hard": TaskName.IDENTITY_FRAUD,
}


def _resolve_task(task: str | TaskName | None, difficulty: str | None = None) -> TaskName:
    if isinstance(task, TaskName):
        return task
    if isinstance(task, str) and task.strip():
        raw_task = task.strip().lower()
        if raw_task in DIFFICULTY_DEFAULTS:
            return DIFFICULTY_DEFAULTS[raw_task]
        return TaskName(raw_task)
    if isinstance(difficulty, str) and difficulty.strip():
        return DIFFICULTY_DEFAULTS.get(
            difficulty.strip().lower(),
            TaskName.PHISHING_TRIAGE,
        )
    return TaskName.PHISHING_TRIAGE


class DeepShieldOpenEnv(Environment[DeepShieldAction, DeepShieldObservation, DeepShieldState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self._env = DeepfakePhishingEnv()
        self._episode_id: str | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str | TaskName | None = None,
        difficulty: str | None = None,
        **kwargs: Any,
    ) -> DeepShieldObservation:
        del seed, kwargs
        self._episode_id = episode_id
        target_task = _resolve_task(task, difficulty)
        observation = self._env.reset(target_task)
        return DeepShieldObservation(
            task=observation.task,
            step=observation.step,
            content=observation.content,
            metadata=observation.metadata,
            done=observation.done,
            reward=None,
        )

    def step(
        self,
        action: DeepShieldAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> DeepShieldObservation:
        del timeout_s, kwargs
        core_action = CoreAction(
            task=action.task,
            decision=action.decision,
            confidence=action.confidence,
            reasoning=action.reasoning,
        )
        observation, reward, done, info = self._env.step(core_action)
        merged_metadata = dict(observation.metadata)
        if info:
            merged_metadata["info"] = info
        return DeepShieldObservation(
            task=observation.task,
            step=observation.step,
            content=observation.content,
            metadata=merged_metadata,
            done=done,
            reward=reward.value,
        )

    @property
    def state(self) -> DeepShieldState:
        current = self._env.state()
        active_task = current.get("active_task", TaskName.PHISHING_TRIAGE)
        return DeepShieldState(
            episode_id=self._episode_id,
            step_count=int(current.get("steps", 0)),
            active_task=active_task.value if isinstance(active_task, TaskName) else str(active_task),
            total_reward=float(current.get("total_reward", 0.0)),
            avg_reward=float(current.get("avg_reward", 0.0)),
            task_state=current.get("task_state", {}),
            history=current.get("history", []),
            last_action_error=current.get("last_action_error"),
            closed=bool(current.get("closed", False)),
        )

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8")
        return EnvironmentMetadata(
            name="DeepShield",
            description="Security benchmark for phishing, URL, header, deepfake, and identity-fraud analysis.",
            version="2.0.0",
            author="srigouri2511",
            readme_content=readme_content,
        )

    def close(self) -> None:
        # Keep close idempotent because the HTTP server may construct and close
        # the shared adapter around individual requests.
        return None
