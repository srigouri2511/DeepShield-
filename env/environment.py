from env.models import TaskName, Observation, Action, Reward
from env.tasks import (
    PhishingTriageTask,
    URLReputationTask,
    EmailHeaderAnalysisTask,
    DeepfakeDetectionTask,
    IdentityFraudTask,
)


class DeepfakePhishingEnv:
    """
    OpenEnv-compliant environment for AI Deepfake + Phishing Detection.
    Implements: reset(), step(), state()
    """

    def __init__(self, deepfake_samples=None, identity_samples=None):
        self._tasks = {
            TaskName.PHISHING_TRIAGE: PhishingTriageTask(),
            TaskName.URL_REPUTATION: URLReputationTask(),
            TaskName.EMAIL_HEADER_ANALYSIS: EmailHeaderAnalysisTask(),
            TaskName.DEEPFAKE_DETECTION: DeepfakeDetectionTask(samples=deepfake_samples),
            TaskName.IDENTITY_FRAUD: IdentityFraudTask(samples=identity_samples),
        }
        self._active: TaskName = TaskName.PHISHING_TRIAGE
        self._history: list[dict] = []
        self._total_reward: float = 0.0
        self._steps: int = 0
        self._last_action_error: str | None = None
        self._closed: bool = False

    def reset(self, task: TaskName = TaskName.PHISHING_TRIAGE) -> Observation:
        self._active = task
        self._history.clear()
        self._total_reward = 0.0
        self._steps = 0
        self._last_action_error = None
        self._closed = False
        return self._tasks[task].reset()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._closed:
            raise RuntimeError("Environment is closed. Call reset() before step().")

        task = self._tasks[self._active]
        obs, reward, done, info = task.step(action)
        self._last_action_error = None

        if action.task != self._active:
            self._last_action_error = (
                f"Action task '{action.task.value}' does not match active task '{self._active.value}'."
            )
            reward.value = 0.0
            reward.correct = False
            reward.feedback = self._last_action_error
            info["task_mismatch"] = True

        if reward.feedback == "Invalid decision label.":
            self._last_action_error = reward.feedback

        # penalize infinite loops: same decision repeated > 3 times
        recent = [h["decision"] for h in self._history[-3:]]
        if len(recent) == 3 and len(set(recent)) == 1:
            reward.value = max(0.0, reward.value - 0.1)
            info["loop_penalty"] = True
        info["last_action_error"] = self._last_action_error

        self._total_reward += reward.value
        self._steps += 1
        self._history.append({
            "step": self._steps,
            "decision": action.decision,
            "confidence": action.confidence,
            "reward": reward.value,
            "correct": reward.correct,
        })
        return obs, reward, done, info

    def state(self) -> dict:
        return {
            "active_task": self._active,
            "steps": self._steps,
            "total_reward": round(self._total_reward, 4),
            "avg_reward": round(self._total_reward / self._steps, 4) if self._steps else 0.0,
            "task_state": self._tasks[self._active].state(),
            "history": self._history,
            "last_action_error": self._last_action_error,
            "closed": self._closed,
        }

    def close(self) -> None:
        self._closed = True
