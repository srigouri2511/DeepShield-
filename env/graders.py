from env.models import TaskName


def grade_episode(history: list[dict], task: TaskName) -> dict:
    """
    Deterministic grader for a completed task episode.
    Returns a score between 0.0 and 1.0.
    """
    if not history:
        return {"score": 0.0, "correct": 0, "total": 0, "details": "No steps recorded."}

    total = len(history)
    correct = sum(1 for h in history if h["correct"])
    avg_reward = sum(h["reward"] for h in history) / total

    # accuracy component
    accuracy = correct / total

    # calibration: penalize overconfident wrong answers
    calibration_penalties = sum(
        h["confidence"] for h in history if not h["correct"]
    )
    calibration = max(0.0, 1.0 - calibration_penalties / total)

    # task-specific weighting
    weights = {
        TaskName.PHISHING_TRIAGE:        {"accuracy": 0.5, "reward": 0.3, "calibration": 0.2},
        TaskName.URL_REPUTATION:         {"accuracy": 0.45, "reward": 0.35, "calibration": 0.2},
        TaskName.EMAIL_HEADER_ANALYSIS: {"accuracy": 0.35, "reward": 0.45, "calibration": 0.2},
        TaskName.DEEPFAKE_DETECTION:     {"accuracy": 0.4, "reward": 0.4, "calibration": 0.2},
        TaskName.IDENTITY_FRAUD:         {"accuracy": 0.3, "reward": 0.5, "calibration": 0.2},
    }
    w = weights[task]
    score = w["accuracy"] * accuracy + w["reward"] * avg_reward + w["calibration"] * calibration

    return {
        "score": round(score, 4),
        "accuracy": round(accuracy, 4),
        "avg_reward": round(avg_reward, 4),
        "calibration": round(calibration, 4),
        "correct": correct,
        "total": total,
    }
