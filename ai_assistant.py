import json
import os
from typing import Any

from local_env import load_local_env
from env.models import TaskName

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - defensive import guard
    OpenAI = None


load_local_env()


VALID_LABELS = {
    TaskName.PHISHING_TRIAGE: {"phishing", "legitimate"},
    TaskName.URL_REPUTATION: {"suspicious", "clean"},
    TaskName.EMAIL_HEADER_ANALYSIS: {"suspicious", "clean"},
    TaskName.DEEPFAKE_DETECTION: {"deepfake", "real"},
    TaskName.IDENTITY_FRAUD: {"fraud", "legitimate"},
}


TASK_GUIDANCE = {
    TaskName.PHISHING_TRIAGE: (
        "Classify the email content as phishing or legitimate. Focus on social engineering,"
        " urgency, credential theft signals, and impersonation cues."
    ),
    TaskName.URL_REPUTATION: (
        "Classify the URL as suspicious or clean. Focus on domain quality, impersonation,"
        " obfuscation, and suspicious path patterns."
    ),
    TaskName.EMAIL_HEADER_ANALYSIS: (
        "Classify the headers as suspicious or clean. Focus on SPF, DKIM, DMARC,"
        " mismatched reply-to values, and spoofing indicators."
    ),
    TaskName.DEEPFAKE_DETECTION: (
        "Classify the sample as deepfake or real using the provided heuristic score and context."
        " Do not claim visual certainty beyond the supplied evidence."
    ),
    TaskName.IDENTITY_FRAUD: (
        "Classify the case as fraud or legitimate using the provided face similarity and"
        " email risk context."
    ),
}


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("Model response did not contain JSON.")
    payload = json.loads(text[start:end])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON was not an object.")
    return payload


def _normalize_items(values: Any) -> list[str]:
    if isinstance(values, list):
        items = values
    elif isinstance(values, str) and values.strip():
        items = [values.strip()]
    else:
        items = []

    cleaned = []
    for value in items:
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:4]


class AISecurityAnalyst:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        self._client = OpenAI(api_key=self.api_key) if self.api_key and OpenAI is not None else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def analyze(
        self,
        task: TaskName,
        content: str,
        heuristic_result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_response = {
            "available": self.enabled,
            "used": False,
            "model": self.model,
        }
        if not self.enabled:
            return base_response

        metadata = metadata or {}
        heuristic_snapshot = json.dumps(heuristic_result, ensure_ascii=True, indent=2)
        metadata_snapshot = json.dumps(metadata, ensure_ascii=True, indent=2)
        allowed_labels = sorted(VALID_LABELS[task])

        system_prompt = (
            "You are an experienced security analyst assisting a threat-analysis dashboard. "
            "Return JSON only with keys: "
            "label, confidence, summary, reasoning, evidence, actions. "
            "Confidence must be between 0 and 1. "
            "Evidence and actions must be short string arrays."
        )
        user_prompt = (
            f"Task: {task.value}\n"
            f"Allowed labels: {', '.join(allowed_labels)}\n"
            f"Objective: {TASK_GUIDANCE[task]}\n\n"
            f"Content:\n{content}\n\n"
            f"Heuristic result:\n{heuristic_snapshot}\n\n"
            f"Additional metadata:\n{metadata_snapshot}\n\n"
            "Keep the explanation concise, grounded in the provided evidence, and operationally useful."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=350,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json_object(raw)
            label = str(parsed.get("label") or parsed.get("decision") or "").strip().lower()
            if label not in VALID_LABELS[task]:
                raise ValueError(f"Model returned unsupported label '{label}'.")

            return {
                "available": True,
                "used": True,
                "model": self.model,
                "label": label,
                "confidence": round(_clamp_confidence(parsed.get("confidence")), 4),
                "summary": str(parsed.get("summary") or "").strip(),
                "reasoning": str(parsed.get("reasoning") or "").strip(),
                "evidence": _normalize_items(parsed.get("evidence")),
                "actions": _normalize_items(parsed.get("actions")),
            }
        except Exception as exc:  # pragma: no cover - depends on external service
            return {
                **base_response,
                "error": str(exc),
            }
