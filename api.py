from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from pathlib import Path
from uuid import uuid4
from enum import Enum
from werkzeug.utils import secure_filename
from pydantic import BaseModel, ValidationError
from local_env import load_local_env
from ai_assistant import AISecurityAnalyst
from env.environment import DeepfakePhishingEnv
from env.models import TaskName, Action
from env.graders import grade_episode
from env.features import URLReputation, EmailHeaderAnalyzer
from env.tasks import _phishing_score, _laplacian_score

load_local_env()

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"
UPLOAD_DIR = BASE_DIR / "uploads"

TASK_ALIASES = {
    "email": TaskName.PHISHING_TRIAGE,
    "phishing": TaskName.PHISHING_TRIAGE,
    "phishing_triage": TaskName.PHISHING_TRIAGE,
    "url": TaskName.URL_REPUTATION,
    "url_reputation": TaskName.URL_REPUTATION,
    "headers": TaskName.EMAIL_HEADER_ANALYSIS,
    "header": TaskName.EMAIL_HEADER_ANALYSIS,
    "email_header_analysis": TaskName.EMAIL_HEADER_ANALYSIS,
    "image": TaskName.DEEPFAKE_DETECTION,
    "deepfake": TaskName.DEEPFAKE_DETECTION,
    "deepfake_detection": TaskName.DEEPFAKE_DETECTION,
    "identity": TaskName.IDENTITY_FRAUD,
    "identity_fraud": TaskName.IDENTITY_FRAUD,
}

DIFFICULTY_DEFAULTS = {
    "easy": TaskName.PHISHING_TRIAGE,
    "medium": TaskName.EMAIL_HEADER_ANALYSIS,
    "hard": TaskName.IDENTITY_FRAUD,
}

POSITIVE_LABELS = {
    TaskName.PHISHING_TRIAGE: "phishing",
    TaskName.URL_REPUTATION: "suspicious",
    TaskName.EMAIL_HEADER_ANALYSIS: "suspicious",
    TaskName.DEEPFAKE_DETECTION: "deepfake",
    TaskName.IDENTITY_FRAUD: "fraud",
}

NEGATIVE_LABELS = {
    TaskName.PHISHING_TRIAGE: "legitimate",
    TaskName.URL_REPUTATION: "clean",
    TaskName.EMAIL_HEADER_ANALYSIS: "clean",
    TaskName.DEEPFAKE_DETECTION: "real",
    TaskName.IDENTITY_FRAUD: "legitimate",
}

RISK_THRESHOLDS = {
    TaskName.PHISHING_TRIAGE: 0.4,
    TaskName.URL_REPUTATION: 0.4,
    TaskName.EMAIL_HEADER_ANALYSIS: 0.4,
    TaskName.DEEPFAKE_DETECTION: 0.5,
    TaskName.IDENTITY_FRAUD: 0.4,
}

RECOMMENDED_ACTIONS = {
    TaskName.PHISHING_TRIAGE: {
        "phishing": [
            "Do not click links or open attachments from the message.",
            "Verify the request with the sender through a trusted channel.",
            "Quarantine or report the message to your security team.",
        ],
        "legitimate": [
            "Continue with normal verification before responding.",
            "Keep a record if the message is part of an active investigation.",
        ],
    },
    TaskName.URL_REPUTATION: {
        "suspicious": [
            "Avoid opening the link in a primary browser session.",
            "Check the domain registration and destination in a sandbox.",
            "Block or warn on the URL if it targets users or customers.",
        ],
        "clean": [
            "Continue with routine validation if the link is business-critical.",
            "Monitor for redirects or changes if the URL came from an untrusted source.",
        ],
    },
    TaskName.EMAIL_HEADER_ANALYSIS: {
        "suspicious": [
            "Treat the message as spoofed until sender authenticity is confirmed.",
            "Review SPF, DKIM, and DMARC failures in your mail gateway.",
            "Escalate to mail security if the sender targets sensitive workflows.",
        ],
        "clean": [
            "Keep standard review in place if the surrounding message still looks unusual.",
            "Document the clean header result alongside any content-level findings.",
        ],
    },
    TaskName.DEEPFAKE_DETECTION: {
        "deepfake": [
            "Request the original source media before approving or sharing it.",
            "Cross-check with another detector or manual reviewer.",
            "Treat identity-sensitive use cases as high risk until verified.",
        ],
        "real": [
            "Preserve the original file for auditability if it is part of a case.",
            "Use a second opinion if the image affects identity or financial decisions.",
        ],
    },
    TaskName.IDENTITY_FRAUD: {
        "fraud": [
            "Pause the workflow until the identity can be verified out of band.",
            "Request stronger proof of identity or a live verification step.",
            "Alert fraud operations if money movement or account recovery is involved.",
        ],
        "legitimate": [
            "Proceed with normal controls while logging the verification result.",
            "Retain the comparison artifacts if the case may be reviewed later.",
        ],
    },
}

AI_ENGINE = AISecurityAnalyst()
OPENENV_ENV = DeepfakePhishingEnv()


def load_history():
    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open(encoding="utf-8") as f:
                history = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        return history if isinstance(history, list) else []
    return []


def save_history(entry: dict):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = load_history()
        history.insert(0, entry)
        with HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(history[:50], f, indent=2)
    except OSError:
        pass


def parse_payload() -> dict:
    return request.get_json(silent=True) or {}


def json_ready(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return default


def resolve_task(payload: dict) -> TaskName:
    raw_task = str(payload.get("task") or payload.get("type") or "phishing_triage").strip().lower()
    task = TASK_ALIASES.get(raw_task)
    if task is None:
        raise ValueError(raw_task)
    return task


def resolve_openenv_task(payload: dict | None = None) -> TaskName:
    payload = payload or {}
    raw_task = (
        request.args.get("task")
        or payload.get("task")
        or request.args.get("difficulty")
        or payload.get("difficulty")
        or "phishing_triage"
    )
    raw_task = str(raw_task).strip().lower()
    if raw_task in DIFFICULTY_DEFAULTS:
        return DIFFICULTY_DEFAULTS[raw_task]
    task = TASK_ALIASES.get(raw_task)
    if task is None:
        raise ValueError(raw_task)
    return task


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def build_binary_result(task: TaskName, content: str, risk_score: float, flags=None, extra=None) -> dict:
    risk_score = clamp_score(risk_score)
    is_positive = risk_score > RISK_THRESHOLDS[task]
    label = POSITIVE_LABELS[task] if is_positive else NEGATIVE_LABELS[task]
    result = {
        "task": task.value,
        "input": content,
        "label": label,
        "decision": label,
        "confidence": clamp_score(risk_score if is_positive else 1.0 - risk_score),
        "risk_score": risk_score,
        "flags": list(flags or []),
    }
    if extra:
        result.update(extra)
    return result


def label_to_risk(task: TaskName, label: str, confidence: float) -> float:
    return clamp_score(confidence if label == POSITIVE_LABELS[task] else 1.0 - confidence)


def heuristic_recommendations(task: TaskName, label: str) -> list[str]:
    return RECOMMENDED_ACTIONS.get(task, {}).get(label, [])


def heuristic_summary(task: TaskName, result: dict) -> str:
    label = result["label"].replace("_", " ")
    flags = result.get("flags", [])
    flag_text = ", ".join(flags[:3])
    if task == TaskName.PHISHING_TRIAGE:
        return (
            f"This message is currently classified as {label} based on language cues."
            if not flag_text
            else f"This message is currently classified as {label} because it contains cues such as {flag_text}."
        )
    if task == TaskName.URL_REPUTATION:
        return (
            f"This URL is currently classified as {label}."
            if not flag_text
            else f"This URL is currently classified as {label} due to signals like {flag_text}."
        )
    if task == TaskName.EMAIL_HEADER_ANALYSIS:
        return (
            f"These headers are currently classified as {label}."
            if not flag_text
            else f"These headers are currently classified as {label} because of signals such as {flag_text}."
        )
    if task == TaskName.DEEPFAKE_DETECTION:
        heuristic_score = result.get("heuristic_score", result.get("risk_score", 0.0))
        return f"The sample is currently classified as {label} using a deepfake heuristic score of {heuristic_score:.2f}."
    return f"This case is currently classified as {label}."


def heuristic_reasoning(task: TaskName, result: dict) -> str:
    if task == TaskName.DEEPFAKE_DETECTION:
        return "The current decision is grounded in the Laplacian-based image heuristic rather than a full vision model review."
    flags = result.get("flags", [])
    if flags:
        return "The decision is based on the strongest heuristic indicators found in the submitted content."
    return "The decision is based on the available heuristic score for this task."


def build_heuristic_detection(task: TaskName, content: str) -> dict:
    if task == TaskName.PHISHING_TRIAGE:
        risk_score, flags = _phishing_score(content)
        return build_binary_result(task, content, risk_score, flags)
    if task == TaskName.URL_REPUTATION:
        result = URLReputation().analyze(content)
        return {
            "task": task.value,
            "input": content,
            "label": result["label"],
            "decision": result["label"],
            "confidence": result["confidence"],
            "risk_score": result["risk_score"],
            "flags": result["flags"],
            **result,
        }
    if task == TaskName.EMAIL_HEADER_ANALYSIS:
        result = EmailHeaderAnalyzer().analyze(content)
        return {
            "task": task.value,
            "input": content,
            "label": result["label"],
            "decision": result["label"],
            "confidence": result["confidence"],
            "risk_score": result["risk_score"],
            "flags": result["flags"],
            **result,
        }
    if task == TaskName.DEEPFAKE_DETECTION:
        risk_score = _laplacian_score(content)
        return build_binary_result(task, content, risk_score, extra={"heuristic_score": clamp_score(risk_score)})
    if task == TaskName.IDENTITY_FRAUD:
        raise ValueError("Use /api/verify for identity checks that require two images.")
    raise ValueError(f"Unsupported task: {task.value}")


def finalize_detection(task: TaskName, content: str, use_ai: bool = True) -> dict:
    heuristic_result = build_heuristic_detection(task, content)
    ai_result = AI_ENGINE.analyze(task, content, heuristic_result) if use_ai else {
        "available": AI_ENGINE.enabled,
        "used": False,
        "model": AI_ENGINE.model,
    }

    final_result = dict(heuristic_result)
    final_result["heuristic"] = {
        "label": heuristic_result["label"],
        "confidence": heuristic_result["confidence"],
        "risk_score": heuristic_result["risk_score"],
        "flags": list(heuristic_result.get("flags", [])),
    }

    if ai_result.get("used"):
        blended_risk = clamp_score(
            heuristic_result["risk_score"] * 0.45
            + label_to_risk(task, ai_result["label"], ai_result["confidence"]) * 0.55
        )
        blended = build_binary_result(task, content, blended_risk, heuristic_result.get("flags"))
        final_result.update({
            "label": blended["label"],
            "decision": blended["label"],
            "confidence": blended["confidence"],
            "risk_score": blended["risk_score"],
            "analysis_mode": "hybrid",
            "summary": ai_result.get("summary") or heuristic_summary(task, heuristic_result),
            "reasoning": ai_result.get("reasoning") or heuristic_reasoning(task, heuristic_result),
            "evidence": ai_result.get("evidence") or list(heuristic_result.get("flags", []))[:4],
            "recommended_actions": ai_result.get("actions") or heuristic_recommendations(task, blended["label"]),
            "ai": ai_result,
        })
    else:
        final_result.update({
            "analysis_mode": "heuristic",
            "summary": heuristic_summary(task, heuristic_result),
            "reasoning": heuristic_reasoning(task, heuristic_result),
            "evidence": list(heuristic_result.get("flags", []))[:4],
            "recommended_actions": heuristic_recommendations(task, heuristic_result["label"]),
            "ai": ai_result,
        })

    return final_result


def build_heuristic_action(task: TaskName, obs) -> Action:
    if task == TaskName.PHISHING_TRIAGE:
        risk_score, _ = _phishing_score(obs.content)
        result = build_binary_result(task, obs.content, risk_score)
    elif task == TaskName.URL_REPUTATION:
        result = obs.metadata or URLReputation().analyze(obs.content)
    elif task == TaskName.EMAIL_HEADER_ANALYSIS:
        result = obs.metadata or EmailHeaderAnalyzer().analyze(obs.content)
    elif task == TaskName.DEEPFAKE_DETECTION:
        risk_score = obs.metadata.get("heuristic_score", 0.0)
        result = build_binary_result(task, obs.content, risk_score)
    else:
        face_similarity = float(obs.metadata.get("face_similarity", 0.0))
        email_risk = float(obs.metadata.get("email_risk", _phishing_score(obs.content)[0]))
        fraud_risk = ((1.0 - face_similarity) + email_risk) / 2.0
        result = build_binary_result(task, obs.content, fraud_risk)

    return Action(
        task=task,
        decision=result["label"],
        confidence=clamp_score(result["confidence"]),
        reasoning="heuristic",
    )


@app.route("/health", methods=["GET"])
def health():
    state = OPENENV_ENV.state()
    return jsonify({
        "status": "ok",
        "name": "deepfake-phishing-detection",
        "active_task": json_ready(state["active_task"]),
        "steps": state["steps"],
        "openenv": True,
    })


@app.route("/reset", methods=["POST"])
def openenv_reset():
    payload = parse_payload()
    try:
        task = resolve_openenv_task(payload)
    except ValueError as exc:
        return jsonify({"error": f"Unknown task or difficulty: {exc}"}), 400

    observation = OPENENV_ENV.reset(task)
    return jsonify({
        "observation": json_ready(observation),
        "info": {"task": task.value},
    })


@app.route("/step", methods=["POST"])
def openenv_step():
    payload = parse_payload()
    action_payload = payload.get("action", payload)
    try:
        action = Action.model_validate(action_payload)
    except ValidationError as exc:
        return jsonify({"error": "Invalid action payload", "details": exc.errors()}), 400

    try:
        observation, reward, done, info = OPENENV_ENV.step(action)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "observation": json_ready(observation),
        "reward": json_ready(reward),
        "done": bool(done),
        "info": json_ready(info),
    })


@app.route("/state", methods=["GET"])
def openenv_state():
    return jsonify({"state": json_ready(OPENENV_ENV.state())})


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "ai_enabled": AI_ENGINE.enabled,
        "model": AI_ENGINE.model,
        "history_count": len(load_history()),
        "default_analysis_mode": "hybrid" if AI_ENGINE.enabled else "heuristic",
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    data = parse_payload()
    content = str(data.get("input", "")).strip()
    use_ai = parse_bool(data.get("use_ai"), True)
    if not content:
        return jsonify({"error": "Missing 'input'"}), 400

    try:
        task = resolve_task(data)
    except ValueError:
        raw_task = data.get("task") or data.get("type")
        return jsonify({"error": f"Unknown task: {raw_task}"}), 400

    try:
        output = finalize_detection(task, content, use_ai=use_ai)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    save_history({
        "type": output["task"],
        "input": content[:120],
        "label": output["label"],
        "confidence": output["confidence"],
        "risk_score": output.get("risk_score"),
        "analysis_mode": output.get("analysis_mode"),
        "summary": output.get("summary", "")[:140],
    })
    return jsonify(output)


@app.route("/api/detect/url/reputation", methods=["POST"])
def url_reputation():
    data = parse_payload()
    url = str(data.get("url", "")).strip()
    use_ai = parse_bool(data.get("use_ai"), True)
    if not url:
        return jsonify({"error": "Missing 'url'"}), 400
    output = finalize_detection(TaskName.URL_REPUTATION, url, use_ai=use_ai)
    save_history({"type": "url_reputation", "input": url, "label": output["label"],
                  "confidence": output["confidence"], "risk_score": output["risk_score"],
                  "analysis_mode": output.get("analysis_mode"), "summary": output.get("summary", "")[:140]})
    return jsonify(output)


@app.route("/api/detect/email/headers", methods=["POST"])
def email_headers():
    data = parse_payload()
    headers = str(data.get("headers", "")).strip()
    use_ai = parse_bool(data.get("use_ai"), True)
    if not headers:
        return jsonify({"error": "Missing 'headers'"}), 400
    output = finalize_detection(TaskName.EMAIL_HEADER_ANALYSIS, headers, use_ai=use_ai)
    save_history({"type": "email_headers", "input": headers[:80], "label": output["label"],
                  "confidence": output["confidence"], "risk_score": output["risk_score"],
                  "analysis_mode": output.get("analysis_mode"), "summary": output.get("summary", "")[:140]})
    return jsonify(output)


@app.route("/api/verify", methods=["POST"])
def verify():
    if "image_a" not in request.files or "image_b" not in request.files:
        return jsonify({"error": "Upload both image_a and image_b"}), 400
    UPLOAD_DIR.mkdir(exist_ok=True)

    file_a = request.files["image_a"]
    file_b = request.files["image_b"]
    context = request.form.get("context", "").strip()
    name_a = secure_filename(file_a.filename) or "image_a.png"
    name_b = secure_filename(file_b.filename) or "image_b.png"
    path_a = UPLOAD_DIR / f"{uuid4().hex}_a_{name_a}"
    path_b = UPLOAD_DIR / f"{uuid4().hex}_b_{name_b}"
    file_a.save(path_a)
    file_b.save(path_b)

    from env.identity import IdentityVerifier
    result = IdentityVerifier().verify(str(path_a), str(path_b))
    summary = (
        "The uploaded images appear to represent the same person."
        if result["match"]
        else "The uploaded images appear to represent different people."
    )
    recommendations = (
        ["Continue with normal identity controls and retain the comparison for audit."]
        if result["match"]
        else [
            "Pause the verification flow until identity is confirmed out of band.",
            "Request stronger identity proof or a live verification step.",
        ]
    )
    output = {
        **result,
        "context": context,
        "analysis_mode": "heuristic",
        "summary": summary,
        "reasoning": "This verification is based on the similarity score derived from the uploaded images.",
        "recommended_actions": recommendations,
        "ai": {
            "available": AI_ENGINE.enabled,
            "used": False,
            "model": AI_ENGINE.model,
        },
    }
    save_history({"type": "identity_verify", "input": f"{path_a.name} vs {path_b.name}",
                  "label": result["label"], "confidence": result["confidence"],
                  "risk_score": result["similarity"], "analysis_mode": "heuristic",
                  "summary": summary[:140]})
    return jsonify(output)


@app.route("/api/episode", methods=["POST"])
def run_episode():
    """Run a full task episode and return graded results."""
    data = parse_payload()
    try:
        task = resolve_task(data)
    except ValueError:
        raw_task = data.get("task") or data.get("type")
        return jsonify({"error": f"Unknown task: {raw_task}"}), 400

    env = DeepfakePhishingEnv()
    obs = env.reset(task)
    steps = []
    while not obs.done:
        action = build_heuristic_action(task, obs)
        obs, reward, done, info = env.step(action)
        steps.append({
            "decision": action.decision,
            "confidence": action.confidence,
            "reward": reward.value,
            "correct": reward.correct,
        })
        if done:
            break

    state = env.state()
    grade = grade_episode(state["history"], task)
    return jsonify({"task": task.value, "grade": grade, "steps": steps})


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
    except OSError:
        pass
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
