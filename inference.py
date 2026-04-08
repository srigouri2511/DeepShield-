import argparse
import json
import os

from openai import OpenAI

from local_env import load_local_env
from env.environment import DeepfakePhishingEnv
from env.graders import grade_episode
from env.models import Action, Observation, TaskName

load_local_env()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "hybrid")
BENCHMARK_NAME = "deepfake-phishing-detection"

HYBRID_PROVIDER_BY_TASK = {
    TaskName.PHISHING_TRIAGE: "openai",
    TaskName.URL_REPUTATION: "hf",
    TaskName.EMAIL_HEADER_ANALYSIS: "openai",
    TaskName.DEEPFAKE_DETECTION: "hf",
    TaskName.IDENTITY_FRAUD: "openai",
}

TASK_PROMPTS = {
    TaskName.PHISHING_TRIAGE: (
        "You are a cybersecurity analyst.\n"
        "Classify the email as 'phishing' or 'legitimate'.\n"
        "Return JSON only with keys: decision, confidence, reasoning.\n\n"
        "Email:\n{content}"
    ),
    TaskName.URL_REPUTATION: (
        "You are a URL security analyst.\n"
        "Classify the URL as 'suspicious' or 'clean'.\n"
        "Return JSON only with keys: decision, confidence, reasoning.\n\n"
        "URL:\n{content}\n"
        "Signals:\n{metadata}"
    ),
    TaskName.EMAIL_HEADER_ANALYSIS: (
        "You are an email authentication analyst.\n"
        "Classify the headers as 'suspicious' or 'clean'.\n"
        "Return JSON only with keys: decision, confidence, reasoning.\n\n"
        "Headers:\n{content}\n"
        "Signals:\n{metadata}"
    ),
    TaskName.DEEPFAKE_DETECTION: (
        "You are a deepfake detection analyst.\n"
        "Classify the sample as 'deepfake' or 'real'.\n"
        "Use the path and heuristic metadata.\n"
        "Return JSON only with keys: decision, confidence, reasoning.\n\n"
        "Sample:\n{content}\n"
        "Signals:\n{metadata}"
    ),
    TaskName.IDENTITY_FRAUD: (
        "You are a fraud analyst.\n"
        "Classify the case as 'fraud' or 'legitimate'.\n"
        "Use the email context and the metadata fields.\n"
        "Return JSON only with keys: decision, confidence, reasoning.\n\n"
        "Context:\n{content}\n"
        "Signals:\n{metadata}"
    ),
}

DEFAULT_DECISIONS = {
    TaskName.PHISHING_TRIAGE: "legitimate",
    TaskName.URL_REPUTATION: "clean",
    TaskName.EMAIL_HEADER_ANALYSIS: "clean",
    TaskName.DEEPFAKE_DETECTION: "real",
    TaskName.IDENTITY_FRAUD: "legitimate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenEnv baseline inference runner")
    parser.add_argument(
        "--task",
        default="all",
        choices=[task.value for task in TaskName] + ["all"],
        help="Task to run. Defaults to all tasks.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Backward-compatible model override. Applies to the active provider in single-provider mode and to Hugging Face in hybrid mode.",
    )
    parser.add_argument(
        "--provider",
        default=MODEL_PROVIDER,
        choices=["hf", "openai", "hybrid"],
        help="Inference backend strategy. Defaults to MODEL_PROVIDER or hybrid.",
    )
    parser.add_argument(
        "--hf-model",
        default=MODEL_NAME,
        help="Hugging Face model identifier. Defaults to MODEL_NAME.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_MODEL_NAME,
        help="OpenAI model identifier. Defaults to OPENAI_MODEL.",
    )
    return parser.parse_args()


def create_client() -> OpenAI:
    return create_hf_client()


def create_hf_client() -> OpenAI:
    token = os.getenv("HF_TOKEN") or HF_TOKEN
    if not token:
        raise ValueError("HF_TOKEN environment variable is required")
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", API_BASE_URL),
        api_key=token,
    )


def create_openai_client() -> OpenAI:
    token = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not token:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=token)


def create_clients(provider: str) -> dict[str, OpenAI]:
    clients: dict[str, OpenAI] = {}
    errors: list[str] = []

    if provider in {"hf", "hybrid"}:
        try:
            clients["hf"] = create_hf_client()
        except ValueError as exc:
            if provider == "hf":
                raise
            errors.append(str(exc))

    if provider in {"openai", "hybrid"}:
        try:
            clients["openai"] = create_openai_client()
        except ValueError as exc:
            if provider == "openai":
                raise
            errors.append(str(exc))

    if not clients:
        raise ValueError("; ".join(errors) or "No inference clients are configured")

    return clients


def resolve_model_map(args: argparse.Namespace) -> dict[str, str]:
    hf_model = args.hf_model
    openai_model = args.openai_model

    if args.model:
        if args.provider == "openai":
            openai_model = args.model
        else:
            hf_model = args.model

    return {
        "hf": hf_model,
        "openai": openai_model,
    }


def resolve_tasks(task_name: str) -> list[TaskName]:
    return list(TaskName) if task_name == "all" else [TaskName(task_name)]


def resolve_client_type(task: TaskName, provider: str, clients: dict[str, OpenAI]) -> str:
    if provider in {"hf", "openai"}:
        if provider not in clients:
            raise ValueError(f"{provider} client is not configured")
        return provider

    preferred = HYBRID_PROVIDER_BY_TASK.get(task, "hf")
    if preferred in clients:
        return preferred

    fallback = "openai" if preferred == "hf" else "hf"
    if fallback in clients:
        return fallback

    raise ValueError(f"No client available for task '{task.value}'")


def build_prompt(task: TaskName, observation: Observation) -> str:
    metadata = json.dumps(observation.metadata, sort_keys=True)
    return TASK_PROMPTS[task].format(
        content=observation.content,
        metadata=metadata,
    )


def extract_json_payload(raw_text: str) -> dict:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response did not include a JSON object")
    return json.loads(raw_text[start:end + 1])


def clamp_confidence(value, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def sanitize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def serialize_action(action: Action) -> str:
    return (
        f"task={action.task.value};"
        f"decision={sanitize_text(action.decision)};"
        f"confidence={action.confidence:.2f};"
        f"reasoning={sanitize_text(action.reasoning) or 'none'}"
    )


def format_error(error: str | None) -> str:
    return "null" if not error else sanitize_text(error)


def default_action(task: TaskName, reason: str) -> Action:
    return Action(
        task=task,
        decision=DEFAULT_DECISIONS[task],
        confidence=0.5,
        reasoning=sanitize_text(reason) or "fallback action",
    )


def empty_scorecard() -> dict[str, float | int]:
    return {
        "score": 0.0,
        "accuracy": 0.0,
        "avg_reward": 0.0,
        "calibration": 0.0,
        "correct": 0,
        "total": 0,
    }


def query_model(
    clients: dict[str, OpenAI],
    client_type: str,
    model_name: str,
    task: TaskName,
    prompt: str,
) -> Action:
    try:
        client = clients[client_type]
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content or ""
        payload = extract_json_payload(raw_text)
        return Action(
            task=task,
            decision=payload.get("decision", DEFAULT_DECISIONS[task]),
            confidence=clamp_confidence(payload.get("confidence", 0.5)),
            reasoning=sanitize_text(payload.get("reasoning", "")),
        )
    except Exception as exc:
        return default_action(task, f"{client_type}_error: {exc}")


def run_episode(
    clients: dict[str, OpenAI],
    model_map: dict[str, str],
    provider: str,
    task: TaskName,
) -> dict[str, object]:
    env = DeepfakePhishingEnv()
    rewards: list[float] = []
    steps = 0
    success = False
    scorecard = empty_scorecard()
    client_type = resolve_client_type(task, provider, clients)
    model_name = model_map[client_type]
    print(f"[START] task={task.value} env={BENCHMARK_NAME} model={model_name}")

    try:
        observation = env.reset(task)
        done = False
        while not done:
            prompt = build_prompt(task, observation)
            action = query_model(clients, client_type, model_name, task, prompt)
            observation, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward.value)
            error = info.get("last_action_error")
            print(
                f"[STEP]  step={steps} action={serialize_action(action)} "
                f"reward={reward.value:.2f} done={str(done).lower()} error={format_error(error)}"
            )
        scorecard = grade_episode(env.state()["history"], task)
        success = bool(rewards) and float(scorecard["score"]) > 0.0
    except Exception:
        success = False
    finally:
        env.close()
        reward_series = ",".join(f"{value:.2f}" for value in rewards)
        print(
            f"[END]   success={str(success).lower()} steps={steps} "
            f"rewards={reward_series} score={float(scorecard['score']):.4f} "
            f"accuracy={float(scorecard['accuracy']):.4f} avg_reward={float(scorecard['avg_reward']):.4f} "
            f"calibration={float(scorecard['calibration']):.4f}"
        )
    return {
        "task": task.value,
        "provider": client_type,
        "model": model_name,
        "success": success,
        "steps": steps,
        "rewards": rewards,
        "score": float(scorecard["score"]),
        "accuracy": float(scorecard["accuracy"]),
        "avg_reward": float(scorecard["avg_reward"]),
        "calibration": float(scorecard["calibration"]),
    }


def summarize_suite(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "success": False,
            "tasks": 0,
            "baseline_score": 0.0,
            "task_scores": "",
        }

    baseline_score = round(
        sum(float(result["score"]) for result in results) / len(results),
        4,
    )
    task_scores = ",".join(
        f"{result['task']}:{float(result['score']):.4f}" for result in results
    )
    return {
        "success": all(bool(result["success"]) for result in results),
        "tasks": len(results),
        "baseline_score": baseline_score,
        "task_scores": task_scores,
    }


def main() -> None:
    args = parse_args()
    clients = create_clients(args.provider)
    model_map = resolve_model_map(args)
    results = [run_episode(clients, model_map, args.provider, task) for task in resolve_tasks(args.task)]
    if len(results) > 1:
        summary = summarize_suite(results)
        print(
            f"[END]   success={str(summary['success']).lower()} tasks={summary['tasks']} "
            f"baseline_score={float(summary['baseline_score']):.4f} task_scores={summary['task_scores']}"
        )


if __name__ == "__main__":
    main()
