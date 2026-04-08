# DeepShield: OpenEnv Security Benchmark

DeepShield is a real-world OpenEnv environment for security analysis tasks that humans actually perform:

- phishing email triage
- URL reputation analysis
- email header analysis
- deepfake media review
- identity fraud verification

The project includes an OpenEnv-compatible environment, deterministic task graders, a Hugging Face Space-ready Flask dashboard, and an `inference.py` baseline runner that uses the OpenAI client with an OpenAI-compatible endpoint.

## Why This Environment

Security teams regularly make multi-step classification decisions under uncertainty. This benchmark turns those real workflows into RL-friendly tasks with typed observations, typed actions, per-step rewards, and deterministic grading. It is designed for evaluating agent behavior on practical analyst work instead of toy puzzles.

## OpenEnv Interface

The environment implementation is in `env/environment.py` and exposes:

- `reset(task: TaskName = phishing_triage) -> Observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> dict`
- `close() -> None`

Typed Pydantic models are defined in `env/models.py`:

- `Observation`
- `Action`
- `Reward`

The benchmark metadata file is `openenv.yaml`.

## Observation Space

```text
Observation:
  task: TaskName
  step: int
  content: str
  metadata: dict[str, Any]
  done: bool
```

`content` carries the raw analyst input such as an email body, URL, header block, image path, or fraud context. `metadata` carries supporting signals such as heuristic scores, parsed flags, face similarity, and email risk.

## Action Space

```text
Action:
  task: TaskName
  decision: str
  confidence: float
  reasoning: str | None
```

Valid decisions by task:

- `phishing_triage`: `phishing` or `legitimate`
- `url_reputation`: `suspicious` or `clean`
- `email_header_analysis`: `suspicious` or `clean`
- `deepfake_detection`: `deepfake` or `real`
- `identity_fraud`: `fraud` or `legitimate`

## Tasks

The environment ships with five tasks, which exceeds the minimum three-task requirement.

| Task | Difficulty | Objective | Grader style |
|---|---|---|---|
| `phishing_triage` | easy | classify realistic email text | deterministic label and confidence reward |
| `url_reputation` | easy | classify URLs as suspicious or clean | deterministic heuristic-assisted reward |
| `email_header_analysis` | medium | detect spoofing and auth failures in headers | deterministic header-signal reward |
| `deepfake_detection` | medium | classify image samples as deepfake or real | deterministic heuristic-alignment reward |
| `identity_fraud` | hard | combine face similarity and email context | deterministic multi-signal fraud reward |

## Reward Function

Reward is computed on every `step(...)`, not only at the end of the episode.

Where reward lives:

- `env/tasks.py`
- `env/environment.py`
- `env/graders.py`

How reward works:

- Each task has its own `_grade(...)` method in `env/tasks.py`.
- Correct decisions earn more reward as confidence increases.
- Wrong decisions lose reward, especially when confidence is high.
- URL, header, deepfake, and identity tasks also reward alignment with meaningful intermediate signals such as URL risk, header confidence, image heuristics, face similarity, and email risk.
- `env/environment.py` adds a loop penalty when the same decision repeats too many times in a row.
- `env/graders.py` computes the final deterministic episode score from step rewards, accuracy, and calibration.

If you want the exact reward code paths, start here:

- `PhishingTriageTask._grade`
- `URLReputationTask._grade`
- `EmailHeaderAnalysisTask._grade`
- `DeepfakeDetectionTask._grade`
- `IdentityFraudTask._grade`
- `DeepfakePhishingEnv.step`
- `grade_episode`

## Setup

```bash
pip install -r requirements.txt
```

For local secrets, copy `.env.example` to `.env.local` and add your own keys there.
The app, dashboard AI copilot, and inference runner automatically load `.env.local` if present.

## Local Usage

### Gradio app

You can run a simple Hugging Face Space-style UI locally with:

```bash
python app.py
```

This launches the Gradio app on `http://127.0.0.1:7860` by default.

### Web dashboard

```bash
python api.py
```

Then open `http://127.0.0.1:5000`.

Optional dashboard AI mode:

```bash
OPENAI_API_KEY=<your_openai_key> python api.py
```

The dashboard remains usable without `OPENAI_API_KEY`; it falls back to deterministic heuristics.
The dashboard AI copilot reads `OPENAI_API_KEY` from the environment in `ai_assistant.py`.
You can also place `OPENAI_API_KEY` in `.env.local` for local development.

### OpenEnv baseline inference

`inference.py` lives in the project root as required by the hackathon rules.

Required environment variables:

- `HF_TOKEN`: required for `hf` provider, optional fallback client for `hybrid`
- `OPENAI_API_KEY`: required for `openai` provider, optional fallback client for `hybrid`
- `API_BASE_URL`: optional, defaults to `https://router.huggingface.co/v1`
- `MODEL_NAME`: optional, defaults to `meta-llama/Llama-3.1-8B-Instruct`
- `OPENAI_MODEL`: optional, defaults to `gpt-4o-mini`
- `MODEL_PROVIDER`: optional, one of `hf`, `openai`, or `hybrid`; defaults to `hybrid`

Backend strategy:

- `hf`: use Hugging Face for every task
- `openai`: use OpenAI for every task
- `hybrid`: use OpenAI for higher-reasoning tasks such as phishing, header analysis, and identity fraud, and Hugging Face for URL and deepfake analysis when available

Examples:

```bash
HF_TOKEN=<your_token> python inference.py --task all
OPENAI_API_KEY=<your_openai_key> python inference.py --provider openai --task phishing_triage
HF_TOKEN=<your_token> OPENAI_API_KEY=<your_openai_key> MODEL_PROVIDER=hybrid python inference.py --task all
HF_TOKEN=<your_token> python inference.py --task phishing_triage
HF_TOKEN=<your_token> MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python inference.py --task identity_fraud
```

The script prints only the required line types:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn> score=<0.0000> accuracy=<0.0000> avg_reward=<0.0000> calibration=<0.0000>
[END] success=<true|false> tasks=<n> baseline_score=<0.0000> task_scores=<task_a:score,...>
```

When you run `--task all`, the final `[END]` line reports the aggregate baseline score across all tasks.
For the hybrid path, add your OpenAI key before running:

```bash
export OPENAI_API_KEY="your_openai_key"
```

Or store it locally in `.env.local`:

```text
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
MODEL_PROVIDER=hybrid
```

## GitHub

Push the project to GitHub with:

```bash
git init
git add .
git commit -m "Initial DeepShield commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Secrets such as `.env.local` are ignored by `.gitignore` and should not be committed.

## Validation

If the OpenEnv CLI is installed in your environment, run:

```bash
openenv validate
```

## Baseline Scores

Current deterministic heuristic reference scores in `openenv.yaml`:

| Task | Score |
|---|---|
| `phishing_triage` | 0.9750 |
| `url_reputation` | 0.9825 |
| `email_header_analysis` | 0.9876 |
| `deepfake_detection` | 0.9996 |
| `identity_fraud` | 0.9185 |

These come from the built-in heuristic controller and are locally reproducible. You can add hosted-model baseline numbers later by running `inference.py` with a live `HF_TOKEN`.

## Docker

```bash
docker build -t deepshield .
docker run -p 7860:7860 deepshield
```

The container sets `PORT=7860` and starts the Gradio app from `app.py` on port `7860`.

If you want AI-enabled runs inside Docker, pass secrets with an env file:

```bash
docker run --env-file .env.local -p 7860:7860 deepshield
```

If you want the old Flask dashboard instead of Gradio, run it explicitly:

```bash
docker run --env-file .env.local -p 7860:7860 deepshield python api.py
```

## Hugging Face Spaces

This repo is set up to be deployable as a Docker Space:

- includes a `Dockerfile`
- exposes port `7860`
- includes the `openenv` tag in `openenv.yaml`
- starts the Gradio app from `app.py`
- keeps runtime dependencies lightweight enough for CPU-only execution

Recommended Space secrets:

- `HF_TOKEN` for baseline inference usage
- `OPENAI_API_KEY` only if you want dashboard AI copilot mode

### Gradio Space option

This project also includes a Gradio entrypoint in `app.py`, so you can deploy it as a standard Gradio Space:

1. Create a new Space on Hugging Face
2. Choose `Gradio` as the SDK
3. Connect the GitHub repository or upload the project files
4. Add secrets in Space settings if needed:
   - `OPENAI_API_KEY`
   - `HF_TOKEN`
   - `MODEL_PROVIDER`
5. The Space will start from `app.py`

## Project Layout

```text
openenv-deepfake-phishing/
  env/
    models.py
    environment.py
    tasks.py
    graders.py
    features.py
    identity.py
  static/
    css/dashboard.css
    js/dashboard.js
  templates/
    dashboard.html
  ai_assistant.py
  api.py
  inference.py
  openenv.yaml
  Dockerfile
  requirements.txt
  README.md
```
