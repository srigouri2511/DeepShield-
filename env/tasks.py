import re
import os
from functools import lru_cache
import cv2
import numpy as np
from env.models import TaskName, Observation, Action, Reward
from env.features import URLReputation, EmailHeaderAnalyzer


# ── Shared helpers ────────────────────────────────────────────────────────────

PHISHING_KEYWORDS = [
    "verify your account", "click here", "urgent", "suspended",
    "confirm password", "login immediately", "prize", "winner",
    "bank account", "social security", "update billing",
    "unusual activity", "limited access", "act now", "free gift",
    "you have been selected", "your account has been",
    "identity verification", "mismatched sender", "return address",
]

URL_RULES = [
    (re.compile(r"(bit\.ly|tinyurl|goo\.gl|ow\.ly)", re.I), 0.3, "url_shortener"),
    (re.compile(r"@"), 0.5, "at_sign"),
    (re.compile(r"\d{1,3}(\.\d{1,3}){3}"), 0.5, "ip_address"),
    (re.compile(r"\.(tk|ml|cf|ga|gq)(/|$)", re.I), 0.4, "free_tld"),
    (re.compile(r"paypal|apple|amazon|microsoft|google", re.I), 0.4, "brand_impersonation"),
    (re.compile(r"login\.|secure\.", re.I), 0.3, "suspicious_subdomain"),
]


def _phishing_score(text: str) -> tuple[float, list[str]]:
    lower = text.lower()
    hits = [kw for kw in PHISHING_KEYWORDS if kw in lower]
    return min(1.0, len(hits) * 0.2), hits


def _url_score(url: str) -> tuple[float, list[str]]:
    hits, total = [], 0.0
    for pattern, weight, label in URL_RULES:
        if pattern.search(url):
            hits.append(label)
            total += weight
    return min(1.0, total), hits


@lru_cache(maxsize=128)
def _laplacian_score(path: str) -> float:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(1.0 - variance / 500.0, 0.0, 1.0))


@lru_cache(maxsize=128)
def _histogram_similarity(path_a: str, path_b: str) -> float:
    def load(p):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        return cv2.resize(img, (128, 128))
    a, b = load(path_a), load(path_b)
    scores = []
    for i in range(3):
        ha = cv2.calcHist([a], [i], None, [256], [0, 256])
        hb = cv2.calcHist([b], [i], None, [256], [0, 256])
        scores.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    histogram_similarity = float(np.mean(scores))
    pixel_similarity = 1.0 - float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
    hybrid_similarity = 0.3 * histogram_similarity + 0.7 * pixel_similarity
    return float(np.clip(hybrid_similarity, 0.0, 1.0))


def _sample_dir() -> str:
    target = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))
    os.makedirs(target, exist_ok=True)
    return target


def _build_sample_image(path: str, variant: int = 0, blur: bool = False) -> None:
    if os.path.exists(path):
        return
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    if variant == 0:
        cv2.rectangle(canvas, (40, 40), (216, 216), (220, 180, 80), -1)
        cv2.circle(canvas, (128, 128), 60, (70, 170, 240), -1)
    elif variant == 1:
        cv2.line(canvas, (0, 0), (256, 256), (220, 110, 190), 14)
        cv2.line(canvas, (256, 0), (0, 256), (120, 210, 100), 14)
    else:
        for i in range(0, 256, 16):
            color = ((i * 3) % 255, (255 - i) % 255, (i * 7) % 255)
            cv2.line(canvas, (i, 0), (i, 255), color, 10)
    if blur:
        canvas = cv2.GaussianBlur(canvas, (17, 17), 10)
    cv2.imwrite(path, canvas)


# ── Task 1: Phishing Triage (Easy) ────────────────────────────────────────────

PHISHING_EMAIL_SAMPLES = [
    ("urgent! verify your account now or it will be suspended. click here.", True),
    ("Hi, your order has been shipped and will arrive in 3 days.", False),
    ("You have been selected as a winner! claim your free gift now.", True),
    ("Your invoice #4821 is attached. Please review at your convenience.", False),
    ("Unusual activity detected. Confirm password immediately to avoid limited access.", True),
    ("Team meeting rescheduled to 3pm tomorrow. Please update your calendar.", False),
]

URL_SAMPLES = [
    ("http://bit.ly/free-prize-claim", True),
    ("https://www.google.com/search?q=weather", False),
    ("http://192.168.1.1/login.php", True),
    ("https://github.com/openai/openai-python", False),
    ("http://paypal-secure.tk/verify@user", True),
    ("https://docs.python.org/3/library/re.html", False),
]

HEADER_SAMPLES = [
    ("From: support@paypal.com\nReply-To: support@paypal.com\nDKIM=pass SPF=pass DMARC=pass\n", False),
    ("From: security@bank.com\nReply-To: alert@bank-security.net\nDKIM=fail SPF=fail DMARC=fail\n", True),
    ("From: service@amazon.com\nReply-To: service@amazon.com\nX-Mailer: bulk mailer\n", True),
    ("From: friend@example.com\nReply-To: friend@example.com\nReceived: from 192.168.0.2 by mail.example.com\n", False),
]


class PhishingTriageTask:
    """Easy — classify email text as phishing or legitimate."""
    name = TaskName.PHISHING_TRIAGE
    VALID_DECISIONS = {"phishing", "legitimate"}

    def __init__(self):
        self._samples = PHISHING_EMAIL_SAMPLES
        self._idx = 0

    def reset(self) -> Observation:
        self._idx = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        content, ground_truth = self._samples[self._idx]
        reward = self._grade(action, ground_truth, content)
        self._idx += 1
        done = self._idx >= len(self._samples)
        obs = self._observe() if not done else Observation(
            task=self.name, step=self._idx, content="", done=True
        )
        return obs, reward, done, {"ground_truth": ground_truth, "sample": content}

    def state(self) -> dict:
        return {"task": self.name, "step": self._idx, "total": len(self._samples)}

    def _observe(self) -> Observation:
        content, _ = self._samples[self._idx]
        return Observation(task=self.name, step=self._idx, content=content)

    def _grade(self, action: Action, ground_truth: bool, content: str) -> Reward:
        if action.decision not in self.VALID_DECISIONS:
            return Reward(value=0.0, correct=False, feedback="Invalid decision label.")
        predicted = action.decision == "phishing"
        correct = predicted == ground_truth

        if correct:
            value = 0.5 + 0.5 * action.confidence
        else:
            value = max(0.0, 0.3 - 0.3 * action.confidence)

        return Reward(value=round(value, 4), correct=correct,
                      feedback="Correct." if correct else f"Wrong. Expected {'phishing' if ground_truth else 'legitimate'}.")


class URLReputationTask:
    """Easy — classify URL reputation as suspicious or clean."""
    name = TaskName.URL_REPUTATION
    VALID_DECISIONS = {"suspicious", "clean"}

    def __init__(self):
        self._samples = URL_SAMPLES
        self._analyzer = URLReputation()
        self._analysis_cache: dict[str, dict] = {}
        self._idx = 0

    def reset(self) -> Observation:
        self._idx = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        url, ground_truth = self._samples[self._idx]
        result = self._analyze(url)
        reward = self._grade(action, result)
        self._idx += 1
        done = self._idx >= len(self._samples)
        obs = self._observe() if not done else Observation(task=self.name, step=self._idx, content="", done=True)
        return obs, reward, done, {"ground_truth": ground_truth, "details": result}

    def state(self) -> dict:
        return {"task": self.name, "step": self._idx, "total": len(self._samples)}

    def _observe(self) -> Observation:
        url, _ = self._samples[self._idx]
        result = self._analyze(url)
        return Observation(task=self.name, step=self._idx, content=url, metadata=result)

    def _analyze(self, url: str) -> dict:
        if url not in self._analysis_cache:
            self._analysis_cache[url] = self._analyzer.analyze(url)
        return self._analysis_cache[url]

    def _grade(self, action: Action, result: dict) -> Reward:
        if action.decision not in self.VALID_DECISIONS:
            return Reward(value=0.0, correct=False, feedback="Invalid decision label.")
        predicted = action.decision == "suspicious"
        correct = predicted == result["is_suspicious"]
        base = result["confidence"]
        if correct:
            value = 0.4 + 0.4 * action.confidence + 0.2 * base
        else:
            value = max(0.0, 0.25 * (1.0 - base) - 0.25 * action.confidence)
        return Reward(value=round(value, 4), correct=correct,
                      feedback="Correct." if correct else f"Wrong. Expected {'suspicious' if result['is_suspicious'] else 'clean'}.")


class EmailHeaderAnalysisTask:
    """Medium — analyze raw email headers for authentication and spoofing issues."""
    name = TaskName.EMAIL_HEADER_ANALYSIS
    VALID_DECISIONS = {"suspicious", "clean"}

    def __init__(self):
        self._samples = HEADER_SAMPLES
        self._analyzer = EmailHeaderAnalyzer()
        self._analysis_cache: dict[str, dict] = {}
        self._idx = 0

    def reset(self) -> Observation:
        self._idx = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        headers, ground_truth = self._samples[self._idx]
        result = self._analyze(headers)
        reward = self._grade(action, result)
        self._idx += 1
        done = self._idx >= len(self._samples)
        obs = self._observe() if not done else Observation(task=self.name, step=self._idx, content="", done=True)
        return obs, reward, done, {"ground_truth": ground_truth, "details": result}

    def state(self) -> dict:
        return {"task": self.name, "step": self._idx, "total": len(self._samples)}

    def _observe(self) -> Observation:
        headers, _ = self._samples[self._idx]
        result = self._analyze(headers)
        return Observation(task=self.name, step=self._idx, content=headers, metadata=result)

    def _analyze(self, headers: str) -> dict:
        if headers not in self._analysis_cache:
            self._analysis_cache[headers] = self._analyzer.analyze(headers)
        return self._analysis_cache[headers]

    def _grade(self, action: Action, result: dict) -> Reward:
        if action.decision not in self.VALID_DECISIONS:
            return Reward(value=0.0, correct=False, feedback="Invalid decision label.")
        predicted = action.decision == "suspicious"
        correct = predicted == result["is_suspicious"]
        confidence = result["confidence"]
        if correct:
            value = 0.45 + 0.35 * action.confidence + 0.2 * confidence
        else:
            value = max(0.0, 0.2 * (1.0 - confidence) - 0.2 * action.confidence)
        return Reward(value=round(value, 4), correct=correct,
                      feedback="Correct." if correct else f"Wrong. Expected {'suspicious' if result['is_suspicious'] else 'clean'}.")


# ── Task 2: Deepfake Detection (Medium) ───────────────────────────────────────

class DeepfakeDetectionTask:
    """Medium — analyze image/video path and classify as deepfake or real."""
    name = TaskName.DEEPFAKE_DETECTION
    VALID_DECISIONS = {"deepfake", "real"}

    def __init__(self, samples: list[tuple[str, bool]] = None):
        self._samples = samples or self._create_default_samples()
        self._idx = 0

    def _create_default_samples(self) -> list[tuple[str, bool]]:
        base = _sample_dir()
        real_path = os.path.join(base, "real_sample.png")
        fake_path = os.path.join(base, "deepfake_sample.png")
        _build_sample_image(real_path, variant=0, blur=False)
        _build_sample_image(fake_path, variant=1, blur=True)
        return [(real_path, False), (fake_path, True)]

    def reset(self) -> Observation:
        self._idx = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        path, ground_truth = self._samples[self._idx]
        heuristic_score = _laplacian_score(path)
        reward = self._grade(action, ground_truth, heuristic_score)
        self._idx += 1
        done = self._idx >= len(self._samples)
        obs = self._observe() if not done else Observation(
            task=self.name, step=self._idx, content="", done=True
        )
        return obs, reward, done, {"ground_truth": ground_truth, "heuristic_score": heuristic_score}

    def state(self) -> dict:
        return {"task": self.name, "step": self._idx, "total": len(self._samples)}

    def _observe(self) -> Observation:
        path, _ = self._samples[self._idx]
        return Observation(
            task=self.name, step=self._idx, content=path,
            metadata={"heuristic_score": round(_laplacian_score(path), 4)}
        )

    def _grade(self, action: Action, ground_truth: bool, heuristic: float) -> Reward:
        if action.decision not in self.VALID_DECISIONS:
            return Reward(value=0.0, correct=False, feedback="Invalid decision label.")
        predicted = action.decision == "deepfake"
        correct = predicted == ground_truth
        # reward alignment with heuristic even if wrong label
        heuristic_alignment = heuristic if ground_truth else (1.0 - heuristic)
        if correct:
            value = 0.4 + 0.4 * action.confidence + 0.2 * heuristic_alignment
        else:
            value = max(0.0, 0.2 * heuristic_alignment - 0.2 * action.confidence)
        return Reward(value=round(value, 4), correct=correct,
                      feedback="Correct." if correct else f"Wrong. Expected {'deepfake' if ground_truth else 'real'}.")


# ── Task 3: Identity Fraud (Hard) ─────────────────────────────────────────────

class IdentityFraudTask:
    """Hard — compare two face images + email context to detect identity fraud."""
    name = TaskName.IDENTITY_FRAUD
    VALID_DECISIONS = {"fraud", "legitimate"}

    def __init__(self, samples: list[tuple[str, str, str, bool]] = None):
        self._samples = samples or self._create_default_samples()
        self._idx = 0

    def _create_default_samples(self) -> list[tuple[str, str, str, bool]]:
        base = _sample_dir()
        same_a = os.path.join(base, "identity_legit_a.png")
        same_b = os.path.join(base, "identity_legit_b.png")
        diff_a = os.path.join(base, "identity_fraud_a.png")
        diff_b = os.path.join(base, "identity_fraud_b.png")
        _build_sample_image(same_a, variant=0, blur=False)
        _build_sample_image(same_b, variant=0, blur=False)
        _build_sample_image(diff_a, variant=1, blur=False)
        _build_sample_image(diff_b, variant=2, blur=False)
        return [
            (same_a, same_b, "Email indicates a new device login request from a known contact.", False),
            (diff_a, diff_b, "Urgent identity verification request with mismatched sender and return address.", True),
        ]

    def reset(self) -> Observation:
        self._idx = 0
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        img_a, img_b, email, ground_truth = self._samples[self._idx]
        sim = _histogram_similarity(img_a, img_b)
        email_score, _ = _phishing_score(email)
        reward = self._grade(action, ground_truth, sim, email_score)
        self._idx += 1
        done = self._idx >= len(self._samples)
        obs = self._observe() if not done else Observation(
            task=self.name, step=self._idx, content="", done=True
        )
        return obs, reward, done, {
            "ground_truth": ground_truth,
            "face_similarity": sim,
            "email_risk": email_score,
        }

    def state(self) -> dict:
        return {"task": self.name, "step": self._idx, "total": len(self._samples)}

    def _observe(self) -> Observation:
        img_a, img_b, email, _ = self._samples[self._idx]
        sim = round(_histogram_similarity(img_a, img_b), 4)
        email_risk, _ = _phishing_score(email)
        return Observation(
            task=self.name, step=self._idx,
            content=f"Email: {email}",
            metadata={
                "image_a": img_a,
                "image_b": img_b,
                "face_similarity": sim,
                "email_risk": round(email_risk, 4),
            },
        )

    def _grade(self, action: Action, ground_truth: bool, sim: float, email_risk: float) -> Reward:
        if action.decision not in self.VALID_DECISIONS:
            return Reward(value=0.0, correct=False, feedback="Invalid decision label.")
        predicted = action.decision == "fraud"
        correct = predicted == ground_truth
        # multi-signal reward: face dissimilarity + email risk both matter
        signal = (1.0 - sim) * 0.5 + email_risk * 0.5 if ground_truth else sim * 0.5 + (1.0 - email_risk) * 0.5
        if correct:
            value = 0.3 + 0.4 * action.confidence + 0.3 * signal
        else:
            value = max(0.0, 0.2 * signal - 0.3 * action.confidence)
        return Reward(value=round(value, 4), correct=correct,
                      feedback="Correct." if correct else f"Wrong. Expected {'fraud' if ground_truth else 'legitimate'}.")
