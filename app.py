import json
import os

import gradio as gr

from api import AI_ENGINE, finalize_detection
from env.identity import IdentityVerifier
from env.models import TaskName
from local_env import load_local_env

load_local_env()


def summarize_result(data: dict) -> str:
    lines = [
        f"### Verdict: {str(data.get('label', 'unknown')).replace('_', ' ').title()}",
        f"- Confidence: {float(data.get('confidence', 0.0)):.2%}",
        f"- Mode: {data.get('analysis_mode', 'heuristic')}",
    ]

    if "risk_score" in data:
        lines.append(f"- Risk score: {float(data.get('risk_score', 0.0)):.2%}")

    if "similarity" in data:
        lines.append(f"- Similarity: {float(data.get('similarity', 0.0)):.2%}")

    summary = data.get("summary")
    reasoning = data.get("reasoning")
    evidence = data.get("evidence") or data.get("flags") or []
    actions = data.get("recommended_actions") or []

    if summary:
        lines.extend(["", summary])

    if reasoning:
        lines.extend(["", f"Reasoning: {reasoning}"])

    if evidence:
        lines.extend(["", "Key evidence:"])
        lines.extend([f"- {item}" for item in evidence[:4]])

    if actions:
        lines.extend(["", "Recommended actions:"])
        lines.extend([f"- {item}" for item in actions[:4]])

    return "\n".join(lines)


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2)


def run_url(url: str, use_ai: bool) -> tuple[str, str]:
    if not url.strip():
        raise gr.Error("Enter a URL first.")
    result = finalize_detection(TaskName.URL_REPUTATION, url.strip(), use_ai=use_ai)
    return summarize_result(result), render_json(result)


def run_email(content: str, use_ai: bool) -> tuple[str, str]:
    if not content.strip():
        raise gr.Error("Paste email content first.")
    result = finalize_detection(TaskName.PHISHING_TRIAGE, content.strip(), use_ai=use_ai)
    return summarize_result(result), render_json(result)


def run_headers(headers: str, use_ai: bool) -> tuple[str, str]:
    if not headers.strip():
        raise gr.Error("Paste email headers first.")
    result = finalize_detection(TaskName.EMAIL_HEADER_ANALYSIS, headers.strip(), use_ai=use_ai)
    return summarize_result(result), render_json(result)


def run_image(image_path: str | None, use_ai: bool) -> tuple[str, str]:
    if not image_path:
        raise gr.Error("Upload an image first.")
    result = finalize_detection(TaskName.DEEPFAKE_DETECTION, image_path, use_ai=use_ai)
    return summarize_result(result), render_json(result)


def run_identity(image_a: str | None, image_b: str | None, context: str) -> tuple[str, str]:
    if not image_a or not image_b:
        raise gr.Error("Upload both images first.")

    result = IdentityVerifier().verify(image_a, image_b)
    result.update({
        "analysis_mode": "heuristic",
        "summary": (
            "The uploaded images appear to represent the same person."
            if result["match"]
            else "The uploaded images appear to represent different people."
        ),
        "reasoning": "This decision is based on the similarity score between the uploaded images.",
        "context": context.strip(),
        "recommended_actions": (
            ["Continue with standard verification controls."]
            if result["match"]
            else [
                "Pause the identity workflow until the user is verified out of band.",
                "Request stronger proof of identity or a live verification step.",
            ]
        ),
    })
    return summarize_result(result), render_json(result)


def status_text() -> str:
    mode = "AI Copilot available" if AI_ENGINE.enabled else "Heuristic mode"
    model = AI_ENGINE.model if AI_ENGINE.enabled else "No OpenAI key detected"
    return f"{mode} | {model}"


with gr.Blocks(title="DeepShield", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DeepShield")
    gr.Markdown(
        "Simple security review workspace for phishing, URL, header, deepfake, and identity checks."
    )
    gr.Markdown(status_text())

    with gr.Row():
        use_ai = gr.Checkbox(
            label="Use AI copilot when available",
            value=AI_ENGINE.enabled,
            interactive=AI_ENGINE.enabled,
        )

    with gr.Tab("URL"):
        url_input = gr.Textbox(label="Target URL", placeholder="https://example.com/login")
        with gr.Row():
            url_run = gr.Button("Analyze URL", variant="primary")
        url_summary = gr.Markdown()
        url_raw = gr.Code(label="Raw result", language="json")
        gr.Examples(
            examples=[
                ["http://paypal-secure.tk/verify@user"],
                ["https://docs.python.org/3/library/re.html"],
            ],
            inputs=url_input,
        )
        url_run.click(run_url, inputs=[url_input, use_ai], outputs=[url_summary, url_raw])

    with gr.Tab("Email"):
        email_input = gr.Textbox(label="Email body", lines=8, placeholder="Paste suspicious email content")
        email_run = gr.Button("Analyze Email", variant="primary")
        email_summary = gr.Markdown()
        email_raw = gr.Code(label="Raw result", language="json")
        gr.Examples(
            examples=[
                ["Urgent: verify your account now and click here to avoid suspension."],
                ["Hi team, the weekly report is attached for your review."],
            ],
            inputs=email_input,
        )
        email_run.click(run_email, inputs=[email_input, use_ai], outputs=[email_summary, email_raw])

    with gr.Tab("Headers"):
        headers_input = gr.Textbox(label="Email headers", lines=10, placeholder="Paste raw SMTP headers")
        headers_run = gr.Button("Analyze Headers", variant="primary")
        headers_summary = gr.Markdown()
        headers_raw = gr.Code(label="Raw result", language="json")
        gr.Examples(
            examples=[
                ["From: security@bank.com\nReply-To: alert@bank-security.net\nDKIM=fail SPF=fail DMARC=fail"],
                ["From: support@paypal.com\nReply-To: support@paypal.com\nDKIM=pass SPF=pass DMARC=pass"],
            ],
            inputs=headers_input,
        )
        headers_run.click(run_headers, inputs=[headers_input, use_ai], outputs=[headers_summary, headers_raw])

    with gr.Tab("Deepfake"):
        image_input = gr.Image(label="Upload image", type="filepath")
        image_run = gr.Button("Analyze Image", variant="primary")
        image_summary = gr.Markdown()
        image_raw = gr.Code(label="Raw result", language="json")
        image_run.click(run_image, inputs=[image_input, use_ai], outputs=[image_summary, image_raw])

    with gr.Tab("Identity"):
        with gr.Row():
            image_a = gr.Image(label="Face image A", type="filepath")
            image_b = gr.Image(label="Face image B", type="filepath")
        identity_context = gr.Textbox(label="Case notes", lines=4, placeholder="Optional investigation notes")
        identity_run = gr.Button("Compare Faces", variant="primary")
        identity_summary = gr.Markdown()
        identity_raw = gr.Code(label="Raw result", language="json")
        identity_run.click(
            run_identity,
            inputs=[image_a, image_b, identity_context],
            outputs=[identity_summary, identity_raw],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
