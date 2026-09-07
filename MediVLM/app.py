"""
MediVLM + AMG-RAG local interface
==================================
Upload a chest X-ray, generate a report with your trained MediVLM checkpoint, then
run it through the AMG-RAG 5-step hallucination detection pipeline -- all in one
browser UI, with the pipeline's step-by-step output shown exactly as it prints in
the notebook version.

Run from inside your cloned MediVLM repo folder:
    cd path/to/MediVLM
    pip install gradio torch pillow pandas networkx requests python-decouple langchain-google-genai langchain-classic langchain-core
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import io
import os
import sys
import inspect
import contextlib
import re
import traceback
import warnings

import torch
import pandas as pd
import gradio as gr
from PIL import Image

# ──────────────────────────────────────────────
# Make sure the MediVLM repo is importable, then the AMG-RAG module in this folder
# ──────────────────────────────────────────────
sys.path.append(os.getcwd())

# Set the Gemini API key in env BEFORE importing rag_module_source,
# because that module reads GOOGLE_API_KEY at import time via decouple.
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZCYEOgB9D0mfl7L6385kCou8M9kV1EJE"

from rag_module_source import AMG_RAG_ReportSystem  # noqa: E402

# These come from your MediVLM repo -- app.py assumes it's run from inside that repo,
# same as your existing inference notebooks.
from medivlm.data.transforms import build_image_transform  # noqa: E402
from medivlm.models import MediVLM  # noqa: E402
from medivlm.utils import load_checkpoint, load_config  # noqa: E402


# ──────────────────────────────────────────────
# Defaults -- edit these if your paths differ
# ──────────────────────────────────────────────
DEFAULT_CHECKPOINT_PATH = r"C:\Users\mohansai\OneDrive\Desktop\VLIL\MediVLM\mimic_sample.ckpt"
DEFAULT_CONFIG_FILE = "configs/mimic_cxr_sample.yaml"  # matches the "sample" checkpoint naming
DEFAULT_NUM_BEAMS = 4  # fixed, no longer user-adjustable
DEFAULT_GOOGLE_API_KEY = "AIzaSyAZCYEOgB9D0mfl7L6385kCou8M9kV1EJE"
DEFAULT_PUBMED_KEY = os.environ.get("pubmed_api", "")

# Global handles populated once the model is loaded, so we don't reload on every click.
# The model is also preloaded once at startup so the UI is ready immediately.
STATE = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model": None,
    "transform": None,
    "cfg": None,
    "rag_system": None,
}


def _startup_preload_model() -> str:
    """Load the default MediVLM checkpoint during app startup."""
    try:
        status = load_medivlm(DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_FILE)
        return status
    except Exception as e:
        return f"Startup preload failed: {e}"


def _startup_preload_rag() -> str:
    """Load the AMG-RAG system during app startup when a key is available."""
    if not DEFAULT_GOOGLE_API_KEY:
        return "Startup skipped: GOOGLE_API_KEY is not set."
    try:
        status = load_rag_system(DEFAULT_GOOGLE_API_KEY, DEFAULT_PUBMED_KEY)
        return status
    except Exception as e:
        return f"Startup AMG-RAG preload failed: {e}"


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_medivlm(checkpoint_path, config_file):
    """(Re)load the MediVLM model. Returns a status string for the UI."""
    try:
        if not os.path.exists(checkpoint_path):
            return f"Checkpoint not found at: {checkpoint_path}"

        cfg = load_config(config_file)
        model = MediVLM(cfg).to(STATE["device"])
        load_checkpoint(checkpoint_path, model=model, map_location=STATE["device"])
        model.eval()

        transform = build_image_transform(cfg.image_encoder.image_size, train=False)

        STATE["model"] = model
        STATE["transform"] = transform
        STATE["cfg"] = cfg

        return (
            f"MediVLM loaded.\n"
            f"  Config: {config_file}\n"
            f"  Checkpoint: {checkpoint_path}\n"
            f"  Device: {STATE['device']}"
        )
    except Exception as e:
        return f"Failed to load MediVLM: {e}\n\n{traceback.format_exc()}"


def load_rag_system(api_key, pubmed_key):
    """(Re)initialize the AMG-RAG system with the given API key(s)."""
    try:
        if not api_key:
            return "Enter a Google API key before loading the RAG system."
        if pubmed_key:
            os.environ["pubmed_api"] = pubmed_key
        STATE["rag_system"] = AMG_RAG_ReportSystem(google_api_key=api_key)
        return "AMG-RAG system initialized."
    except Exception as e:
        return f"Failed to initialize AMG-RAG system: {e}\n\n{traceback.format_exc()}"


# ──────────────────────────────────────────────
# Generation helper -- mirrors the safe-kwarg-introspection approach used
# earlier for the standalone inference notebook, since we don't have direct
# visibility into this repo's exact generate() signature.
# ──────────────────────────────────────────────
def call_generate_safely(model, image_tensor, num_beams, max_new_tokens=300):
    sig = inspect.signature(model.generate)
    accepted = set(sig.parameters.keys())
    candidate_kwargs = {
        "num_beams": num_beams,
        "num_return_sequences": 1,
        "max_new_tokens": max_new_tokens,
        "max_length": max_new_tokens,
        "min_new_tokens": 20,
        "min_length": 20,
        "early_stopping": False,
        "length_penalty": 1.0,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }
    used = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    skipped = [k for k in candidate_kwargs if k not in accepted]

    log_lines = [f"generate() accepts: {sorted(accepted)}", f"using kwargs: {used}"]
    if skipped:
        log_lines.append(f"skipped (not in signature): {skipped}")

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `repetition_penalty` with `inputs_embeds` and without `input_ids` to `generate`.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `no_repeat_ngram_size` with `inputs_embeds` and without `input_ids` to `generate`.*",
                category=UserWarning,
            )
            reports = model.generate(image_tensor, **used)

    report = reports[0]
    # Defensive: some generate() implementations return (batch, beams) instead
    # of (batch,) even when num_return_sequences=1 — unwrap to a plain string.
    while isinstance(report, (list, tuple)):
        report = report[0]

    return report, "\n".join(log_lines)


def _clean_display_report(report_text: str) -> str:
    """Make the displayed report easier to read without changing the model output."""
    if not report_text:
        return report_text

    # Collapse repeated blank lines and trim surrounding whitespace.
    cleaned = re.sub(r"\n{3,}", "\n\n", report_text).strip()

    # If the model repeats section headers, keep the text but make it less noisy.
    cleaned = re.sub(r"(?im)^\s*(findings?|impression)\s*:\s*", r"\1: ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


# ──────────────────────────────────────────────
# Main pipeline callback
# ──────────────────────────────────────────────
def analyze_xray(image, checkpoint_path, config_file, api_key, pubmed_key):
    if image is None:
        return "Please upload an image first.", "", "", None, ""

    if STATE["model"] is None:
        status = load_medivlm(checkpoint_path, config_file)
        if STATE["model"] is None:
            return status, "", "", None, ""

    if STATE["rag_system"] is None:
        status = load_rag_system(api_key, pubmed_key)
        if STATE["rag_system"] is None:
            return status, "", "", None, ""

    try:
        img = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        image_tensor = STATE["transform"](img).unsqueeze(0).to(STATE["device"])
        report_text, gen_log = call_generate_safely(
            STATE["model"], image_tensor, num_beams=DEFAULT_NUM_BEAMS
        )
        report_text = _clean_display_report(report_text)
    except Exception as e:
        return f"MediVLM generation failed: {e}\n\n{traceback.format_exc()}", "", "", None, ""

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = STATE["rag_system"].evaluate_medical_report(report_text)
    except Exception as e:
        pipeline_log = buf.getvalue()
        pipeline_log += f"\n\nAMG-RAG pipeline failed: {e}\n\n{traceback.format_exc()}"
        return report_text, gen_log, pipeline_log, None, ""

    pipeline_log = buf.getvalue()

    findings = result.get("detailed_findings", [])
    df = pd.DataFrame([
        {
            "Finding": f["finding"],
            "Target": f["target"],
            "Baseline": f["baseline_classification"],
            "Contradicted": f["is_contradicted"],
            "Grounding score": round(f["grounding_score"], 2),
            "Hallucination risk": f["hallucination_risk"],
        }
        for f in findings
    ])

    verdict = "HALLUCINATED" if result.get("hallucination_detected") else "NOT HALLUCINATED"
    confidence = result.get("overall_confidence", 0.0)
    summary = f"VERDICT: {verdict}\nOverall confidence: {confidence:.2f}"

    return report_text, gen_log, pipeline_log, df, summary


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
with gr.Blocks(title="MediVLM + AMG-RAG") as demo:
    gr.Markdown(
        """
        # MediVLM + AMG-RAG: X-Ray Report Generation & Hallucination Check
        Upload a chest X-ray, generate a report with your MediVLM checkpoint, and run it
        through the AMG-RAG pipeline (extraction -> baseline check -> PubMed grounding ->
        contradiction engine -> verdict) -- with the step-by-step output shown below,
        the same as the notebook version.
        """
    )

    with gr.Accordion("Model & API settings", open=True):
        gr.Markdown(
            "MediVLM is preloaded at startup. Use the reload button only if you change the checkpoint or config path."
        )
        with gr.Row():
            checkpoint_path_in = gr.Textbox(
                label="MediVLM checkpoint path",
                value=DEFAULT_CHECKPOINT_PATH,
            )
            config_file_in = gr.Textbox(
                label="Config file (relative to MediVLM repo root)",
                value=DEFAULT_CONFIG_FILE,
            )
        with gr.Row():
            api_key_in = gr.Textbox(
                label="Google API key",
                type="password",
                value=DEFAULT_GOOGLE_API_KEY,
            )
            pubmed_key_in = gr.Textbox(
                label="PubMed API key (optional)",
                type="password",
                value=os.environ.get("pubmed_api", ""),
            )
        load_status = gr.Textbox(
            label="Status",
            value="MediVLM will load automatically when the app starts.",
            interactive=False,
        )
        with gr.Row():
            load_model_btn = gr.Button("Reload MediVLM")
            load_rag_btn = gr.Button("Load / Reload AMG-RAG system")

        load_model_btn.click(
            fn=load_medivlm, inputs=[checkpoint_path_in, config_file_in], outputs=load_status
        )
        load_rag_btn.click(
            fn=load_rag_system, inputs=[api_key_in, pubmed_key_in], outputs=load_status
        )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Chest X-ray", type="pil")
            analyze_btn = gr.Button("Generate report & analyze", variant="primary")
            gr.Markdown(
                "One uploaded image produces one generated report. The AMG-RAG section below only breaks that single report into claim-level checks."
            )

        with gr.Column(scale=2):
            report_out = gr.Textbox(label="Generated report from one X-ray", lines=4)
            verdict_out = gr.Textbox(label="Final verdict", lines=2)
            findings_out = gr.Dataframe(label="Per-finding breakdown from the same report")

    with gr.Accordion("Generation log", open=False):
        gen_log_out = gr.Textbox(label="MediVLM generation details", lines=6)

    with gr.Accordion("AMG-RAG step-by-step pipeline output", open=False):
        pipeline_log_out = gr.Textbox(label="Steps 1-5", lines=30)

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[
            image_in, checkpoint_path_in, config_file_in,
            api_key_in, pubmed_key_in,
        ],
        outputs=[report_out, gen_log_out, pipeline_log_out, findings_out, verdict_out],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    print(_startup_preload_model())
    print(_startup_preload_rag())
    demo.launch()
