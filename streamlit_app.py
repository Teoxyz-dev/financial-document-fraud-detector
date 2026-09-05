"""Streamlit web interface for the fraud-detection pipeline.

Academic portfolio prototype; not for production fraud adjudication.

Real-time document upload + multimodal vision analysis + multi-agent
assessment + composite risk scoring. Mirrors the logic in the notebook
(fraud_detector.ipynb) so screenshots, demos, and presentation runs
do not require launching Jupyter.

Usage:
    streamlit run streamlit_app.py

Environment:
    GEMINI_API_KEY      required for the Gemma vision/text channels
    OPENROUTER_API_KEY  optional; enables the Nemotron vision channel
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from datetime import datetime, date

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

import google.generativeai as genai
from openai import OpenAI


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

GEMMA_MODEL = "gemma-4-31b-it"
NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"


def _gemini_key() -> str:
    """Resolve the Gemini key from sidebar input first, env var second."""
    return (st.session_state.get("gemini_key") or "").strip() \
        or os.environ.get("GEMINI_API_KEY", "")


def _openrouter_key() -> str:
    return (st.session_state.get("openrouter_key") or "").strip() \
        or os.environ.get("OPENROUTER_API_KEY", "")


def _ensure_gemini() -> bool:
    """Configure the genai client with the current key. Returns True if configured."""
    key = _gemini_key()
    if not key:
        return False
    genai.configure(api_key=key)
    return True


def _openrouter_client() -> OpenAI | None:
    key = _openrouter_key()
    if not key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


IMAGE_PROMPT = """You are a forensic document examiner. Analyze the attached financial document image
(receipt, invoice, or bank statement) for visual fraud indicators.

Check for: inconsistent fonts/typography, misaligned text, pixelation around numbers,
altered totals, mismatched logos, suspicious layouts, copy-paste artifacts, unusual whitespace.

Respond ONLY with valid JSON in this exact schema:
{
  "document_type": "receipt|invoice|bank_statement|unknown",
  "fraud_indicators": ["..."],
  "visual_confidence": 0.0,
  "fraud_risk_score": 0,
  "summary": "one short sentence"
}
where visual_confidence is 0.0-1.0 and fraud_risk_score is 0-100."""


EXPERT_PROMPTS = {
    "forensic_accountant": (
        "You are a senior forensic accountant with 20 years of experience investigating financial fraud. "
        "Focus on: financial inconsistencies, unusual amounts, accounting red flags, vendor patterns, "
        "transaction timing. Be precise and cite which fields drove your conclusion."
    ),
    "compliance_officer": (
        "You are a compliance officer specializing in AML and KYC. "
        "Focus on: regulatory red flags, structuring, duplicate transactions, suspicious vendor relationships, "
        "documentation gaps. Frame your reasoning in terms of policy violations."
    ),
    "risk_analyst": (
        "You are a quantitative risk analyst. "
        "Focus on: statistical anomalies, outlier detection, base-rate reasoning, combined-signal risk. "
        "Reason about how many weak signals stack, not just any single one."
    ),
}

AGENT_TEMPLATE = """{system}

Document under review:
{doc}

Respond ONLY with valid JSON:
{{
  "risk_score": <int 1-10>,
  "verdict": "low|medium|high",
  "key_concerns": ["..."],
  "rationale": "2-3 sentences"
}}"""


WEIGHTS = {"image": 0.30, "metadata": 0.40, "agents": 0.30}
REVIEW_THRESHOLD = 35
REJECT_THRESHOLD = 65


# ---------------------------------------------------------------------------
# Pipeline functions (kept in sync with notebook cells 5, 11, 13)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {"_raw": text, "_parse_error": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"_raw": text, "_parse_error": True}


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _call_with_retry(fn, *, attempts: int = 3, backoff: tuple[float, ...] = (3.0, 8.0)):
    """Call `fn()` with retries on transient API errors.

    Retries on 5xx server errors, 429 rate limits, and timeouts — anything that's
    plausibly transient. Permanent errors (401 invalid key, 400 bad request) raise
    on the first attempt without burning the retry budget.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # broad on purpose; classify by message + type name
            last_exc = exc
            name = type(exc).__name__
            msg = str(exc).lower()
            transient = (
                name in {"InternalServerError", "ServiceUnavailable", "DeadlineExceeded",
                         "ResourceExhausted", "TooManyRequests", "APIConnectionError",
                         "APITimeoutError"}
                or any(s in msg for s in ("500", "502", "503", "504", "timeout",
                                          "rate limit", "temporarily"))
            )
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
    raise last_exc  # unreachable but keeps mypy/linters happy


def analyze_image_gemma(image: Image.Image) -> dict:
    if not _ensure_gemini():
        return {"error": "Gemini API key not provided", "fraud_risk_score": 0}
    try:
        model = genai.GenerativeModel(GEMMA_MODEL)
        response = _call_with_retry(lambda: model.generate_content([IMAGE_PROMPT, image]))
        out = _extract_json(response.text)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "fraud_risk_score": 0, "model": GEMMA_MODEL}
    out["model"] = GEMMA_MODEL
    return out


def analyze_image_nemotron(image: Image.Image) -> dict:
    client = _openrouter_client()
    if client is None:
        return {"error": "OpenRouter API key not provided", "fraud_risk_score": 0}
    try:
        response = _call_with_retry(lambda: client.chat.completions.create(
            model=NEMOTRON_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }],
        ))
        out = _extract_json(response.choices[0].message.content or "")
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "fraud_risk_score": 0, "model": NEMOTRON_MODEL}
    out["model"] = NEMOTRON_MODEL
    return out


def _doc_to_text(doc: dict) -> str:
    return (
        f"ID: {doc.get('document_id', 'N/A')}\n"
        f"Type: {doc.get('document_type')}\n"
        f"Amount: {doc.get('amount')}\n"
        f"Date: {doc.get('date_created')}\n"
        f"Vendor: {doc.get('vendor_name')}\n"
        f"Weekend transaction: {doc.get('is_weekend_transaction')}\n"
        f"Rounded amount: {doc.get('amount_rounded')}\n"
        f"Duplicate vendor same day: {doc.get('duplicate_vendor_same_day')}"
    )


def get_expert_opinion(doc: dict, expert_type: str) -> dict:
    if not _ensure_gemini():
        return {"expert_type": expert_type,
                "error": "Gemini API key not provided",
                "risk_score": 0}
    try:
        prompt = AGENT_TEMPLATE.format(system=EXPERT_PROMPTS[expert_type], doc=_doc_to_text(doc))
        model = genai.GenerativeModel(GEMMA_MODEL)
        response = _call_with_retry(lambda: model.generate_content(prompt))
        out = _extract_json(response.text)
    except Exception as exc:
        return {"expert_type": expert_type,
                "error": f"{type(exc).__name__}: {exc}",
                "risk_score": 0}
    out["expert_type"] = expert_type
    return out


def multi_agent_assessment(doc: dict, sleep_s: float = 7.0,
                           progress_cb=None) -> dict:
    opinions = []
    for i, expert in enumerate(EXPERT_PROMPTS):
        if progress_cb:
            progress_cb(i, expert)
        try:
            opinions.append(get_expert_opinion(doc, expert))
        except Exception as exc:
            opinions.append({"expert_type": expert, "error": str(exc), "risk_score": 0})
        if i < len(EXPERT_PROMPTS) - 1:
            time.sleep(sleep_s)

    scores = [float(o.get("risk_score", 0) or 0) for o in opinions]
    avg = float(np.mean(scores)) if scores else 0.0
    mx = float(np.max(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    consensus = "agree" if std < 1.5 else "disagree"
    w = min(std / 3.0, 0.5)
    aggregated = (1 - w) * avg + w * mx
    return {
        "opinions": opinions,
        "avg_risk_score": aggregated,
        "mean_risk_score": avg,
        "max_risk_score": mx,
        "score_disagreement": std,
        "consensus": consensus,
    }


def _amplify_metadata(prob: float) -> float:
    import math
    return 100.0 / (1.0 + math.exp(-8.0 * (prob - 0.4)))


def integrate(image_analysis, metadata_pred, agent_assessment) -> dict:
    image_score = float(image_analysis.get("fraud_risk_score", 0)) if image_analysis else 0.0
    raw_meta = float(metadata_pred.get("fraud_probability", 0)) if metadata_pred else 0.0
    metadata_score = _amplify_metadata(raw_meta) if metadata_pred else 0.0
    agent_score = (float(agent_assessment.get("avg_risk_score", 0)) * 10) if agent_assessment else 0.0

    active = {}
    if image_analysis is not None and "error" not in image_analysis:
        active["image"] = image_score
    if metadata_pred is not None:
        active["metadata"] = metadata_score
    if agent_assessment is not None:
        active["agents"] = agent_score
    total_w = sum(WEIGHTS[k] for k in active) or 1.0
    composite = sum(active[k] * WEIGHTS[k] for k in active) / total_w

    if composite >= REJECT_THRESHOLD:
        rec = "REJECT"
    elif composite >= REVIEW_THRESHOLD:
        rec = "REVIEW"
    else:
        rec = "APPROVE"

    return {
        "composite_risk_score": round(composite, 1),
        "recommendation": rec,
        "component_scores": {"image": image_score, "metadata": metadata_score, "agents": agent_score},
        "raw_metadata_probability": round(raw_meta, 3),
        "agent_consensus": agent_assessment.get("consensus") if agent_assessment else None,
    }


def heuristic_metadata_prob(meta: dict) -> float:
    """The same generative `_fraud_probability` model used in the notebook's
    synthetic-data slice. Used here as a real-time stand-in for the trained
    Random Forest, since the RF needs a fitted model loaded from disk."""
    p = 0.04
    if meta.get("is_weekend_transaction"):
        p += 0.18
    if meta.get("amount_rounded"):
        p += 0.12
    if meta.get("duplicate_vendor_same_day"):
        p += 0.35
    if (meta.get("amount") or 0) > 250:
        p += 0.15
    if meta.get("document_type") == "bank_statement":
        p += 0.05
    return min(p, 0.95)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detector — Real-Time Document Analysis",
    page_icon=":mag:",
    layout="wide",
)

st.title("AI-Powered Financial Fraud Detector")
st.caption(
    "Upload a receipt, invoice, or bank-statement image. Vision analysis, "
    "metadata heuristics, and a multi-agent expert committee combine into a "
    "single composite risk score and recommendation."
)

with st.sidebar:
    st.header("API Keys")
    st.text_input(
        "Google AI / Gemini key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        key="gemini_key",
        help="Used for the Gemma vision model and the multi-agent committee. "
             "Pre-filled from GEMINI_API_KEY env var if available.",
    )
    st.text_input(
        "OpenRouter key (optional)",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        type="password",
        key="openrouter_key",
        help="Required only if you want to use Nemotron Nano 12B VL instead of "
             "Gemma. Get one at openrouter.ai/keys. Pre-filled from "
             "OPENROUTER_API_KEY env var if available.",
    )
    if not _gemini_key():
        st.error("Gemini key required to run the pipeline.")

    st.divider()
    st.header("Configuration")
    use_nemotron = st.toggle(
        "Use Nemotron (NVIDIA Nano 12B VL) instead of Gemma",
        value=False,
        help="Mirrors the USE_NEMOTRON_VISION toggle in cell 1 of the notebook. "
             "Requires an OpenRouter key above.",
    )
    if use_nemotron and not _openrouter_key():
        st.warning("OpenRouter key not provided; will fall back to Gemma.")

    run_agents = st.toggle(
        "Run multi-agent committee",
        value=True,
        help="Three expert personas (forensic accountant, compliance officer, "
             "risk analyst). Adds ~21s of API calls.",
    )

    st.divider()
    st.caption(f"Gemini:     {'configured' if _gemini_key() else 'missing'}")
    st.caption(f"OpenRouter: {'configured' if _openrouter_key() else 'not set'}")

uploaded = st.file_uploader(
    "Upload financial document image (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    help="The image is sent to the configured vision model for forensic analysis.",
)

with st.expander("Optional metadata (improves multi-agent + metadata channel accuracy)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        document_id = st.text_input("Document ID", value="UPLOAD_001")
        document_type = st.selectbox("Document type", ["invoice", "receipt", "bank_statement"])
        vendor_name = st.text_input("Vendor / supplier", value="")
    with col2:
        amount = st.number_input("Amount", min_value=0.0, value=0.0, step=10.0, format="%.2f")
        date_created = st.date_input("Date", value=date.today())
        is_weekend = st.checkbox("Weekend transaction", value=False)
    with col3:
        amount_rounded = st.checkbox("Round amount (e.g. $500.00)", value=False)
        duplicate_same_day = st.checkbox("Duplicate vendor same day", value=False)

run_clicked = st.button("Analyze document", type="primary", disabled=uploaded is None)


# ---------------------------------------------------------------------------
# Run pipeline on click
# ---------------------------------------------------------------------------

if run_clicked and uploaded is not None:
    image = Image.open(uploaded)

    preview_col, results_col = st.columns([1, 2])
    with preview_col:
        st.image(image, caption=uploaded.name, use_container_width=True)

    metadata = {
        "document_id": document_id,
        "document_type": document_type,
        "amount": amount,
        "date_created": str(date_created),
        "vendor_name": vendor_name,
        "is_weekend_transaction": int(is_weekend),
        "amount_rounded": int(amount_rounded),
        "duplicate_vendor_same_day": int(duplicate_same_day),
    }

    with results_col:
        # ---- Vision channel ----
        nemotron_ready = use_nemotron and bool(_openrouter_key())
        active_model = "Nemotron" if nemotron_ready else "Gemma"
        with st.status(f"Vision analysis ({active_model}) ...", expanded=False) as status:
            if nemotron_ready:
                image_analysis = analyze_image_nemotron(image)
            else:
                image_analysis = analyze_image_gemma(image)
            if "error" in image_analysis:
                status.update(label=f"Vision analysis ({active_model}) — failed",
                              state="error")
                st.error(
                    f"Vision call failed: **{image_analysis['error']}**. "
                    "This is usually a transient server error — click *Analyze "
                    "document* again. If it persists, check the API key in the "
                    "sidebar or try toggling between Gemma and Nemotron."
                )
            else:
                status.update(label=f"Vision analysis ({active_model}) — done",
                              state="complete")

        # ---- Metadata channel ----
        meta_prob = heuristic_metadata_prob(metadata)
        metadata_pred = {"fraud_probability": meta_prob}

        # ---- Multi-agent channel ----
        agent_assessment = None
        if run_agents:
            with st.status("Multi-agent committee ...", expanded=False) as status:
                progress = st.progress(0.0, text="Starting...")
                def _cb(i, expert):
                    progress.progress(i / len(EXPERT_PROMPTS),
                                      text=f"Querying {expert} ...")
                agent_assessment = multi_agent_assessment(metadata, progress_cb=_cb)
                progress.progress(1.0, text="Done")
                status.update(label="Multi-agent committee — done", state="complete")

        # ---- Integrate ----
        report = integrate(image_analysis, metadata_pred, agent_assessment)

        # Big metric + color-coded recommendation
        rec = report["recommendation"]
        rec_color = {"APPROVE": "#22c55e", "REVIEW": "#f59e0b", "REJECT": "#ef4444"}[rec]
        st.markdown(
            f"### Composite risk: **{report['composite_risk_score']}/100** &nbsp; "
            f"<span style='background:{rec_color};color:white;padding:4px 12px;"
            f"border-radius:6px;font-size:14px;'>{rec}</span>",
            unsafe_allow_html=True,
        )

        comp = report["component_scores"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Image (vision)", f"{comp['image']:.0f} / 100")
        c2.metric("Metadata", f"{comp['metadata']:.0f} / 100",
                  help=f"Raw RF probability: {report['raw_metadata_probability']}")
        c3.metric("Agents", f"{comp['agents']:.0f} / 100",
                  help=f"Consensus: {report['agent_consensus']}" if report['agent_consensus'] else "")

        st.divider()

        with st.expander("Vision analysis details"):
            st.json(image_analysis)

        if agent_assessment:
            with st.expander("Multi-agent committee details"):
                for op in agent_assessment["opinions"]:
                    if "error" in op:
                        st.error(f"**{op['expert_type']}** — {op['error']}")
                        continue
                    st.markdown(
                        f"**{op['expert_type']}** — verdict: *{op.get('verdict', '?')}*, "
                        f"risk score: {op.get('risk_score', '?')}/10"
                    )
                    if op.get("rationale"):
                        st.caption(op["rationale"])
                    if op.get("key_concerns"):
                        st.markdown("Concerns: " + ", ".join(op["key_concerns"]))
                    st.divider()

        with st.expander("Full JSON report"):
            full = {
                "document_id": document_id,
                "uploaded_filename": uploaded.name,
                "active_vision_model": active_model,
                "metadata_input": metadata,
                "image_analysis": image_analysis,
                "metadata_prediction": metadata_pred,
                "agent_assessment": agent_assessment,
                "integrated_report": report,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
            st.json(full)
            st.download_button(
                "Download JSON report",
                data=json.dumps(full, indent=2, default=str),
                file_name=f"{document_id}_fraud_report.json",
                mime="application/json",
            )
