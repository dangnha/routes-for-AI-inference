"""Streamlit frontend for the Local AI Inference Platform."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

import samples as sample_lib
from client import BackendClient

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Local AI Inference", page_icon="🧠", layout="wide")

CSS = """
<style>
.hero {
  background: linear-gradient(135deg,#4f46e5 0%,#0ea5e9 55%,#22d3ee 100%);
  padding: 2.1rem 2.4rem; border-radius: 18px; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 12px 30px rgba(79,70,229,.25);
}
.hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.5px; }
.hero p { margin: .45rem 0 0; opacity: .92; font-size: 1.02rem; }
.hero .pill {
  display:inline-block; margin-top:.9rem; margin-right:.4rem; padding:.22rem .7rem;
  border-radius:999px; background:rgba(255,255,255,.18); font-size:.78rem;
  border:1px solid rgba(255,255,255,.35);
}
.mcard {
  border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.2rem;
  background: rgba(128,128,128,.06); height: 100%;
}
.mcard h4 { margin: 0 0 .35rem; }
.mcard .tag { font-size:.82rem; opacity:.7; }
.badge { display:inline-block; padding:.16rem .62rem; border-radius:999px;
         font-size:.76rem; font-weight:700; }
.badge-pending  { background:#f59e0b22; color:#b45309; }
.badge-running  { background:#3b82f622; color:#1d4ed8; }
.badge-succeeded{ background:#22c55e22; color:#15803d; }
.badge-failed   { background:#ef444422; color:#b91c1c; }
.small-note { font-size:.82rem; opacity:.65; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS_BADGE = {
    "pending": ("badge-pending", "⏳ Pending"),
    "running": ("badge-running", "⚙️ Running"),
    "succeeded": ("badge-succeeded", "✅ Succeeded"),
    "failed": ("badge-failed", "❌ Failed"),
}


def status_badge(status: str) -> str:
    cls, text = STATUS_BADGE.get(status, ("badge-pending", status))
    return f'<span class="badge {cls}">{text}</span>'


if "client" not in st.session_state:
    st.session_state.client = BackendClient(BACKEND_URL)
if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None

client: BackendClient = st.session_state.client
client.token = st.session_state.token


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🧠 Local AI Inference Platform</h1>
          <p>Run OCR models entirely on your own machine — no cloud, no streaming, just webhooks.</p>
          <span class="pill">TrOCR · single line</span>
          <span class="pill">EasyOCR · multi-line</span>
          <span class="pill">GOT-OCR2.0 · documents</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def auth_view() -> None:
    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Log in", type="primary"):
                try:
                    data = client.login(email, password)
                    st.session_state.token = data["access_token"]
                    st.session_state.email = email
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Login failed: {exc}")

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            if st.form_submit_button("Register"):
                try:
                    client.register(email, password)
                    st.success("Registered — check your inbox if confirmation is on, then log in.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Register failed: {exc}")


def model_card(slug: str, info: dict) -> None:
    best_for = sample_lib.label(slug)
    st.markdown(
        f"""
        <div class="mcard">
          <h4>{info['name']}</h4>
          <div class="tag">{info['description']}</div>
          <div class="tag" style="margin-top:.4rem">Best for: {best_for}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_source_control(model_slug: str | None, key: str) -> tuple[bytes, str, str] | None:
    """Let the user pick a bundled example or upload their own image."""
    source = st.radio(
        "Image source",
        ["🖼️ Example images", "📤 Upload your own"],
        horizontal=True,
        key=f"{key}_source",
    )

    if source.startswith("🖼️"):
        files = (
            sample_lib.sample_files(model_slug)
            if model_slug
            else sample_lib.all_samples()
        )
        if not files:
            st.info("No bundled examples for this model — upload an image instead.")
            return None

        st.caption(sample_lib.label(model_slug) if model_slug else "All examples")
        cols = st.columns(min(len(files), 4))
        for idx, path in enumerate(files):
            with cols[idx % len(cols)]:
                st.image(str(path), width="stretch")
                st.caption(sample_lib.pretty_name(path))
                if st.button("Use this", key=f"{key}_btn_{idx}", width="stretch"):
                    st.session_state[f"{key}_pick"] = str(path)
                    st.session_state.pop(f"{key}_upload", None)
                    st.rerun()

        picked = st.session_state.get(f"{key}_pick")
        if picked and Path(picked).exists():
            return Path(picked).read_bytes(), Path(picked).name, "image/png"
        st.info("Pick an example above.")
        return None

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"{key}_upload",
    )
    if uploaded is not None:
        st.session_state.pop(f"{key}_pick", None)
        return uploaded.getvalue(), uploaded.name, (uploaded.type or "image/png")
    return None


def poll_task(task_id: str, placeholder) -> dict:
    while True:
        current = client.get_task(task_id)
        placeholder.markdown(
            f"{status_badge(current['status'])} &nbsp; "
            f"<span class='small-note'>task {task_id[:8]} · {current['status']}</span>",
            unsafe_allow_html=True,
        )
        if current["status"] in ("succeeded", "failed"):
            return current
        time.sleep(0.4)


def run_inference(
    image_bytes: bytes, filename: str, content_type: str, model: str, placeholder
) -> dict:
    presign = client.presign(filename, content_type)
    client.upload(presign["upload_url"], image_bytes, content_type)
    task = client.create_task(presign["object_key"], model, filename)
    return poll_task(task["id"], placeholder)


def draw_boxes(image_bytes: bytes, lines: list[dict]) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in lines:
        box = item.get("box") or []
        if len(box) < 2:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        draw.rectangle((min(xs), min(ys), max(xs), max(ys)), outline=(220, 40, 40), width=3)
    return image


def render_result(result: dict, image_bytes: bytes | None = None) -> None:
    badge = status_badge(result["status"])
    st.markdown(badge, unsafe_allow_html=True)

    if result["status"] == "failed":
        st.error(result.get("error") or "Inference failed")
        return

    text = result.get("text") or ""
    confidence = result.get("confidence")

    left, right = st.columns([3, 1])
    with left:
        st.markdown("**Recognized text**")
        st.code(text if text else "(empty)", language=None)
    with right:
        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%" if confidence is not None else "—",
        )
        st.caption(f"Model `{result['model']}`")

    lines = result.get("lines") or []
    if lines and image_bytes is not None:
        with st.expander(f"Detected regions ({len(lines)})", expanded=True):
            st.image(draw_boxes(image_bytes, lines), width="stretch")


def try_it_tab(models_by_slug: dict[str, dict]) -> None:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1 · Pick a model")
        slug = st.radio(
            "OCR model",
            options=list(models_by_slug),
            format_func=lambda s: models_by_slug[s]["name"],
            key="try_model",
        )
        model_card(slug, models_by_slug[slug])

        st.subheader("2 · Choose an image")
        source = image_source_control(slug, key="try")

        run = st.button(
            "▶ Run inference",
            type="primary",
            disabled=source is None,
            width="stretch",
        )

    with right:
        st.subheader("3 · Result")
        placeholder = st.empty()

    if run and source is not None:
        image_bytes, filename, content_type = source
        with left:
            st.image(image_bytes, caption=filename, width="stretch")
        try:
            result = run_inference(image_bytes, filename, content_type, slug, placeholder)
        except Exception as exc:  # noqa: BLE001
            placeholder.empty()
            st.error(f"Request failed: {exc}")
            return
        with right:
            render_result(result, image_bytes)


def compare_tab(models_by_slug: dict[str, dict]) -> None:
    st.subheader("Run every model on the same image")
    source = image_source_control(None, key="compare")

    if st.button("▶ Compare all models", type="primary", disabled=source is None):
        image_bytes, filename, content_type = source
        st.image(image_bytes, caption=filename, width="stretch")
        columns = st.columns(len(models_by_slug))
        for column, (slug, info) in zip(columns, models_by_slug.items()):
            with column:
                st.markdown(f"**{info['name']}**")
                placeholder = st.empty()
                try:
                    result = run_inference(
                        image_bytes, filename, content_type, slug, placeholder
                    )
                    render_result(result, image_bytes)
                except Exception as exc:  # noqa: BLE001
                    placeholder.empty()
                    st.error(str(exc))


def history_tab() -> None:
    st.subheader("Your recent tasks")
    if st.button("🔄 Refresh"):
        st.rerun()
    try:
        tasks = client.list_tasks()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load tasks: {exc}")
        return

    if not tasks:
        st.info("No tasks yet — run an inference first.")
        return

    for task in tasks:
        confidence = task.get("confidence")
        label = (
            f"{status_badge(task['status'])} &nbsp; **{task['filename']}** · "
            f"`{task['model']}`"
            + (f" · {confidence * 100:.1f}%" if confidence is not None else "")
        )
        with st.expander(f"{task['filename']} · {task['status']}"):
            st.markdown(label, unsafe_allow_html=True)
            if task.get("text"):
                st.code(task["text"], language=None)
            if task.get("error"):
                st.error(task["error"])


def workspace_view() -> None:
    with st.sidebar:
        st.markdown(f"Signed in as **{st.session_state.email}**")
        st.caption(f"Backend: {BACKEND_URL}")
        if st.button("Log out"):
            st.session_state.token = None
            st.session_state.email = None
            client.token = None
            st.rerun()

    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach the backend at {BACKEND_URL}: {exc}")
        return

    models_by_slug = {m["slug"]: m for m in models}

    try_it, compare, history = st.tabs(["🎯 Try it", "⚖️ Compare", "🕘 History"])
    with try_it:
        try_it_tab(models_by_slug)
    with compare:
        compare_tab(models_by_slug)
    with history:
        history_tab()


hero()
if st.session_state.token:
    workspace_view()
else:
    auth_view()
