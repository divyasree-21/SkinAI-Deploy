
import hashlib
import numpy as np
import streamlit as st
from PIL import Image

from history_db import (
    save_screening_result,
    get_recent_history,
    delete_all_history,
)

from inference import predict, CLASS_NAMES
from gradcam_app import generate_gradcam


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SkinAI | Intelligent Screening",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# PREMIUM DARK UI
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%,
                rgba(101, 74, 220, 0.17), transparent 38%),
            #0b1020;
        color: #f3f4f8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    .block-container {
        max-width: 1180px;
        padding: 1.5rem 1.2rem 3rem;
    }

    [data-testid="stSidebar"] {
        background: #11182b;
    }

    .brand {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: #c4b5fd;
    }

    .hero {
        padding: clamp(1.5rem, 5vw, 3.5rem);
        border: 1px solid rgba(167, 139, 250, 0.27);
        border-radius: 26px;
        background: linear-gradient(
            125deg,
            rgba(43, 42, 91, 0.96),
            rgba(20, 29, 53, 0.96) 65%,
            rgba(18, 42, 59, 0.95)
        );
        margin: 1.2rem 0 1.6rem;
    }

    .eyebrow {
        color: #c4b5fd;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.1rem, 5vw, 3.6rem);
        line-height: 1.12;
        margin: 0.8rem 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1rem;
        max-width: 670px;
        line-height: 1.8;
    }

    .section-label {
        color: #a78bfa;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .panel {
        background: rgba(20, 28, 48, 0.88);
        border: 1px solid #29334d;
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .panel h3 {
        color: #f8fafc;
        margin-top: 0.35rem;
    }

    .muted {
        color: #a8b3c7;
        line-height: 1.7;
    }

    .result-card {
        background: linear-gradient(
            135deg,
            rgba(91, 67, 174, 0.22),
            rgba(20, 28, 48, 0.96)
        );
        border: 1px solid rgba(167, 139, 250, 0.42);
        border-radius: 20px;
        padding: 1.3rem;
        margin: 0.8rem 0 1rem;
    }

    .result-class {
        color: #ddd6fe;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    div.stButton > button {
        min-height: 3rem;
        border-radius: 12px;
        border: 1px solid #8b78ed;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #7c3aed, #6366f1);
        color: white;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: rgba(20, 28, 48, 0.6);
        border: 1px dashed #7162b5;
        border-radius: 15px;
        padding: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: #141c30;
        border: 1px solid #29334d;
        padding: 1rem;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #a8b3c7;
    }

    [data-testid="stMetricValue"] {
        color: #e9e4ff;
    }

    hr {
        border-color: #29334d;
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 0.8rem 0.7rem 2rem;
        }

        .hero {
            border-radius: 18px;
            padding: 1.4rem 1rem;
        }

        .panel {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# WELCOME SCREEN — no account or signup required
# --------------------------------------------------
if not st.session_state.get("skinai_started", False):
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(145deg, #f3fbfc 0%, #edf5fb 58%, #e6f5f2 100%);
            color: #163447;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1050px; padding-top: 2rem; }
        .welcome-top { display:flex; align-items:center; gap:12px; color:#155e75;
            font-weight:800; letter-spacing:.13em; font-size:.9rem; }
        .medical-mark { width:48px; height:48px; border-radius:16px;
            display:flex; align-items:center; justify-content:center;
            background:#d5f2ee; color:#087e8b; font-size:1.8rem; font-weight:800;
            box-shadow:0 8px 24px rgba(8,126,139,.12); }
        .welcome-card { margin:2.2rem 0 1rem; padding:clamp(1.5rem,5vw,4rem);
            border:1px solid #d7e9ee; border-radius:30px;
            background:rgba(255,255,255,.88);
            box-shadow:0 24px 70px rgba(29,78,96,.09); }
        .welcome-eyebrow { color:#087e8b; font-size:.76rem; font-weight:800;
            letter-spacing:.16em; text-transform:uppercase; }
        .welcome-card h1 { color:#12384a; font-size:clamp(2.3rem,6vw,4.1rem);
            line-height:1.08; margin:.8rem 0 1rem; }
        .welcome-card p { color:#4a6675; max-width:680px; font-size:1.05rem; line-height:1.8; }
        .welcome-points { display:flex; flex-wrap:wrap; gap:10px; margin-top:1.5rem; }
        .welcome-chip { border:1px solid #d3e9e9; border-radius:999px; padding:8px 12px;
            color:#245c69; background:#f4fbfa; font-size:.86rem; }
        .welcome-note { color:#637b87; font-size:.82rem; line-height:1.65; margin-top:1.2rem; }
        div.stButton > button[kind="primary"] { background:#087e8b; color:#fff;
            border:0; border-radius:14px; min-height:3.2rem; font-size:1.05rem; }
        div.stButton > button[kind="primary"]:hover { background:#066875; border:0; }
        @media (max-width:600px) { .welcome-card { border-radius:22px; padding:1.35rem; } }
        </style>
        <div class="welcome-top"><div class="medical-mark">+</div><span>SKINAI · EARLY SCREENING RESEARCH</span></div>
        <div class="welcome-card">
          <div class="welcome-eyebrow">EXPLAINABLE AI · HUMAN-CENTRED CARE</div>
          <h1>A thoughtful first step<br>for skin health.</h1>
          <p>Explore an AI-assisted skin image screening prototype with a visual explanation of the model’s attention. No account or signup is needed to begin.</p>
          <div class="welcome-points">
            <span class="welcome-chip">＋ Camera or gallery</span>
            <span class="welcome-chip">＋ AI-assisted screening</span>
            <span class="welcome-chip">＋ Visual heatmap</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, start_col, _ = st.columns([1, 1.4, 1])
    with start_col:
        if st.button("Start screening  →", type="primary", use_container_width=True):
            st.session_state["skinai_started"] = True
            st.rerun()
    st.markdown(
        '<div class="welcome-note" style="text-align:center">Research prototype only. It does not diagnose skin conditions or replace advice from a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------
# HEADER AND HERO
# --------------------------------------------------
header_left, header_right = st.columns([1.5, 1])

with header_left:
    st.markdown(
        '<div class="brand">✦ SKINAI</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;color:#a8b3c7;'
        'padding-top:5px;font-size:0.85rem;">'
        'AI RESEARCH PROTOTYPE · v1.0</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">EXPLAINABLE EDGE AI</div>
        <h1>Skin screening,<br>with AI-assisted insights.</h1>
        <p>
            Explore a research prototype that combines image-based
            classification with visual explanations to support
            early skin disease screening research.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# PROJECT HIGHLIGHTS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<div class="panel"><div class="section-label">'
        'MODEL</div><h3>GoogLeNet</h3>'
        '<div class="muted">Knowledge-distilled student model</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="panel"><div class="section-label">'
        'INFERENCE</div><h3>ONNX Runtime</h3>'
        '<div class="muted">Local CPU inference in this app</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        '<div class="panel"><div class="section-label">'
        'EXPLAINABILITY</div><h3>Grad-CAM</h3>'
        '<div class="muted">Visual model explanation</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown('<div class="section-label">01 / ANALYZE</div>',
            unsafe_allow_html=True)
st.header("Explore a skin image")
st.markdown(
    '<p class="muted">Upload a JPG, JPEG, or PNG image to run '
    'the screening prototype.</p>',
    unsafe_allow_html=True,
)

camera_image = st.camera_input(
    "📸 Capture an image using your camera",
    key="skinai_camera_upload",
)

uploaded_file = st.file_uploader(
    "🖼️ Or choose an image from your gallery",
    type=["jpg", "jpeg", "png"],
    key="skinai_image_upload",
)

# Prefer the camera capture when both inputs contain an image.
selected_file = camera_image if camera_image is not None else uploaded_file

if selected_file is not None:
    try:
        image_bytes = selected_file.getvalue()
        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        image_hash = hashlib.sha256(image_bytes).hexdigest()

        left, right = st.columns([1, 1.15], gap="large")

        with left:
            st.markdown(
                '<div class="section-label">YOUR IMAGE</div>',
                unsafe_allow_html=True,
            )
            st.image(
                image,
                caption="Selected image",
                use_container_width=True,
            )

        with right:
            st.markdown(
                '<div class="panel">'
                '<div class="section-label">READY TO ANALYZE</div>'
                '<h3>AI-assisted screening</h3>'
                '<p class="muted">Your image will be resized and '
                'normalized by the existing inference pipeline.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

            analyze_clicked = st.button(
                "✦  Analyze image",
                type="primary",
                use_container_width=True,
            )

        # Only reuse the current result if the uploaded image is unchanged.
        if analyze_clicked:
            with st.spinner("Running model inference..."):
                logits = np.asarray(
                    predict(image),
                    dtype=np.float64,
                ).reshape(-1)

                if (
                    len(logits) != len(CLASS_NAMES)
                    or not np.all(np.isfinite(logits))
                ):
                    raise ValueError(
                        "The model returned an invalid prediction."
                    )

                shifted = logits - np.max(logits)
                exp_logits = np.exp(shifted)
                probabilities = exp_logits / exp_logits.sum()

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(
                    probabilities[predicted_index] * 100
                )

                save_screening_result(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    probabilities=probabilities.tolist(),
             )

                st.session_state["skinai_last_result"] = {
                    "image_hash": image_hash,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                }

        result = st.session_state.get("skinai_last_result")

        if result and result.get("image_hash") == image_hash:
            st.divider()
            st.markdown(
                '<div class="section-label">02 / RESULTS</div>',
                unsafe_allow_html=True,
            )
            st.header("Screening overview")

            st.markdown(
                f'<div class="result-card">'
                f'<div class="section-label">MODEL PREDICTION</div>'
                f'<div class="result-class">'
                f'{result["predicted_class"]}</div>'
                f'<div class="muted">Predicted class from the '
                f'configured prototype classes</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Model confidence",
                    f'{result["confidence"]:.2f}%',
                )

            with metric2:
                st.metric("Classes configured", len(CLASS_NAMES))

            st.caption(
                "Model confidence is derived from the model's "
                "softmax output. It is not clinical certainty."
            )

            st.subheader("Class probability distribution")

            for index, class_name in enumerate(CLASS_NAMES):
                score = float(result["probabilities"][index])
                st.write(f"**{class_name}** · {score * 100:.2f}%")
                st.progress(min(max(score, 0.0), 1.0))

            st.divider()
            st.markdown(
                '<div class="section-label">03 / EXPLAINABILITY</div>',
                unsafe_allow_html=True,
            )
            st.header("Inside the model's attention")

            st.markdown(
                '<p class="muted">Grad-CAM highlights image '
                'regions that influenced the model output. '
                'These visualizations are exploratory explanations, '
                'not verified lesion boundaries.</p>',
                unsafe_allow_html=True,
            )

            try:
                with st.spinner("Generating Grad-CAM..."):
                    _, heatmap, overlay = generate_gradcam(image)

                cam1, cam2 = st.columns(2)

                with cam1:
                    st.image(
                        heatmap,
                        caption="Grad-CAM heatmap",
                        use_container_width=True,
                    )

                with cam2:
                    st.image(
                        overlay,
                        caption="Grad-CAM overlay",
                        use_container_width=True,
                    )

            except Exception:
                st.warning(
                    "The prediction succeeded, but the Grad-CAM "
                    "visualization could not be generated for "
                    "this image. You can still view the prediction."
                )

            st.info(
                "This prototype is configured for AK and BCC only. "
                "It may misclassify other conditions or unsuitable "
                "images. Do not use the result to make treatment "
                "decisions; consult a qualified healthcare professional."
            )

        elif result:
            st.caption(
                "The selected image has changed. Analyze it again "
                "to generate a new result."
            )

    except Exception:
        st.error(
            "We couldn't process this image. Please try another "
            "JPG, JPEG, or PNG file and check that the model "
            "dependencies are installed."
        )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------
st.divider()

with st.expander("About the SkinAI research prototype"):
    st.write("**Teacher:** RegNet-Y-32GF")
    st.write("**Student:** GoogLeNet")
    st.write("**Training approach:** Knowledge distillation")
    st.write("**Prediction runtime:** ONNX Runtime")
    st.write("**Explainability:** Grad-CAM using the PyTorch model")
    st.write("**Input dimensions:** 224 × 224 pixels")
    st.write("**Configured classes:** AK and BCC")

st.markdown(
    """
    <div style="text-align:center;padding:1.5rem 0 0.5rem;">
        <div class="brand">✦ SKINAI</div>
        <p class="muted" style="font-size:0.8rem;">
            AI-Assisted Early Skin Disease Screening<br>
            Research prototype · Not a medical diagnostic system
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SCREENING HISTORY
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-label">04 / HISTORY</div>',
    unsafe_allow_html=True,
)
st.header("Screening history")

st.markdown(
    '<p class="muted">Your recent screening results saved '
    'on this device\'s application database.</p>',
    unsafe_allow_html=True,
)


history = get_recent_history(limit=50)

if history:
    display_records = []

    for record in history:
        probabilities = record["Probabilities"]

        # Handle probabilities stored as either a list or dictionary
        if isinstance(probabilities, list):
            probability_text = ", ".join(
                f"{CLASS_NAMES[i]}: {float(value) * 100:.1f}%"
                for i, value in enumerate(probabilities)
                if i < len(CLASS_NAMES)
            )
        elif isinstance(probabilities, dict):
            probability_text = ", ".join(
                f"{name}: {float(value) * 100:.1f}%"
                for name, value in probabilities.items()
            )
        else:
            probability_text = "Unavailable"

        display_records.append({
            "Date and time": record["Date and time"],
            "Prediction": record["Predicted class"],
            "Confidence (%)": record["Confidence (%)"],
            "Class probabilities": probability_text,
        })

    st.dataframe(
        display_records,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(display_records)} recent screening records."
    )

    with st.expander("Manage screening history"):
        st.warning(
            "Deleting history permanently removes all saved records."
        )

        confirm_delete = st.checkbox(
            "I understand that all saved results will be deleted."
        )

        if st.button(
            "Delete all history",
            disabled=not confirm_delete,
        ):
            delete_all_history()
            st.success("Screening history deleted.")
            st.rerun()

else:
    st.info(
        "No screening history yet. Analyze an image to create "
        "your first saved record."
    )
