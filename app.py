"""
============================================================
 Crop Disease Detection System
 By Musinguzi Dickson, Muhairwe Dominic, Kalyegira Emmanuel, Emelda Nakacwa
 File: app.py
 Description: A Streamlit web application for live crop disease
              diagnosis. Upload a leaf photo and get AI diagnosis
              + treatment recommendation.

 Run with: streamlit run app.py
============================================================
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONSTANTS ──────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = "models/crop_disease_model.h5"
LABELS_PATH = "models/class_labels.json"
TOP_K       = 3   # Show top-3 predictions

# ─── TREATMENT DATABASE ─────────────────────────────────────
# Maps disease class names to treatment recommendations
TREATMENTS = {
    "Tomato___Late_blight": {
        "description": "Caused by Phytophthora infestans. Brown lesions on leaves and stems.",
        "treatment": "Apply copper-based fungicide. Remove infected leaves. Improve air circulation.",
        "prevention": "Use resistant varieties. Avoid overhead irrigation. Rotate crops."
    },
    "Tomato___Early_blight": {
        "description": "Caused by Alternaria solani. Dark spots with concentric rings.",
        "treatment": "Apply mancozeb or chlorothalonil fungicide. Remove lower infected leaves.",
        "prevention": "Mulch soil surface. Stake plants for better airflow."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Caused by Puccinia sorghi. Orange-red pustules on leaves.",
        "treatment": "Apply triazole fungicide at early infection. Remove crop debris after harvest.",
        "prevention": "Plant resistant hybrid varieties. Scout fields early in season."
    },
    "Cassava___Bacterial_Blight": {
        "description": "Angular water-soaked lesions, wilting and die-back of shoots.",
        "treatment": "Remove and destroy infected plant parts. Use clean planting material.",
        "prevention": "Use certified disease-free cuttings. Avoid working in wet fields."
    },
    "Cassava___Brown_Streak_Disease": {
        "description": "Yellow streaks on leaves, brown necrosis in tubers.",
        "treatment": "No cure — destroy infected plants to prevent spread.",
        "prevention": "Use virus-tested cuttings. Control whitefly vectors."
    },
    # Default  for any unmatched class
    "DEFAULT": {
        "description": "AI has detected a potential crop health issue.",
        "treatment": "Consult a local agronomist for specific treatment recommendations.",
        "prevention": "Practice crop rotation and use certified disease-free planting material."
    }
}


# ─── MODEL LOADING ──────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    """Load model once and cache it for performance."""
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    return model, labels


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalize image for model input.
    Input:  PIL Image (any size, any mode)
    Output: numpy array of shape (1, 224, 224, 3), values in [0, 1]
    """
    image = image.convert("RGB")           # Ensure 3-channel RGB
    image = image.resize(IMG_SIZE)         # Resizing to model input size
    arr   = np.array(image) / 255.0       # Normalize to [0, 1]
    return np.expand_dims(arr, axis=0)     # Add batch dimension


def get_prediction(model, image_array, labels, top_k=3):
    """
    Run inference and return top-K predictions with confidence.
    Returns list of (class_name, confidence_percent) tuples.
    """
    proba   = model.predict(image_array, verbose=0)[0]  # Shape: (num_classes,)
    top_idx = np.argsort(proba)[::-1][:top_k]           # Indices of top-K classes

    results = []
    for idx in top_idx:
        class_name  = labels[str(idx)]
        confidence  = float(proba[idx]) * 100
        results.append((class_name, confidence))

    return results


def format_class_name(raw: str) -> str:
    """Convert 'Tomato___Late_blight' → 'Tomato — Late Blight'"""
    parts = raw.replace("___", "|").replace("_", " ").split("|")
    if len(parts) == 2:
        return f"{parts[0].strip().title()} — {parts[1].strip().title()}"
    return raw.replace("_", " ").title()


def get_treatment(class_name: str) -> dict:
    """Look up treatment info, fallback to default if not found."""
    return TREATMENTS.get(class_name, TREATMENTS["DEFAULT"])


# ─── UI ─────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRa1mISqJ8xAERPSZEJYlvb9SbYShQD727xbPKnIBB9A5NLBQudaWRRCXwBBaWN1p4gwU3q7q1JW6FSG7TWUd4U8mgAkmLw85IMoYx-qajpG_r2xX2mE7rhuNlMR-syleM&s=10&ec=121630528",
                 caption="Example: Cassava Mosaic Disease")
        st.markdown("### About This System")
        st.markdown("""
        This AI-powered system helps farmers identify crop diseases
        from a simple leaf photograph.

        **Supported crops include:**
        - Tomato
        - Maize (Corn)
        - Cassava
        - Potato
        - Grapes
        - Apples and more

        **How it works:**
        1. Upload a photo of a diseased leaf
        2. The CNN model analyses the image
        3. You receive a diagnosis + treatment advice

        **Model:** MobileNetV2 (Transfer Learning)
        **Dataset:** PlantVillage (50,000+ images)
        **Accuracy:** ~90%+ on test set
        """)
        st.markdown("---")
        st.markdown("*Group 2 Agriculture AI Project*")

    # Main area
    st.title("🌿 Crop Disease Detection System")
    st.markdown(
        "Upload a clear photo of a plant leaf and the AI will diagnose "
        "the disease and recommend treatment."
    )
    st.markdown("---")

    # Load model
    model, labels = load_model_and_labels()

    if model is None:
        st.error(
            "⚠️ **Model not found.** Please run `train_model.py` first to train "
            "and save the model, then restart this app."
        )
        st.code("python src/train_model.py", language="bash")
        return

    # File upload
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📷 Upload Leaf Image")
        uploaded = st.file_uploader(
            "Choose an image (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Take a close-up photo of the affected leaf for best results."
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if uploaded:
            st.subheader("🔬 AI Diagnosis")

            with st.spinner("Analysing image..."):
                img_array = preprocess_image(image)
                predictions = get_prediction(model, img_array, labels, top_k=TOP_K)

            # Top prediction
            top_class, top_conf = predictions[0]
            display_name = format_class_name(top_class)
            treatment = get_treatment(top_class)

            # Confidence colour
            conf_color = "green" if top_conf > 70 else "orange" if top_conf > 50 else "red"

            st.markdown(f"### 🌱 Diagnosis: **{display_name}**")
            st.markdown(
                f"<p style='font-size:18px;color:{conf_color};'>"
                f"Confidence: <strong>{top_conf:.1f}%</strong></p>",
                unsafe_allow_html=True
            )

            # Description
            st.info(f"📋 **Description:** {treatment['description']}")

            # Treatment & prevention
            with st.expander("💊 Recommended Treatment", expanded=True):
                st.success(treatment["treatment"])

            with st.expander("🛡️ Prevention Tips"):
                st.warning(treatment["prevention"])

            # Alternative predictions
            st.markdown("#### Other Possibilities")
            for cls, conf in predictions[1:]:
                st.progress(int(conf), text=f"{format_class_name(cls)} — {conf:.1f}%")

            # Disclaimer
            st.markdown("---")
            st.caption(
                "⚠️ This AI diagnosis is a decision-support tool. Always verify with "
                "a certified agronomist before applying treatments."
            )

        else:
            st.info("👈 Upload a leaf image on the left to get started.")
            st.markdown("""
            **Tips for best results:**
            - Use a clear, well-lit photo
            - Focus on the affected leaf area
            - Avoid shadows or blurry images
            - Single leaf works better than full plant shots
            """)
if __name__ == "__main__":
    main()
