"""
Global Dog Breed Identification Using Deep Learning — Streamlit Web App.

Identifies dog breeds across 356 supported dog breed classes and provides
detailed breed information, history, origin, temperament, and health details.
"""

import sys
import os
import json

# Add project root and app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from PIL import Image

from src.predict import load_production_model, predict_image
from src.dog_detector import detect_dog
from src.data_loader import DOG_BREEDS_CLASSES, NUM_CLASSES
from utils import validate_uploaded_file, load_image_from_upload, get_image_info


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Global Dog Breed Identification",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS Styling ──────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global Typography & Spacing */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Main Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        color: #FFFFFF !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .hero-badge {
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 0.03em;
    }

    /* Primary Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: #FFFFFF;
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .prediction-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #93C5FD;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .prediction-breed-name {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        line-height: 1.25;
        color: #FFFFFF;
    }
    .prediction-conf-pill {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.35rem 1.1rem;
        border-radius: 30px;
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Upload Placeholder */
    .upload-placeholder {
        text-align: center;
        padding: 2.8rem 1.5rem;
        background: rgba(248, 250, 252, 0.04);
        border-radius: 14px;
        border: 2px dashed #94A3B8;
        margin-top: 1rem;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(254, 243, 199, 0.25);
        color: #D97706;
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        font-size: 0.88rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Model & Breed Info Database Loaders ─────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained 356-class dog breed model once at startup."""
    try:
        model, model_type, img_size, class_names = load_production_model()
        return model, model_type, img_size, class_names, None
    except Exception as e:
        return None, None, None, None, str(e)


@st.cache_data
def load_breed_info_db():
    """Load 356 dog breed information database."""
    json_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'breed_information', 'dog_breed_info.json'
    )
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ─── Breed Information Component ─────────────────────────────────────────────

def render_breed_info_section(breed_name, breed_db):
    """
    Render all 15 detailed information fields for a specific breed in a clean, balanced layout.
    """
    info = breed_db.get(breed_name, None)

    st.divider()
    st.subheader(f"📖 Dog Breed Information & History: {breed_name}")

    if not info:
        st.info(f"Information entry for **{breed_name}** is currently being generated.")
        return

    # Balanced Two-Column Overview Layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🐕 Breed Name:** {info.get('breed_name', breed_name)}")
        st.markdown(f"**🏷️ Group / Type:** {info.get('group', 'Working Dog')}")
        st.markdown(f"**📍 Country of Origin:** {info.get('origin', 'International')}")
        st.markdown(f"**🎯 Original Purpose:** {info.get('purpose', 'Companionship & Working')}")
        st.markdown(f"**⏳ Average Lifespan:** {info.get('lifespan', '10–14 years')}")

    with col2:
        st.markdown(f"**📏 Size Category:** {info.get('size', 'Medium')}")
        st.markdown(f"**📐 Average Height:** {info.get('height', '45–60 cm')}")
        st.markdown(f"**⚖️ Average Weight:** {info.get('weight', '15–30 kg')}")
        st.markdown(f"**🧠 Temperament:** {info.get('temperament', 'Loyal, Intelligent, Friendly')}")

    # Tabs for Detailed Information
    tab1, tab2, tab3, tab4 = st.tabs(["📜 Breed History", "🏃 Care & Living", "🏥 Health Info", "💡 Fun Facts"])

    with tab1:
        st.markdown("#### 📜 Breed History & Evolution")
        st.write(info.get('history', 'Historical records document this breed across working and companionship roles.'))

    with tab2:
        st.markdown("#### 🏃 Exercise, Grooming & Environment")
        st.markdown(f"**⚡ Exercise Needs:** {info.get('exercise_needs', 'Daily walks and active play required.')}")
        st.markdown(f"**✂️ Grooming Needs:** {info.get('grooming_needs', 'Regular coat maintenance and brushing.')}")
        st.markdown(f"**🏠 Suitable Living Environment:** {info.get('environment', 'Suitable for home or apartment living with adequate exercise.')}")

    with tab3:
        st.markdown("#### 🏥 Common Health Information")
        st.write(info.get('health_info', 'Routine health checkups, hip screenings, and vaccinations recommended.'))

    with tab4:
        st.markdown("#### 💡 Interesting Facts")
        st.write(info.get('facts', 'Recognized in international 356 FCI breed classification catalogs.'))

    # Required Veterinary Disclaimer
    st.markdown(
        """
        <div class='disclaimer-box'>
            ⚠️ <strong>Disclaimer:</strong> This information is for general educational purposes only and is not a substitute for professional veterinary advice.
        </div>
        """,
        unsafe_allow_html=True
    )


# ─── Sidebar Component ───────────────────────────────────────────────────────

def render_sidebar(class_names):
    """Render sidebar with project information and quick model summary."""
    with st.sidebar:
        st.header("🐶 Breed Classifier AI")
        st.markdown(
            "Identify dog breeds instantly using Deep Learning and explore "
            "detailed history across **356 dog breeds**."
        )

        st.info("ℹ️ **Supports 356 dog breeds included in the trained dataset.**")

        st.divider()

        # Model Info
        model, model_type, img_size, _, error = load_model()
        if model is not None:
            st.subheader("🧠 Model Architecture")
            m_name = "MobileNetV2 Transfer Learning" if model_type == 'transfer' else "DogBreed CNN Scratch"
            st.markdown(f"• **Architecture:** {m_name}")
            st.markdown(f"• **Input Resolution:** {img_size}×{img_size} RGB")
            st.markdown(f"• **Supported Classes:** 356 Breeds")

            comp_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_comparison.json')
            if os.path.exists(comp_path):
                with open(comp_path, 'r', encoding='utf-8') as f:
                    comp = json.load(f)
                best_acc = comp.get('best_accuracy', 0)
                st.markdown(f"• **Test Accuracy:** {best_acc*100:.2f}%")

        st.divider()
        st.caption("Built with PyTorch & Streamlit")


# ─── Main Application ────────────────────────────────────────────────────────

def main():
    model, model_type, img_size, class_names, error = load_model()
    if class_names is None:
        class_names = DOG_BREEDS_CLASSES

    breed_db = load_breed_info_db()
    render_sidebar(class_names)

    # Hero Header Section
    st.markdown(
        """
        <div class='hero-container'>
            <h1 class='hero-title'>🐕 Global Dog Breed Identification</h1>
            <p class='hero-subtitle'>AI-Powered Visual Recognition & Comprehensive Breed Intelligence</p>
            <span class='hero-badge'>Scope: Exactly 356 FCI Dog Breeds</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if error:
        st.warning(f"⚠️ **Model Initialization Note:** {error}")
        st.info("Train a model first by running: `python -m src.train` in your terminal.")

    # Main Navigation Tabs: Image Identification vs. Breed Search Explorer
    main_tab1, main_tab2 = st.tabs(["📷 Upload & Identify Image", "🔍 Search 356 Breed Information"])

    with main_tab1:
        st.subheader("📤 Upload Dog Image")

        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Choose a dog photo...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a dog (JPG/PNG, max 5 MB)",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded_file is not None:
            is_valid, error_msg = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return

            image = load_image_from_upload(uploaded_file)
            img_info = get_image_info(image)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🖼️ Uploaded Image")
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                st.caption(f"Resolution: {img_info['width']}×{img_info['height']} | Format: {img_info['format']}")

                # Clean "Try Another Image" reset button experience
                if st.button("🔄 Try Another Image", use_container_width=True, type="secondary"):
                    st.session_state.uploader_key += 1
                    st.rerun()

            with col2:
                st.subheader("🎯 Detection & Classification")

                if model is None:
                    st.error("Model is not ready for inference. Please check setup.")
                    return

                with st.spinner("Pre-screening image for dog detection..."):
                    detection_result = detect_dog(image)

                if not detection_result['is_dog']:
                    st.error("⚠️ **No dog detected. Please upload a clear image of a dog.**")
                    st.info("The uploaded image was screened by the dog detection filter and did not contain a recognizable dog breed.")
                else:
                    with st.spinner("Analyzing image features & classifying breed..."):
                        device = next(model.parameters()).device
                        result = predict_image(image, model, img_size, device, top_k=3, class_names=class_names)

                    pred_class = result['predicted_class']
                    conf_pct = result['confidence_pct']

                    # Prominent Predicted Breed Result Card
                    st.markdown(
                        f"""
                        <div class='prediction-card'>
                            <div class='prediction-card-title'>Predicted Dog Breed</div>
                            <div class='prediction-breed-name'>🐕 {pred_class}</div>
                            <div class='prediction-conf-pill'>Confidence: {conf_pct}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if result.get('non_dog_warning', False):
                        st.warning("⚠️ **Low Confidence Notice:** The prediction confidence is low. Please ensure the uploaded image contains a clear view of a dog breed.")

                    # Top 3 Predictions Display
                    st.divider()
                    st.subheader("📊 Top 3 Predictions")

                    top_preds = result['top_predictions']
                    medals = ["🥇", "🥈", "🥉"]
                    for rank, (breed, prob) in enumerate(top_preds, 1):
                        medal = medals[rank - 1] if rank <= 3 else f"#{rank}"
                        c_label, c_bar, c_pct = st.columns([3.5, 4.5, 2])
                        with c_label:
                            st.markdown(f"**{medal} {breed}**")
                        with c_bar:
                            st.progress(float(prob))
                        with c_pct:
                            st.markdown(f"**{prob*100:.1f}%**")

            # AUTOMATICALLY DISPLAY BREED INFORMATION & HISTORY FOR PREDICTED BREED ACROSS FULL CENTERED WIDTH
            if 'detection_result' in locals() and detection_result.get('is_dog') and 'pred_class' in locals():
                render_breed_info_section(pred_class, breed_db)

        else:
            st.markdown(
                """
                <div class='upload-placeholder'>
                    <h3 style='margin-bottom: 0.5rem; font-weight: 600;'>👆 Upload a dog photo to identify its breed</h3>
                    <p style='color: #94A3B8; margin-bottom: 0;'>Drag and drop or click <strong>Browse Files</strong> above</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with main_tab2:
        st.subheader("🔍 Explore Breed History & Information (356 Breeds)")
        st.markdown("Search or select any of the **356 supported dog breeds** to view its history, care requirements, and characteristics without uploading an image.")

        selected_breed = st.selectbox(
            "Select or type a dog breed name:",
            options=class_names,
            index=class_names.index("Golden Retriever") if "Golden Retriever" in class_names else 0
        )

        if selected_breed:
            render_breed_info_section(selected_breed, breed_db)


if __name__ == '__main__':
    main()
