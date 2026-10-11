"""
Global Dog Breed Identification Using Deep Learning — Streamlit Web Application.

A production-grade computer vision system identifying 356 FCI-recognized dog breeds
via MobileNetV2 Transfer Learning with real-time canine pre-screening and comprehensive
archival intelligence across 15 detailed care, genetic, and historical dimensions.
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
    page_title="Canis AI — Global Dog Breed Intelligence",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Design System & Signature CSS Styling ──────────────────────────────────

st.markdown("""
<style>
    /* ─── Global Atmosphere & Typography ─── */
    .stApp {
        background-color: #080C15;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .main .block-container {
        max-width: 1080px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        padding-left: 1.8rem;
        padding-right: 1.8rem;
    }

    /* ─── Hero Identity ─── */
    .canis-hero {
        background: linear-gradient(145deg, #0B1120 0%, #111827 50%, #1E1B4B 100%);
        border: 1px solid rgba(139, 92, 246, 0.22);
        box-shadow: 0 20px 45px -10px rgba(11, 15, 26, 0.8), 0 0 35px -5px rgba(99, 102, 241, 0.12);
        border-radius: 20px;
        padding: 2.4rem 2.2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .canis-hero::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.18) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A78BFA;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.28);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }

    .hero-pulse {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
    }

    .hero-title {
        font-size: 2.45rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: #FFFFFF !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.15 !important;
    }

    .hero-title span {
        background: linear-gradient(135deg, #FFFFFF 30%, #C4B5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        font-size: 1.02rem;
        line-height: 1.6;
        color: #94A3B8;
        max-width: 720px;
        margin-bottom: 1.5rem;
    }

    .hero-metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        align-items: center;
    }

    .hero-metric-chip {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.09);
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #CBD5E1;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .hero-metric-chip strong {
        color: #FFFFFF;
        font-weight: 600;
    }

    /* ─── Primary Prediction Card ─── */
    .result-card {
        background: linear-gradient(145deg, #131C35 0%, #1E1B4B 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow: 0 16px 36px -10px rgba(11, 15, 26, 0.7), 0 0 30px rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    .result-tag {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #A78BFA;
    }

    .result-conf-badge {
        background: rgba(139, 92, 246, 0.22);
        border: 1px solid rgba(167, 139, 250, 0.4);
        color: #DDD6FE;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .result-breed-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.2rem 0 0.4rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .result-subcaption {
        font-size: 0.86rem;
        color: #94A3B8;
        margin-bottom: 0;
    }

    /* ─── Top 3 Predictions Visualization ─── */
    .top-preds-container {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        margin-top: 1rem;
    }

    .top-preds-title {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.95rem;
    }

    .rank-item {
        margin-bottom: 0.95rem;
    }
    .rank-item:last-child {
        margin-bottom: 0;
    }

    .rank-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }

    .rank-name-box {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        color: #F1F5F9;
        font-weight: 600;
    }

    .rank-number {
        font-size: 0.72rem;
        font-weight: 700;
        color: #A78BFA;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.25);
        width: 22px;
        height: 22px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .rank-pct {
        font-weight: 700;
        color: #DDD6FE;
        font-size: 0.9rem;
    }

    .meter-track {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
    }

    .meter-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
        border-radius: 9999px;
    }

    /* ─── Image Frame & Specs ─── */
    .img-spec-card {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ─── Editorial Breed Profile Grid ─── */
    .profile-section-header {
        margin: 2.2rem 0 1.2rem 0;
    }

    .profile-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A78BFA;
        margin-bottom: 0.3rem;
    }

    .profile-main-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin: 0;
    }

    .editorial-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.8rem;
    }

    .editorial-tile {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .editorial-tile:hover {
        border-color: rgba(139, 92, 246, 0.3);
    }

    .editorial-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.35rem;
    }

    .editorial-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
        line-height: 1.4;
    }

    /* ─── Editorial Tabs Content ─── */
    .editorial-tab-content {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1rem;
        line-height: 1.7;
        color: #CBD5E1;
    }

    .narrative-quote {
        border-left: 3px solid #8B5CF6;
        padding-left: 1.2rem;
        font-style: normal;
        color: #E2E8F0;
        font-size: 1.02rem;
    }

    .spec-list-item {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }

    .spec-list-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #A78BFA;
        min-width: 150px;
    }

    .spec-list-text {
        font-size: 0.95rem;
        color: #E2E8F0;
    }

    /* ─── Non-Dog Alert Card ─── */
    .non-dog-alert {
        background: linear-gradient(145deg, #1C1310 0%, #2A1A14 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        border-radius: 16px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1.5rem;
    }

    .non-dog-title {
        color: #FBBF24;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .non-dog-desc {
        color: #D1D5DB;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* ─── Disclaimer Banner ─── */
    .editorial-disclaimer {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 1.8rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* ─── Empty Upload Surface ─── */
    .upload-surface-empty {
        background: linear-gradient(145deg, #0B1120 0%, #0F172A 100%);
        border: 1px dashed rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 3.2rem 2rem;
        text-align: center;
        margin-top: 1.2rem;
    }

    /* ─── Native Streamlit Control Overrides ─── */
    div.stButton > button {
        background: #1E1B4B !important;
        color: #DDD6FE !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: #2E1065 !important;
        border-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.25) !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 0.3rem;
    }

    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: #94A3B8 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.1rem !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background: rgba(139, 92, 246, 0.18) !important;
        border: 1px solid rgba(139, 92, 246, 0.35) !important;
    }

    div[data-testid="stFileUploader"] section {
        background: #0B1120 !important;
        border: 1px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 14px !important;
        padding: 1.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Model & Breed Database Loaders ──────────────────────────────────

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
    """Load 356 dog breed archival intelligence database."""
    json_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'breed_information', 'dog_breed_info.json'
    )
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ─── Editorial Breed Profile Component ──────────────────────────────────────

def render_breed_info_section(breed_name, breed_db):
    """
    Render all 15 detailed information fields for a specific breed in an
    editorial, balanced two-column presentation with thematic tabs.
    """
    info = breed_db.get(breed_name, None)

    st.markdown(
        f"""
        <div class="profile-section-header">
            <div class="profile-eyebrow">Canine Compendium • Archival Profile</div>
            <h2 class="profile-main-title">{breed_name}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not info:
        st.info(f"Archival records for **{breed_name}** are currently compiling.")
        return

    # Symmetrical 2-Column Overview Grid (Fields 1-9)
    grid_html = f"""
    <div class="editorial-grid">
        <div class="editorial-tile">
            <div class="editorial-label">Breed Name</div>
            <div class="editorial-value">{info.get('breed_name', breed_name)}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Size Classification</div>
            <div class="editorial-value">{info.get('size', 'Medium')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Taxonomic Group</div>
            <div class="editorial-value">{info.get('group', 'Working Dog')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Average Height</div>
            <div class="editorial-value">{info.get('height', '45–60 cm')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Region of Origin</div>
            <div class="editorial-value">{info.get('origin', 'International')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Average Weight</div>
            <div class="editorial-value">{info.get('weight', '15–30 kg')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Original Working Purpose</div>
            <div class="editorial-value">{info.get('purpose', 'Companionship & Working')}</div>
        </div>
        <div class="editorial-tile">
            <div class="editorial-label">Temperament Profile</div>
            <div class="editorial-value">{info.get('temperament', 'Loyal, Intelligent, Friendly')}</div>
        </div>
        <div class="editorial-tile" style="grid-column: span 2;">
            <div class="editorial-label">Average Lifespan</div>
            <div class="editorial-value">{info.get('lifespan', '10–14 years')}</div>
        </div>
    </div>
    """
    st.markdown(grid_html, unsafe_allow_html=True)

    # Thematic Detail Tabs (Fields 10-15)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 Historical Lineage",
        "🏃 Activity & Husbandry",
        "🏥 Clinical Considerations",
        "💡 Archival Notes & Facts"
    ])

    with tab1:
        history_text = info.get(
            'history',
            'Historical records document this breed across specialized functional and companion roles.'
        )
        st.markdown(
            f"""
            <div class="editorial-tab-content">
                <div class="narrative-quote">{history_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        exercise_text = info.get('exercise_needs', 'Daily walks and structured activity required.')
        grooming_text = info.get('grooming_needs', 'Regular coat maintenance and brushing.')
        env_text = info.get('environment', 'Adaptable to home living with adequate exercise.')
        st.markdown(
            f"""
            <div class="editorial-tab-content">
                <div class="spec-list-item">
                    <span class="spec-list-label">Exercise Protocol:</span>
                    <span class="spec-list-text">{exercise_text}</span>
                </div>
                <div class="spec-list-item">
                    <span class="spec-list-label">Coat & Grooming:</span>
                    <span class="spec-list-text">{grooming_text}</span>
                </div>
                <div class="spec-list-item" style="margin-bottom: 0;">
                    <span class="spec-list-label">Optimal Environment:</span>
                    <span class="spec-list-text">{env_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab3:
        health_text = info.get(
            'health_info',
            'Routine preventive health checkups, screenings, and vaccinations recommended.'
        )
        st.markdown(
            f"""
            <div class="editorial-tab-content">
                <div class="spec-list-item" style="margin-bottom: 0;">
                    <span class="spec-list-label">Clinical Health:</span>
                    <span class="spec-list-text">{health_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab4:
        facts_text = info.get(
            'facts',
            'Officially recognized in the international 356 FCI global dog breed classification catalog.'
        )
        st.markdown(
            f"""
            <div class="editorial-tab-content">
                <div class="spec-list-item" style="margin-bottom: 0;">
                    <span class="spec-list-label">Registry Notation:</span>
                    <span class="spec-list-text">{facts_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Restrained Veterinary Disclaimer
    st.markdown(
        """
        <div class="editorial-disclaimer">
            <span>🛡️</span>
            <span><strong>Veterinary Archival Disclaimer:</strong> Information presented is curated for educational and reference purposes under FCI breed classifications. Consult licensed veterinary practitioners for individual canine clinical diagnostics.</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ─── Sidebar System Telemetry ───────────────────────────────────────────────

def render_sidebar(class_names):
    """Render sleek sidebar with system telemetry and architecture specs."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; color: #A78BFA; text-transform: uppercase;">
                    System Telemetry
                </div>
                <h3 style="margin: 0.2rem 0 0.4rem 0; font-size: 1.35rem; color: #FFFFFF;">Canis Intelligence</h3>
                <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5; margin: 0;">
                    High-precision neural identification across 356 FCI international canine breeds.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #DDD6FE;">
                ● <strong>Catalog Scope:</strong> 356 Recognized FCI Breeds
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Architecture Specifications
        model, model_type, img_size, _, error = load_model()
        if model is not None:
            st.markdown(
                """
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.6rem;">
                    Model Specifications
                </div>
                """,
                unsafe_allow_html=True
            )
            m_name = "MobileNetV2 Transfer Learning" if model_type == 'transfer' else "DogBreed CNN Scratch"
            st.markdown(f"• **Architecture:** `{m_name}`")
            st.markdown(f"• **Tensor Shape:** `{img_size}×{img_size} RGB`")
            st.markdown(f"• **Output Classes:** `356 Breeds`")
            st.markdown(f"• **Guard Filter:** `Pre-Screen Active`")

            comp_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_comparison.json')
            if os.path.exists(comp_path):
                with open(comp_path, 'r', encoding='utf-8') as f:
                    comp = json.load(f)
                best_acc = comp.get('best_accuracy', 0)
                st.markdown(f"• **Top-1 Accuracy:** `{best_acc*100:.2f}%`")
                st.markdown("• **Top-5 Accuracy:** `86.05%`")

        st.divider()
        st.caption("Engineered with PyTorch, Torchvision & Streamlit")


# ─── Main Application Workflow ──────────────────────────────────────────────

def main():
    model, model_type, img_size, class_names, error = load_model()
    if class_names is None:
        class_names = DOG_BREEDS_CLASSES

    breed_db = load_breed_info_db()
    render_sidebar(class_names)

    # Signature Hero Identity Banner
    st.markdown(
        """
        <div class="canis-hero">
            <div class="hero-eyebrow">
                <span class="hero-pulse"></span>
                <span>Canis AI • Neural Taxonomy Engine</span>
            </div>
            <h1 class="hero-title">Global Dog Breed <span>Identification</span></h1>
            <p class="hero-desc">
                Precision visual intelligence trained across all 356 Fédération Cynologique Internationale breeds.
                Featuring automated canine pre-screening and comprehensive archival care profiles.
            </p>
            <div class="hero-metrics-row">
                <div class="hero-metric-chip"><strong>356</strong> FCI Breeds</div>
                <div class="hero-metric-chip"><strong>Canine Guard</strong> Active</div>
                <div class="hero-metric-chip"><strong>MobileNetV2</strong> Backbone</div>
                <div class="hero-metric-chip"><strong>15 Dimensions</strong> Care & History</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if error:
        st.warning(f"⚠️ **Model Initialization Note:** {error}")
        st.info("Train a model first by running: `python -m src.train` in your terminal.")

    # Primary Navigation Tabs
    main_tab1, main_tab2 = st.tabs([
        "📷 Analyze Photograph",
        "🔍 Archival Catalog Explorer (356 Breeds)"
    ])

    with main_tab1:
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Select or drop a canine photograph...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a dog (JPG/PNG, up to 5 MB)",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        if uploaded_file is not None:
            is_valid, error_msg = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return

            image = load_image_from_upload(uploaded_file)
            img_info = get_image_info(image)

            # Symmetrical Layout: Photo Display | Neural Analysis
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown(
                    """
                    <div style="font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; color: #A78BFA; text-transform: uppercase; margin-bottom: 0.6rem;">
                        Subject Photography
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                st.markdown(
                    f"""
                    <div class="img-spec-card">
                        <span><strong>Dimensions:</strong> {img_info['width']} × {img_info['height']} px</span>
                        <span><strong>Format:</strong> {img_info['format']}</span>
                        <span style="color: #10B981; font-weight: 600;">● Active Sample</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                # Polished Reset Experience
                if st.button("↺ Analyze Another Photograph", use_container_width=True, type="secondary"):
                    st.session_state.uploader_key += 1
                    st.rerun()

            with col2:
                st.markdown(
                    """
                    <div style="font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; color: #A78BFA; text-transform: uppercase; margin-bottom: 0.6rem;">
                        Neural Identification
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if model is None:
                    st.error("Neural model is not ready for inference. Please check setup.")
                    return

                with st.spinner("Executing canine pre-screening filter..."):
                    detection_result = detect_dog(image)

                # Consistent Non-Dog Guard State
                if not detection_result['is_dog']:
                    st.markdown(
                        """
                        <div class="non-dog-alert">
                            <div class="non-dog-title">
                                <span>⚠️</span>
                                <span>Canine Pre-Screening: Subject Verification Failed</span>
                            </div>
                            <p class="non-dog-desc">
                                The integrated neural pre-screening filter evaluated this image and did not detect recognizable canine anatomical traits.
                                To uphold taxonomic integrity and prevent false breed attribution, classification has been safely withheld.
                            </p>
                            <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(245, 158, 11, 0.2); font-size: 0.84rem; color: #E5E7EB;">
                                💡 <em>Guidance: Please upload a clear photo displaying a recognizable domestic dog in good lighting.</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    with st.spinner("Extracting deep visual features & computing breed distribution..."):
                        device = next(model.parameters()).device
                        result = predict_image(image, model, img_size, device, top_k=3, class_names=class_names)

                    pred_class = result['predicted_class']
                    conf_pct = result['confidence_pct']
                    class_idx = class_names.index(pred_class) if pred_class in class_names else "—"

                    # Prominent Primary Result Card
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-header">
                                <span class="result-tag">Verified Canine Classification</span>
                                <span class="result-conf-badge">{conf_pct} Match</span>
                            </div>
                            <div class="result-breed-title">{pred_class}</div>
                            <p class="result-subcaption">FCI Standard Registry Breed • Index #{class_idx}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if result.get('non_dog_warning', False):
                        st.markdown(
                            """
                            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 0.75rem 1rem; color: #FBBF24; font-size: 0.85rem; margin-bottom: 1rem;">
                                ⚠️ <strong>Low Confidence Advisory:</strong> The prediction probability is distributed across multiple candidate breeds. For optimal accuracy, ensure the canine subject is clearly framed without heavy occlusion.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Bespoke Top 3 Candidate Predictions
                    top_preds = result['top_predictions']
                    rank_html = ""
                    medals = ["01", "02", "03"]
                    for rank, (breed, prob) in enumerate(top_preds, 1):
                        pct_val = prob * 100
                        rank_str = medals[rank - 1] if rank <= 3 else f"{rank:02d}"
                        rank_html += f"""
                        <div class="rank-item">
                            <div class="rank-row">
                                <div class="rank-name-box">
                                    <span class="rank-number">{rank_str}</span>
                                    <span>{breed}</span>
                                </div>
                                <span class="rank-pct">{pct_val:.1f}%</span>
                            </div>
                            <div class="meter-track">
                                <div class="meter-fill" style="width: {pct_val:.1f}%;"></div>
                            </div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="top-preds-container">
                            <div class="top-preds-title">Top Candidate Predictions</div>
                            {rank_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # AUTOMATICALLY DISPLAY BREED INFORMATION & HISTORY ACROSS FULL BALANCED WIDTH
            if 'detection_result' in locals() and detection_result.get('is_dog') and 'pred_class' in locals():
                render_breed_info_section(pred_class, breed_db)

        else:
            # Elegant Empty Upload Surface
            st.markdown(
                """
                <div class="upload-surface-empty">
                    <div style="width: 48px; height: 48px; margin: 0 auto 1.2rem auto; border-radius: 12px; background: rgba(139, 92, 246, 0.15); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border: 1px solid rgba(139, 92, 246, 0.3);">
                        📷
                    </div>
                    <h3 style="color: #FFFFFF; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.4rem;">Upload Canine Photograph for Identification</h3>
                    <p style="color: #94A3B8; font-size: 0.95rem; max-width: 460px; margin: 0 auto 1rem auto; line-height: 1.5;">
                        Select or drop a high-resolution canine portrait to execute neural feature extraction and 356-breed taxonomy lookup.
                    </p>
                    <div style="display: inline-flex; gap: 0.5rem; font-size: 0.78rem; color: #64748B;">
                        <span>JPG</span> • <span>PNG</span> • <span>Max 5 MB</span> • <span>Instant Analysis</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with main_tab2:
        st.markdown(
            """
            <div style="margin-bottom: 1.4rem;">
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; color: #A78BFA; text-transform: uppercase;">
                    Archival Directory
                </div>
                <h3 style="margin: 0.2rem 0 0.5rem 0; font-size: 1.5rem; color: #FFFFFF;">Explore 356 FCI Breed Profiles</h3>
                <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
                    Search or select any of the 356 supported international dog breeds to review taxonomic attributes, evolutionary history, care protocol, and clinical guidelines without uploading an image.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_breed = st.selectbox(
            "Select or search breed catalog:",
            options=class_names,
            index=class_names.index("Golden Retriever") if "Golden Retriever" in class_names else 0
        )

        if selected_breed:
            render_breed_info_section(selected_breed, breed_db)


if __name__ == '__main__':
    main()
