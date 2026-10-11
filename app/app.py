"""
CANIS — The Art of Canine Intelligence.
Global Dog Breed Identification Using Deep Learning — Streamlit Web Application.

A cinematic, luxury-grade AI platform identifying 356 recognized dog breeds via
MobileNetV2 Transfer Learning with real-time canine pre-screening and a 15-dimension
botanical wildlife encyclopedia.
"""

import sys
import os
import json
import base64

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
    page_title="CANIS — The Art of Canine Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Botanical & Luxury Editorial Design System ─────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* ─── Color Tokens & Variables ─── */
    :root {
        --deep-forest: #102B24;
        --dark-evergreen: #183B30;
        --botanical-green: #315C48;
        --muted-sage: #A8B9A3;
        --warm-ivory: #F7F4EA;
        --soft-cream: #EDE8DA;
        --antique-gold: #C5A46D;
        --antique-gold-hover: #D4B67F;
        --deep-charcoal: #202923;
        --muted-text: #66736A;
        --serif-font: 'Cormorant Garamond', 'Playfair Display', Georgia, serif;
        --sans-font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* ─── Global Atmosphere ─── */
    .stApp {
        background-color: var(--deep-forest) !important;
        color: var(--warm-ivory) !important;
        font-family: var(--sans-font) !important;
    }

    /* Streamlit layout wrapper */
    .main .block-container {
        max-width: 1140px;
        padding-top: 1.5rem;
        padding-bottom: 4.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    /* ─── Custom Brand Header ─── */
    .canis-brand-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0 1.6rem 0;
        border-bottom: 1px solid rgba(197, 164, 109, 0.25);
        margin-bottom: 2rem;
    }

    .brand-logo-group {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .brand-emblem {
        width: 36px;
        height: 36px;
        color: var(--antique-gold);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .brand-title {
        font-family: var(--serif-font);
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--warm-ivory);
        margin: 0;
        line-height: 1;
    }

    .brand-subtitle {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--antique-gold);
        margin-top: 0.2rem;
    }

    .brand-nav-links {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted-sage);
    }

    .brand-nav-badge {
        background: rgba(197, 164, 109, 0.12);
        border: 1px solid rgba(197, 164, 109, 0.3);
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        color: var(--antique-gold);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* ─── Cinematic Hero Section ─── */
    .canis-hero-card {
        background: linear-gradient(135deg, var(--dark-evergreen) 0%, #12322A 100%);
        border: 1px solid rgba(197, 164, 109, 0.3);
        border-radius: 20px;
        padding: 2.8rem 2.6rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 20px 45px rgba(10, 26, 22, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--antique-gold);
        margin-bottom: 1rem;
    }

    .hero-eyebrow::before {
        content: "";
        width: 18px;
        height: 1px;
        background: var(--antique-gold);
    }

    .hero-headline {
        font-family: var(--serif-font);
        font-size: 3rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--warm-ivory) !important;
        line-height: 1.15 !important;
        margin: 0 0 1.1rem 0 !important;
    }

    .hero-headline em {
        font-style: italic;
        color: var(--antique-gold);
        font-family: var(--serif-font);
    }

    .hero-paragraph {
        font-size: 1.02rem;
        line-height: 1.68;
        color: var(--muted-sage);
        max-width: 580px;
        margin-bottom: 1.8rem;
    }

    .hero-facts-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        padding-top: 1.4rem;
        border-top: 1px solid rgba(168, 185, 163, 0.15);
    }

    .hero-fact-item {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }

    .hero-fact-value {
        font-family: var(--serif-font);
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--warm-ivory);
    }

    .hero-fact-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--antique-gold);
        font-weight: 600;
    }

    .hero-portrait-frame {
        background: var(--deep-forest);
        border: 1px solid rgba(197, 164, 109, 0.35);
        border-radius: 16px;
        overflow: hidden;
        padding: 0.75rem;
        box-shadow: 0 16px 36px rgba(10, 26, 22, 0.6);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .hero-portrait-frame img {
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 10px;
        display: block;
    }

    .hero-portrait-caption {
        font-family: var(--serif-font);
        font-style: italic;
        color: var(--antique-gold);
        font-size: 0.95rem;
        margin-top: 0.75rem;
        letter-spacing: 0.02em;
    }

    /* ─── Compact Feature Highlights ─── */
    .feature-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 2.4rem;
    }

    .feature-card {
        background: rgba(24, 59, 48, 0.5);
        border: 1px solid rgba(168, 185, 163, 0.18);
        border-radius: 14px;
        padding: 1.4rem 1.3rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .feature-card:hover {
        border-color: rgba(197, 164, 109, 0.45);
        transform: translateY(-2px);
    }

    .feature-card-icon {
        color: var(--antique-gold);
        font-size: 1.1rem;
        margin-bottom: 0.6rem;
    }

    .feature-card-title {
        font-family: var(--serif-font);
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--warm-ivory);
        margin-bottom: 0.35rem;
    }

    .feature-card-text {
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--muted-sage);
        margin: 0;
    }

    /* ─── Image Upload Workspace ─── */
    .workspace-section-header {
        margin-bottom: 1.2rem;
    }

    .section-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--antique-gold);
        margin-bottom: 0.3rem;
    }

    .section-headline {
        font-family: var(--serif-font);
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--warm-ivory);
        margin: 0 0 0.4rem 0;
    }

    .section-desc {
        color: var(--muted-sage);
        font-size: 0.95rem;
        max-width: 620px;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .upload-frame {
        background: rgba(24, 59, 48, 0.35);
        border: 1.5px dashed rgba(197, 164, 109, 0.45);
        border-radius: 16px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        transition: border-color 0.2s ease, background-color 0.2s ease;
        margin-bottom: 1.4rem;
    }

    .upload-frame:hover {
        border-color: var(--antique-gold);
        background: rgba(24, 59, 48, 0.5);
    }

    .upload-frame-icon {
        color: var(--antique-gold);
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
    }

    .upload-frame-title {
        font-family: var(--serif-font);
        font-size: 1.35rem;
        color: var(--warm-ivory);
        margin-bottom: 0.35rem;
    }

    .upload-frame-specs {
        font-size: 0.82rem;
        color: var(--muted-sage);
        letter-spacing: 0.04em;
    }

    /* ─── Cinematic Prediction Results Card ─── */
    .result-showcase {
        background: var(--warm-ivory);
        color: var(--deep-charcoal);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        box-shadow: 0 20px 45px rgba(10, 26, 22, 0.35);
        border: 1px solid var(--antique-gold);
        margin-bottom: 2rem;
    }

    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--botanical-green);
        background: var(--soft-cream);
        border: 1px solid rgba(49, 92, 72, 0.25);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
    }

    .result-breed-title {
        font-family: var(--serif-font);
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--deep-forest);
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }

    .result-estimate-text {
        font-size: 0.95rem;
        color: var(--muted-text);
        margin-bottom: 1.5rem;
    }

    .result-estimate-text strong {
        color: var(--botanical-green);
    }

    /* Meter track */
    .confidence-meter-container {
        margin-bottom: 1.6rem;
    }

    .confidence-meter-label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--deep-charcoal);
        margin-bottom: 0.4rem;
    }

    .confidence-meter-track {
        width: 100%;
        height: 8px;
        background: var(--soft-cream);
        border-radius: 9999px;
        overflow: hidden;
        border: 1px solid rgba(168, 185, 163, 0.4);
    }

    .confidence-meter-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--botanical-green) 0%, var(--antique-gold) 100%);
        border-radius: 9999px;
    }

    /* ─── Top 3 Alternative Candidates ─── */
    .top3-container {
        background: #FFFFFF;
        border: 1px solid var(--soft-cream);
        border-radius: 14px;
        padding: 1.4rem;
        margin-top: 1.2rem;
    }

    .top3-header-title {
        font-family: var(--serif-font);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--deep-forest);
        margin-bottom: 0.9rem;
    }

    .top3-row {
        margin-bottom: 0.85rem;
    }
    .top3-row:last-child {
        margin-bottom: 0;
    }

    .top3-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .top3-name-group {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 600;
        color: var(--deep-charcoal);
    }

    .top3-rank-tag {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--antique-gold);
        background: var(--soft-cream);
        border: 1px solid rgba(197, 164, 109, 0.3);
        width: 22px;
        height: 22px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .top3-pct-value {
        font-family: var(--serif-font);
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--botanical-green);
    }

    .top3-bar-track {
        width: 100%;
        height: 6px;
        background: var(--soft-cream);
        border-radius: 9999px;
        overflow: hidden;
    }

    .top3-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--botanical-green) 0%, var(--antique-gold) 100%);
        border-radius: 9999px;
    }

    /* ─── Botanical Breed Encyclopedia (Warm Ivory Surface) ─── */
    .encyclopedia-wrapper {
        background: var(--warm-ivory);
        color: var(--deep-charcoal);
        border: 1px solid rgba(197, 164, 109, 0.4);
        border-radius: 20px;
        padding: 2.6rem 2.4rem;
        box-shadow: 0 20px 45px rgba(10, 26, 22, 0.3);
        margin-top: 2rem;
    }

    .encyclopedia-header {
        border-bottom: 1px solid var(--soft-cream);
        padding-bottom: 1.4rem;
        margin-bottom: 1.8rem;
    }

    .encyclopedia-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--botanical-green);
        margin-bottom: 0.35rem;
    }

    .encyclopedia-title {
        font-family: var(--serif-font);
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--deep-forest);
        margin: 0;
    }

    .encyclopedia-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.8rem;
    }

    .encyclopedia-tile {
        background: #FFFFFF;
        border: 1px solid var(--soft-cream);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .encyclopedia-tile:hover {
        border-color: rgba(197, 164, 109, 0.6);
        box-shadow: 0 4px 12px rgba(32, 41, 35, 0.05);
    }

    .tile-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted-text);
        margin-bottom: 0.3rem;
    }

    .tile-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--deep-charcoal);
        line-height: 1.4;
    }

    /* Encyclopedia Tab Surfaces */
    .encyclopedia-tab-body {
        background: #FFFFFF;
        border: 1px solid var(--soft-cream);
        border-radius: 14px;
        padding: 1.8rem 1.9rem;
        margin-top: 1rem;
        line-height: 1.75;
        color: var(--deep-charcoal);
    }

    .narrative-block {
        font-family: var(--serif-font);
        font-size: 1.25rem;
        font-style: italic;
        line-height: 1.7;
        color: var(--deep-forest);
        border-left: 3px solid var(--antique-gold);
        padding-left: 1.4rem;
    }

    .editorial-spec-row {
        display: flex;
        gap: 1.2rem;
        margin-bottom: 1.1rem;
        align-items: baseline;
    }
    .editorial-spec-row:last-child {
        margin-bottom: 0;
    }

    .editorial-spec-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--botanical-green);
        min-width: 160px;
    }

    .editorial-spec-value {
        font-size: 0.96rem;
        color: var(--deep-charcoal);
        flex: 1;
    }

    /* Veterinary Educational Disclaimer */
    .botanical-disclaimer {
        background: var(--soft-cream);
        border: 1px solid rgba(197, 164, 109, 0.4);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        font-size: 0.82rem;
        color: var(--muted-text);
        margin-top: 1.8rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        line-height: 1.5;
    }

    /* ─── Non-Dog Pre-Screen Alert Card ─── */
    .non-dog-panel {
        background: var(--dark-evergreen);
        border: 1px solid rgba(197, 164, 109, 0.4);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 16px 36px rgba(10, 26, 22, 0.4);
    }

    .non-dog-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: var(--antique-gold);
        font-family: var(--serif-font);
        font-size: 1.45rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .non-dog-message {
        color: var(--warm-ivory);
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .non-dog-detail {
        color: var(--muted-sage);
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }

    .non-dog-action-tip {
        background: rgba(16, 43, 36, 0.5);
        border-left: 3px solid var(--antique-gold);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
        color: var(--warm-ivory);
    }

    /* ─── Streamlit Native Component Styling ─── */
    div.stButton > button {
        background: var(--antique-gold) !important;
        color: var(--deep-forest) !important;
        border: 1px solid var(--antique-gold) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.3rem !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: var(--antique-gold-hover) !important;
        border-color: var(--antique-gold-hover) !important;
        color: var(--deep-charcoal) !important;
        box-shadow: 0 4px 15px rgba(197, 164, 109, 0.3) !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.6rem;
        border-bottom: 1px solid rgba(197, 164, 109, 0.25) !important;
        padding-bottom: 0.4rem;
    }

    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: var(--muted-sage) !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.2rem !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }

    button[data-baseweb="tab"]:hover {
        color: var(--warm-ivory) !important;
        background: rgba(168, 185, 163, 0.08) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--deep-forest) !important;
        background: var(--antique-gold) !important;
        border: 1px solid var(--antique-gold) !important;
    }

    div[data-testid="stFileUploader"] section {
        background: var(--dark-evergreen) !important;
        border: 1.5px dashed rgba(197, 164, 109, 0.4) !important;
        border-radius: 14px !important;
        padding: 1.4rem !important;
    }

    div[data-baseweb="select"] {
        background-color: var(--dark-evergreen) !important;
        border-radius: 10px !important;
    }

    /* ─── Minimal Botanical Footer ─── */
    .canis-footer {
        text-align: center;
        padding-top: 3.5rem;
        margin-top: 3.5rem;
        border-top: 1px solid rgba(197, 164, 109, 0.2);
        color: var(--muted-sage);
        font-size: 0.84rem;
        line-height: 1.6;
    }

    .footer-brand {
        font-family: var(--serif-font);
        font-size: 1.15rem;
        color: var(--antique-gold);
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    /* Accessibility: Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }

    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .feature-strip {
            grid-template-columns: 1fr;
        }
        .encyclopedia-grid {
            grid-template-columns: 1fr;
        }
        .hero-headline {
            font-size: 2.2rem !important;
        }
        .editorial-spec-row {
            flex-direction: column;
            gap: 0.3rem;
        }
        .editorial-spec-label {
            min-width: auto;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Local Hero Image Loader with Fallback ──────────────────────────

def get_hero_image_tag():
    """
    Attempt to load a reliable local portrait from the dataset as hero visual.
    Provides an elegant inline botanical canine emblem fallback if not available.
    """
    sample_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'raw', 'dog_breeds',
        '128_golden_retriever', 'img_001.jpg'
    )
    if os.path.exists(sample_path):
        try:
            with open(sample_path, 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            return f"""
            <div class="hero-portrait-frame">
                <img src="data:image/jpeg;base64,{b64_data}" alt="Canis lupus familiaris" />
                <div class="hero-portrait-caption">Canis lupus familiaris — Archival Specimen</div>
            </div>
            """
        except Exception:
            pass

    # High-end botanical canine crest SVG fallback
    return """
    <div class="hero-portrait-frame" style="padding: 2.5rem 1.5rem;">
        <svg width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#C5A46D" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto 1.2rem auto;">
            <path d="M12 2L9 9H2l6 4.5L5.5 21 12 16.5 18.5 21 16 13.5 22 9h-7l-3-7z"/>
            <circle cx="12" cy="12" r="9" stroke="#A8B9A3" stroke-width="0.8" stroke-dasharray="2 3"/>
        </svg>
        <div style="font-family: var(--serif-font); font-size: 1.35rem; color: #F7F4EA; margin-bottom: 0.4rem;">Archival Canine Registry</div>
        <div style="font-size: 0.8rem; color: #A8B9A3; letter-spacing: 0.05em;">FCI INTERNATIONAL TAXONOMY</div>
    </div>
    """


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


# ─── Botanical Breed Encyclopedia Component ─────────────────────────────────

def render_breed_info_section(breed_name, breed_db):
    """
    Render all 15 detailed information fields for a specific breed in a
    wildlife encyclopedia format on a warm ivory surface.
    """
    info = breed_db.get(breed_name, None)

    st.markdown(
        f"""
        <div class="encyclopedia-wrapper">
            <div class="encyclopedia-header">
                <div class="encyclopedia-eyebrow">Botanical Canine Compendium • Monograph</div>
                <h2 class="encyclopedia-title">{breed_name}</h2>
            </div>
        """,
        unsafe_allow_html=True
    )

    if not info:
        st.info(f"Archival records for **{breed_name}** are currently being indexed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Two-Column Overview Grid (Fields 1-9)
    grid_html = f"""
    <div class="encyclopedia-grid">
        <div class="encyclopedia-tile">
            <div class="tile-label">Registered Breed Name</div>
            <div class="tile-value">{info.get('breed_name', breed_name)}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Size Category</div>
            <div class="tile-value">{info.get('size', 'Medium')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">FCI Classification Group</div>
            <div class="tile-value">{info.get('group', 'Working Dog')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Average Height Standard</div>
            <div class="tile-value">{info.get('height', '45–60 cm')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Country / Region of Origin</div>
            <div class="tile-value">{info.get('origin', 'International')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Average Weight Standard</div>
            <div class="tile-value">{info.get('weight', '15–30 kg')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Original Purpose & Role</div>
            <div class="tile-value">{info.get('purpose', 'Companionship & Working')}</div>
        </div>
        <div class="encyclopedia-tile">
            <div class="tile-label">Temperament Character</div>
            <div class="tile-value">{info.get('temperament', 'Loyal, Intelligent, Friendly')}</div>
        </div>
        <div class="encyclopedia-tile" style="grid-column: span 2;">
            <div class="tile-label">Average Lifespan Expectancy</div>
            <div class="tile-value">{info.get('lifespan', '10–14 years')}</div>
        </div>
    </div>
    """
    st.markdown(grid_html, unsafe_allow_html=True)

    # Thematic Detail Tabs (Fields 10-15)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 Historical Lineage",
        "🌿 Care & Lifestyle",
        "🏥 Health Profile",
        "💡 Archival Notes & Facts"
    ])

    with tab1:
        history_text = info.get(
            'history',
            'Historical records document this breed across specialized functional and companion roles.'
        )
        st.markdown(
            f"""
            <div class="encyclopedia-tab-body">
                <div class="narrative-block">"{history_text}"</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        exercise_text = info.get('exercise_needs', 'Daily walks and structured physical activity required.')
        grooming_text = info.get('grooming_needs', 'Regular coat maintenance and brushing.')
        env_text = info.get('environment', 'Adaptable to domestic home living with adequate exercise.')
        st.markdown(
            f"""
            <div class="encyclopedia-tab-body">
                <div class="editorial-spec-row">
                    <span class="editorial-spec-label">Exercise Protocol:</span>
                    <span class="editorial-spec-value">{exercise_text}</span>
                </div>
                <div class="editorial-spec-row">
                    <span class="editorial-spec-label">Grooming & Coat:</span>
                    <span class="editorial-spec-value">{grooming_text}</span>
                </div>
                <div class="editorial-spec-row">
                    <span class="editorial-spec-label">Living Environment:</span>
                    <span class="editorial-spec-value">{env_text}</span>
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
            <div class="encyclopedia-tab-body">
                <div class="editorial-spec-row">
                    <span class="editorial-spec-label">Clinical Health Notes:</span>
                    <span class="editorial-spec-value">{health_text}</span>
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
            <div class="encyclopedia-tab-body">
                <div class="editorial-spec-row">
                    <span class="editorial-spec-label">Archival Registry Fact:</span>
                    <span class="editorial-spec-value">{facts_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Restrained Veterinary Disclaimer
    st.markdown(
        """
        <div class="botanical-disclaimer">
            <span>🛡️</span>
            <span><strong>Educational Disclaimer:</strong> Archival care, genetic, and breed records are provided solely for general educational and reference purposes. Consult licensed veterinary practitioners for individual canine medical examinations and health care guidance.</span>
        </div>
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
            <div style="padding: 0.6rem 0 1.2rem 0; border-bottom: 1px solid rgba(197, 164, 109, 0.25);">
                <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; color: #C5A46D; text-transform: uppercase;">
                    System Telemetry
                </div>
                <h3 style="margin: 0.25rem 0 0.3rem 0; font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; color: #F7F4EA;">
                    CANIS AI
                </h3>
                <p style="color: #A8B9A3; font-size: 0.85rem; line-height: 1.5; margin: 0;">
                    Deep learning vision system covering 356 international canine lineages.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="background: rgba(24, 59, 48, 0.6); border: 1px solid rgba(197, 164, 109, 0.3); border-radius: 8px; padding: 0.65rem 0.85rem; font-size: 0.8rem; color: #F7F4EA; margin-top: 1rem;">
                🌿 <strong>Taxonomic Scope:</strong> 356 FCI Breeds
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
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; color: #C5A46D; text-transform: uppercase; margin-bottom: 0.6rem;">
                    Neural Architecture
                </div>
                """,
                unsafe_allow_html=True
            )
            m_name = "MobileNetV2 Transfer" if model_type == 'transfer' else "Custom CNN"
            st.markdown(f"• **Backbone:** `{m_name}`")
            st.markdown(f"• **Resolution:** `{img_size}×{img_size} RGB`")
            st.markdown(f"• **Output Classes:** `356 Lineages`")
            st.markdown(f"• **Guard Filter:** `Pre-Screen Active`")

            comp_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_comparison.json')
            if os.path.exists(comp_path):
                with open(comp_path, 'r', encoding='utf-8') as f:
                    comp = json.load(f)
                best_acc = comp.get('best_accuracy', 0)
                st.markdown(f"• **Top-1 Accuracy:** `{best_acc*100:.2f}%`")
                st.markdown("• **Top-5 Accuracy:** `86.05%`")

        st.divider()
        st.caption("CANIS • The Art of Canine Intelligence")


# ─── Main Application Workflow ──────────────────────────────────────────────

def main():
    model, model_type, img_size, class_names, error = load_model()
    if class_names is None:
        class_names = DOG_BREEDS_CLASSES

    breed_db = load_breed_info_db()
    render_sidebar(class_names)

    # 1. Custom Brand Header
    st.markdown(
        """
        <div class="canis-brand-nav">
            <div class="brand-logo-group">
                <div class="brand-emblem">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#C5A46D" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>
                </div>
                <div>
                    <h1 class="brand-title">CANIS</h1>
                    <div class="brand-subtitle">GLOBAL BREED IDENTIFICATION</div>
                </div>
            </div>
            <div class="brand-nav-links">
                <span class="brand-nav-badge">356 FCI Breeds</span>
                <span>Canine Pre-Screen Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Cinematic Hero Section (Two-Column Editorial)
    hero_portrait_html = get_hero_image_tag()
    col_hero_text, col_hero_img = st.columns([1.25, 0.85], gap="large")

    with col_hero_text:
        st.markdown(
            f"""
            <div class="canis-hero-card" style="margin-bottom: 0;">
                <div class="hero-eyebrow">CANIS • CANINE INTELLIGENCE</div>
                <h1 class="hero-headline">Every Breed Has <em>a Story.</em></h1>
                <p class="hero-paragraph">
                    Explore the rich heritage of domestic canines through deep learning.
                    Our vision system analyzes fine-grained visual characteristics to identify likely breeds
                    across 356 recognized lineages and reveal their evolutionary history.
                </p>
                <div class="hero-facts-row">
                    <div class="hero-fact-item">
                        <span class="hero-fact-value">356</span>
                        <span class="hero-fact-label">Breed Classes</span>
                    </div>
                    <div class="hero-fact-item">
                        <span class="hero-fact-value">MobileNetV2</span>
                        <span class="hero-fact-label">Deep Learning</span>
                    </div>
                    <div class="hero-fact-item">
                        <span class="hero-fact-value">15 Fields</span>
                        <span class="hero-fact-label">Archival Knowledge</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_hero_img:
        st.markdown(hero_portrait_html, unsafe_allow_html=True)

    st.write("")

    # 3. Compact Feature Highlights
    st.markdown(
        """
        <div class="feature-strip">
            <div class="feature-card">
                <div class="feature-card-icon">🛡️</div>
                <div class="feature-card-title">Canine Pre-Screen Guard</div>
                <p class="feature-card-text">
                    Automated vision screening verifies canine subjects before classification, rejecting non-dog uploads.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-card-icon">🧠</div>
                <div class="feature-card-title">Transfer Learning Engine</div>
                <p class="feature-card-text">
                    Fine-tuned on MobileNetV2 architecture with 72.47% Top-1 and 86.05% Top-5 accuracy across 356 breeds.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-card-icon">📖</div>
                <div class="feature-card-title">Botanical Encyclopedia</div>
                <p class="feature-card-text">
                    15 archival dimensions covering origin, lifespan, working purpose, temperament, care, and health.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if error:
        st.warning(f"⚠️ **Model Initialization Note:** {error}")
        st.info("Train a model first by running: `python -m src.train` in your terminal.")

    # 4. Primary Navigation Tabs
    main_tab1, main_tab2 = st.tabs([
        "📷 Identify a Breed from Photograph",
        "🔍 Explore Breed Knowledge (356 Catalog)"
    ])

    with main_tab1:
        st.markdown(
            """
            <div class="workspace-section-header">
                <div class="section-eyebrow">Photograph Analysis Workspace</div>
                <h2 class="section-headline">Begin with a photograph</h2>
                <p class="section-desc">
                    Upload a clear photograph of a dog to initiate pre-screening neural analysis and fine-grained breed classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Select or drop a canine photograph...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a dog (JPG, JPEG, or PNG, max 5 MB)",
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
            col_photo, col_analysis = st.columns([1, 1], gap="large")

            with col_photo:
                st.markdown(
                    """
                    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em; color: #C5A46D; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Uploaded Photograph
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                st.markdown(
                    f"""
                    <div style="background: rgba(24, 59, 48, 0.4); border: 1px solid rgba(168, 185, 163, 0.2); border-radius: 10px; padding: 0.75rem 1rem; margin-top: 0.8rem; font-size: 0.82rem; color: #A8B9A3; display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>Dimensions:</strong> {img_info['width']} × {img_info['height']} px</span>
                        <span><strong>Format:</strong> {img_info['format']}</span>
                        <span style="color: #C5A46D; font-weight: 600;">● Active Sample</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")
                # Clean Reset Action
                if st.button("↺ Try Another Image", use_container_width=True, type="secondary"):
                    st.session_state.uploader_key += 1
                    st.rerun()

            with col_analysis:
                st.markdown(
                    """
                    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em; color: #C5A46D; text-transform: uppercase; margin-bottom: 0.5rem;">
                        Neural Vision Analysis
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if model is None:
                    st.error("Neural model is not ready for inference. Please check setup.")
                    return

                with st.spinner("Pre-screening image for canine verification..."):
                    detection_result = detect_dog(image)

                # Guard: Non-Dog Rejection (Preserved Exactly)
                if not detection_result['is_dog']:
                    st.markdown(
                        """
                        <div class="non-dog-panel">
                            <div class="non-dog-header">
                                <span>⚠️</span>
                                <span>Subject Screening: No Canine Detected</span>
                            </div>
                            <div class="non-dog-message">
                                No dog detected. Please upload a clear image of a dog.
                            </div>
                            <div class="non-dog-detail">
                                The automated neural pre-screening filter did not detect recognizable canine traits in this photograph.
                                To uphold taxonomic accuracy and prevent false breed attribution, classification has been safely withheld.
                            </div>
                            <div class="non-dog-action-tip">
                                💡 <strong>Guidance:</strong> Ensure your image features an unobstructed view of a dog in clear natural lighting.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    with st.spinner("Analyzing deep visual features across 356 breeds..."):
                        device = next(model.parameters()).device
                        result = predict_image(image, model, img_size, device, top_k=3, class_names=class_names)

                    pred_class = result['predicted_class']
                    conf_pct = result['confidence_pct']
                    raw_conf = result.get('confidence', 0.0)

                    # 5. Cinematic Prediction Results Card
                    st.markdown(
                        f"""
                        <div class="result-showcase">
                            <div class="result-badge">
                                <span>🌿</span>
                                <span>Identification Result</span>
                            </div>
                            <h2 class="result-breed-title">{pred_class}</h2>
                            <div class="result-estimate-text">
                                Model Estimate: <strong>{conf_pct} confidence</strong> (uncalibrated Softmax probability)
                            </div>
                            <div class="confidence-meter-container">
                                <div class="confidence-meter-label-row">
                                    <span>Relative Confidence Index</span>
                                    <span>{conf_pct}</span>
                                </div>
                                <div class="confidence-meter-track">
                                    <div class="confidence-meter-fill" style="width: {raw_conf*100:.1f}%;"></div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if result.get('non_dog_warning', False):
                        st.warning("⚠️ **Low Confidence Advisory:** Model confidence is distributed across multiple candidate breeds. Ensure the subject is clearly framed.")

                    # Top 3 Candidates
                    top_preds = result['top_predictions']
                    rank_html = ""
                    medals = ["01", "02", "03"]
                    for rank, (breed, prob) in enumerate(top_preds, 1):
                        pct_val = prob * 100
                        rank_str = medals[rank - 1] if rank <= 3 else f"{rank:02d}"
                        rank_html += f"""
                        <div class="top3-row">
                            <div class="top3-meta-row">
                                <div class="top3-name-group">
                                    <span class="top3-rank-tag">{rank_str}</span>
                                    <span>{breed}</span>
                                </div>
                                <span class="top3-pct-value">{pct_val:.1f}%</span>
                            </div>
                            <div class="top3-bar-track">
                                <div class="top3-bar-fill" style="width: {pct_val:.1f}%;"></div>
                            </div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="top3-container">
                            <div class="top3-header-title">Top Candidate Predictions</div>
                            {rank_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 6. Detailed Breed Encyclopedia (Rendered on Warm Ivory Surface)
            if 'detection_result' in locals() and detection_result.get('is_dog') and 'pred_class' in locals():
                render_breed_info_section(pred_class, breed_db)

        else:
            # Elegant Empty Upload Surface
            st.markdown(
                """
                <div class="upload-frame">
                    <div class="upload-frame-icon">🌿</div>
                    <div class="upload-frame-title">Select or Drop a Canine Photograph</div>
                    <div class="upload-frame-specs">
                        Supported formats: <strong>JPG, JPEG, PNG</strong> • Maximum size: <strong>5 MB</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with main_tab2:
        st.markdown(
            """
            <div class="workspace-section-header">
                <div class="section-eyebrow">Archival Knowledge Repository</div>
                <h2 class="section-headline">Explore Breed Knowledge</h2>
                <p class="section-desc">
                    Search or select any of the 356 recognized international dog breeds to review taxonomic attributes,
                    historical development, lifestyle requirements, and health guidance without uploading an image.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected_breed = st.selectbox(
            "Search or select a dog breed:",
            options=class_names,
            index=class_names.index("Golden Retriever") if "Golden Retriever" in class_names else 0
        )

        if selected_breed:
            render_breed_info_section(selected_breed, breed_db)

    # 8. Minimal Botanical Footer
    st.markdown(
        """
        <div class="canis-footer">
            <div class="footer-brand">CANIS — The Art of Canine Intelligence</div>
            <div>Global Dog Breed Identification Using Deep Learning • 356 FCI Breed Classes</div>
            <div style="font-size: 0.75rem; color: #66736A; margin-top: 0.4rem;">
                Built with PyTorch, MobileNetV2 & Streamlit • Botanical Wildlife Editorial Design System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
