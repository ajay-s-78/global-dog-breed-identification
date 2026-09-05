"""
Utility functions for the Dog Breed Identification Streamlit Application.
Handles file validation, image loading, and non-dog image warnings.
"""

import os
from PIL import Image

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_uploaded_file(uploaded_file):
    """
    Validate uploaded image file type, size, and integrity.
    """
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = uploaded_file.name.lower()
    extension = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type (.{extension}). Please upload a JPG or PNG image."

    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f} MB). Max file size is {MAX_FILE_SIZE_MB} MB."

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
    except Exception:
        return False, "Corrupted image file. Please upload a valid image."

    return True, None


def load_image_from_upload(uploaded_file):
    """Load PIL Image in RGB format."""
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def get_image_info(image):
    """Get dimensions and metadata for display."""
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': getattr(image, 'format', 'RGB'),
    }
