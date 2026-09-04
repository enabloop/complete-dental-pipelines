import io
from pathlib import Path

import cv2
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dental Image Processor",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 Dental Image Processor")

st.caption(
    "Επεξεργασία και βελτίωση οδοντιατρικών ακτινογραφιών"
)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_to_uint8(array):
    """
    Convert grayscale image data to 8-bit.

    Handles:
    - uint8
    - uint16
    - floating-point images
    - DICOM pixel data
    - NaN / Inf values
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(array)

    if not np.any(finite):
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    minimum = np.min(array[finite])
    maximum = np.max(array[finite])

    if maximum <= minimum:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = (
        (array - minimum)
        / (maximum - minimum)
        * 255.0
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# DICOM LOADING
# ============================================================

def load_dicom(file):
    """Load a DICOM image and return an 8-bit grayscale image."""

    ds = pydicom.dcmread(
        file,
        force=True,
    )

    # --------------------------------------------------------
    # Read pixel data
    # --------------------------------------------------------

    try:
        image = ds.pixel_array.astype(
            np.float32
        )

    except Exception as error:
        raise RuntimeError(
            "This DICOM contains compressed pixel data "
            "that could not be decoded.\n\n"
            "Additional DICOM decoding libraries may "
            "be required.\n\n"
            f"Original error:\n{error}"
        ) from error

    # --------------------------------------------------------
    # Multi-frame DICOM
    # --------------------------------------------------------

    if image.ndim > 2:
        image = image[0]

    # --------------------------------------------------------
    # Rescale slope / intercept
    # --------------------------------------------------------

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0,
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0,
        )
    )

    image = (
        image * slope
        + intercept
    )

    # --------------------------------------------------------
    # MONOCHROME1 inversion
    # --------------------------------------------------------

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )
    )

    if photometric.upper() == "MONOCHROME1":
        image = (
            np.max(image)
            - image
        )

    # --------------------------------------------------------
    # DICOM window center / width
    # --------------------------------------------------------

    window_center = getattr(
        ds,
        "WindowCenter",
        None,
    )

    window_width = getattr(
        ds,
        "WindowWidth",
        None,
    )

    if (
        window_center is not None
        and window_width is not None
    ):
        try:
            if hasattr(
                window_center,
                "__len__",
            ):
                center = float(
                    window_center[0]
                )
            else:
                center = float(
                    window_center
                )

            if hasattr(
                window_width,
                "__len__",
            ):
                width_value = float(
                    window_width[0]
                )
            else:
                width_value = float(
                    window_width
                )

            if width_value > 1:
                low = (
                    center
                    - width_value / 2.0
                )

                high = (
                    center
                    + width_value / 2.0
                )

                image = np.clip(
                    image,
                    low,
                    high,
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------

    image = normalize_to_uint8(
        image
    )

    return image, ds


# ============================================================
# REGULAR IMAGE LOADING
# ============================================================

def load_regular_image(file):
    """Load PNG/JPEG/TIFF/BMP as grayscale."""

    data = file.read()

    image = Image.open(
        io.BytesIO(data)
    ).convert("L")

    return (
        np.array(image),
        None,
    )


# ============================================================
# GENERAL IMAGE LOADING
# ============================================================

def load_image(file):
    """Automatically detect DICOM or regular image."""

    filename = file.name.lower()

    if filename.endswith(".dcm"):
        return load_dicom(file)

    try:
        file.seek(0)

        return load_regular_image(
            file
        )

    except Exception:
        file.seek(0)

        return load_dicom(
            file
        )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image,
    clip_limit=2.0,
    tile_size=8,
):
    """Apply Contrast Limited Adaptive Histogram Equalization."""

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    tile_size = max(
        2,
        int(tile_size),
    )

    clahe = cv2.createCLAHE(
        clipLimit=float(
            clip_limit
        ),
        tileGridSize=(
            tile_size,
            tile_size,
        ),
    )

    return clahe.apply(
        image
    )


# ============================================================
# ANISOTROPIC DIFFUSION
# ============================================================

def apply_anisotropic_diffusion(
    image,
    n_iter=10,
    kappa=30,
    gamma=0.1,
    method="Exponential"
):
    """
    Perona-Malik Anisotropic Diffusion for edge-preserving smoothing.
    Implemented in pure NumPy to avoid external binary dependencies.
    """
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)

    img = image.astype(np.float32)

    for _ in range(int(n_iter)):
        # Calculate gradients in 4 directions (North, South, East, West)
        deltaN = np.roll(img, -1, axis=0) - img
        deltaS = np.roll(img, 1, axis=0) - img
        deltaE = np.roll(img, -1, axis=1) - img
        deltaW = np.roll(img, 1, axis=1) - img

        # Conduction coefficients calculation
        if method == "Exponential":
            cN = np.exp(-(deltaN / kappa) ** 2)
            cS = np.exp(-(deltaS / kappa) ** 2)
            cE = np.exp(-(deltaE / kappa) ** 2)
            cW = np.exp(-(deltaW / kappa) ** 2)
        else:  # Quadratic
            cN = 1.0 / (1.0 + (deltaN / kappa) ** 2)
            cS = 1.0 / (1.0 + (deltaS / kappa) ** 2)
            cE = 1.0 / (1.0 + (deltaE / kappa) ** 2)
            cW = 1.0 / (1.0 + (deltaW / kappa) ** 2)

        # Update image state
        img += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# TOTAL VARIATION DENOISING
# ============================================================

def apply_total_variation(image, weight=0.1):
    """
    Apply Total Variation Denoising (Chambolle) and return uint8 image.
    """
    # Chambolle algorithm performs better when input is in [0, 1] range
    img_float = image.astype(np.float32) / 255.0
    denoised = denoise_tv_chambolle(img_float, weight=weight)
    
    # Scale back to uint8 range
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# STREAMLIT RUNTIME & MENU SELECT
# ============================================================

# 1. File Uploader in sidebar
uploaded_file = st.sidebar.file_uploader(
    "Επιλέξτε αρχείο εικόνας (PNG, JPG, DCM)", 
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file is not None:
    # Load the source image
    original_img, _ = load_image(uploaded_file)
    
    # 2. Main drop-down menu selection
    enhancement_option = st.sidebar.selectbox(
        "Μέθοδος Βελτίωσης Εικόνας",
        ["Original", "CLAHE", "Anisotropic Diffusion", "Total Variation Denoising"]
    )
    
    # 3. Dynamic adjustment panels based on menu selection
    if enhancement_option == "CLAHE":
        clip = st.sidebar.slider("Clip Limit", 0.1, 10.0, 2.0, step=0.1)
        tile = st.sidebar.slider("Tile Size", 2, 32, 8, step=1)
        processed_img = apply_clahe(original_img, clip_limit=clip, tile_size=tile)
        
    elif enhancement_option == "Anisotropic Diffusion":
        iters = st.sidebar.slider("Iterations (Βήματα)", 1, 50, 15, step=1)
        
