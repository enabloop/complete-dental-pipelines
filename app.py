import io
from pathlib import Path

import cv2
import numpy as np
import pydicom
import streamlit st
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from medpy.filter.smoothing import anisotropic_diffusion  # <--- MedPy εισαγωγή

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
    # Multi-frame DICOM (Διατήρηση 2D δομής για αποφυγή crash)
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

def apply_anisotropic_diffusion(image, niter=10, kappa=50):
    """
    Apply Anisotropic Diffusion for edge-preserving denoising.
    """
    img_float = image.astype(np.float32)
    img_diffused = anisotropic_diffusion(img_float, niter=niter, kappa=kappa, voxelspacing=None, option=1)
    return np.clip(img_diffused, 0, 255).astype(np.uint8)


# ============================================================
# STREAMLIT UI & ORIGINAL DUAL-COLUMN LAYOUT
# ============================================================

st.sidebar.header("📂 Εισαγωγή Αρχείου")
uploaded_file = st.sidebar.file_uploader(
    "Επιλέξτε ακτινογραφία (DICOM, PNG, JPG)...", 
    type=["dcm", "png", "jpg", "jpeg"]
)

# Sidebar Παράμετροι Αλγορίθμων (Όπως στο αρχικό UI)
st.sidebar.header("🎛️ Παράμετροι Επεξεργασίας")

st.sidebar.subheader("Ρυθμίσεις CLAHE")
clip_limit = st.sidebar.slider("Clip Limit", 0.5, 10.0, 2.0, 0.5)
tile_size = st.sidebar.slider("Tile Size", 2, 32, 8, 2)

st.sidebar.subheader("Ρυθμίσεις Anisotropic Diffusion")
niter = st.sidebar.slider("Iterations (Επαναλήψεις)", 1, 30, 10, 1)
kappa = st.sidebar.slider("Kappa (Όριο Ακμών)", 10, 100, 50, 5)

if uploaded_file is not None:
    # Φόρτωση εικόνας
    image, metadata = load_image(uploaded_file)
    
    # Εκτέλεση επεξεργασίας στο παρασκήνιο
    img_clahe = apply_clahe(image, clip_limit=clip_limit, tile_size=tile_size)
    img_denoised = apply_anisotropic_diffusion(image, niter=niter, kappa=kappa)
    
    # --- ΠΡΟΒΟΛΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (ORIGINAL GRID) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Αρχική Εικόνα")
        st.image(image, use_container_width=True)
        
        st.subheader("⚡ 1. CLAHE")
        st.image(img_clahe, use_container_width=True)

    with col2:
        st.subheader("🛡️ 2. Anisotropic Diffusion")
        st.image(img_denoised, use_container_width=True)
        st.caption("Έξυπνο φιλτράρισμα θορύβου. Καθαρίζει την εικόνα κρατώντας τις ακμές των δοντιών 100% κοφτερές.")

    # Εμφάνιση DICOM Metadata
    if metadata is not None:
        st.subheader("📋 DICOM Metadata")
        st.text(f"Patient Name: {getattr(metadata, 'PatientName', 'N/A')}")
        st.text(f"Modality: {getattr(metadata, 'Modality', 'N/A')}")
        st.text(f"Photometric Interpretation: {getattr(metadata, 'PhotometricInterpretation', 'N/A')}")

else:
    st.info("💡 Παρακαλώ ανεβάστε ένα αρχείο ακτινογραφίας από το αριστερό μενού για να ξεκινήσει η επεξεργασία.")
