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
# X-RAY CLAHE
# ============================================================

def apply_xray_clahe(image, clip_limit=2.0, tile_size=8):
    """
    X-Ray CLAHE adapted as a separate enhancement option.

    Keeps the app's original OpenCV CLAHE untouched. This implementation
    performs CLAHE on a normalized 8-bit grayscale image and uses the
    tile/grid terminology of the X-ray enhancement approach.
    """
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)

    tile_size = max(2, int(tile_size))
    clip_limit = max(0.01, float(clip_limit))

    # Work on a float image normalized to [0, 1], then apply CLAHE.
    # The separate function allows X-Ray CLAHE to evolve independently
    # from the app's original CLAHE.
    normalized = image.astype(np.float32) / 255.0
    normalized = np.clip(normalized, 0.0, 1.0)

    xray_input = np.round(normalized * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    return clahe.apply(xray_input)


def xray_clahe_pipeline(image, clip_limit=2.0, tile_size=8):
    """Dedicated X-Ray CLAHE enhancement."""
    return apply_xray_clahe(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
    )


# ===========================================
