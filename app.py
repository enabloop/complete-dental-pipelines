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


# ============================================================
# UNSHARP MASKING
# ============================================================

def apply_usm(
    image,
    sigma=1.0,
    amount=1.0,
    threshold=5.0,
):
    """Standard Unsharp Masking."""

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    image_float = image.astype(
        np.float32
    )

    blurred_float = blurred.astype(
        np.float32
    )

    detail = (
        image_float
        - blurred_float
    )

    threshold = float(
        threshold
    )

    if threshold > 0:
        detail[
            np.abs(detail)
            < threshold
        ] = 0.0

    sharpened = (
        image_float
        + float(amount)
        * detail
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# STANDARD CONVOLUTION MASKING
# ============================================================

def apply_standard_masking(
    image,
    strength=1.0,
):
    """Apply a controlled sharpening convolution."""

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    sharpening_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    identity = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    identity[1, 1] = 1.0

    kernel = (
        identity
        + float(strength)
        * (
            sharpening_kernel
            - identity
        )
    )

    result = cv2.filter2D(
        image,
        -1,
        kernel,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# MEDIAN FILTER
# ============================================================

def apply_median_filter(
    image,
    kernel_size=3,
):
    """Median filtering for noise reduction."""

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    kernel_size = int(
        kernel_size
    )

    if kernel_size < 3:
        kernel_size = 3

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(
        image,
        kernel_size,
    )


# ============================================================
# TV-CLAHE / ENDO SHARP (FIXED BASED ON MDPI 14(15):5554)
# ============================================================

def tv_clahe(image):
    """
    TV-CLAHE / Endo Sharp.

    Pipeline:
    1. Initial CLAHE
    2. Large-scale spatial normalization
    3. Total Variation denoising
    """

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    # --------------------------------------------------------
    # TV-CLAHE parameters based on MDPI Paper
    # --------------------------------------------------------

    clip_limit = 1.5
    tile_grid = (8, 8)

    gaussian_sigma = 50.0
    intensity_floor = 5.0

    # Total Variation denoising weight (Chambolle algorithm).
    # The paper uses 0.1.
    tv_weight = 0.1 

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid,
    )

    # ========================================================
    # 1. INITIAL CLAHE
    # ========================================================

    clahe_image = clahe.apply(
        image
    )

    # ========================================================
    # 2. SPATIAL NORMALIZATION
    # ========================================================

    image_float = clahe_image.astype(
        np.float32
    )

    gaussian_image = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
    )

    denominator = np.maximum(
        gaussian_image,
        intensity_floor,
    )

    normalized = (
        image_float
        / denominator
    )

    # ========================================================
    # ROBUST NORMALIZATION (Rescale for TV filtering)
    # ========================================================

    low = np.percentile(
        normalized,
        1.0,
    )

    high = np.percentile(
        normalized,
        99.0,
    )

    if high > low:
        normalized = (
            normalized - low
        ) / (
            high - low
        )
    else:
        normalized = np.zeros_like(
            normalized,
            dtype=np.float32,
        )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(np.float32)

    # ========================================================
    # 3. TOTAL VARIATION DENOISING
    # ========================================================

    tv_image = denoise_tv_chambolle(
        normalized,
        weight=tv_weight,
        channel_axis=None,
    )

    final_image = np.clip(
        tv_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return final_image


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def contrast_edge_enhancement(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):
    """CLAHE followed by USM."""
    enhanced = apply_clahe(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
    )

    return apply_usm(
        enhanced,
        sigma=sigma,
        amount=amount,
        threshold=threshold,
    )


def spatial_filtering_masking(
    image,
    strength,
):
    """Standard convolution sharpening."""
    return apply_standard_masking(
        image,
        strength=strength,
    )


def noise_reduction_smoothing(
    image,
    kernel_size,
):
    """Median noise reduction."""
    return apply_median_filter(
        image,
        kernel_size=kernel_size,
    )


def endo_preset(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):
    """Endodontic enhancement preset."""
    enhanced = apply_clahe(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
    )

    return apply_usm(
        enhanced,
        sigma=sigma,
        amount=amount,
        threshold=threshold,
    )


def perio_bone_preset(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):
    """Periodontal / bone enhancement preset."""
    enhanced = apply_clahe(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
    )

    return apply_usm(
        enhanced,
        sigma=sigma,
        amount=amount,
        threshold=threshold,
    )


# ============================================================
# DESCRIPTIONS
# ============================================================

PIPELINE_DESCRIPTIONS = {

    "Contrast & Edge Enhancement": (
        "Ενισχύει την τοπική αντίθεση και τις ανατομικές "
        "ακμές χρησιμοποιώντας CLAHE και Unsharp Masking."
    ),

    "Spatial Filtering / Masking": (
        "Ενισχύει τις ακμές και τις τοπικές μεταβολές "
        "έντασης μέσω convolution masking."
    ),

    "Noise Reduction / Smoothing": (
        "Μειώνει μικρό τοπικό θόρυβο και μεμονωμένα "
        "ακραία pixel χρησιμοποιώντας Median Filter."
    ),

    "Endo Preset": (
        "Ρύθμιση για ενδοδοντική απεικόνιση με στόχο "
        "την ανάδειξη λεπτών ενδοδοντικών δομών, "
        "ριζικών σωλήνων και ακρορριζίου."
    ),

    "Perio / Bone Preset": (
        "Ρύθμιση για περιοδοντική και οστική απεικόνιση "
        "με ελεγχόμενη όξυνση."
    ),

    "Endo Sharp": (
        "TV-CLAHE με Total Variation denoising και τοπική αντίθεση "
        "για καθαρότερη ενδοδοντική απεικόνιση (βάσει MDPI JCM 14:15 5554)."
    ),
    "X-Ray CLAHE": (
        "Ξεχωριστή εφαρμογή CLAHE για ακτινογραφικές εικόνες, "
        "διατηρώντας παράλληλα το αρχικό CLAHE της εφαρμογής."
    ),
}


# ============================================================
# SESSION STATE
# ============================================================

if "image" not in st.session_state:
    st.session_state.image = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "dicom" not in st.session_state:
    st.session_state.dicom = None

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Image")

    uploaded_file = st.file_uploader(
        "Drag & drop dental image",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "bmp",
        ],
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    if uploaded_file is not None:

        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if (
            st.session_state.file_signature
            != file_signature
        ):

            try:

                image, dicom = load_image(
                    uploaded_file
                )

                st.session_state.image = image
                st.session_state.filename = uploaded_file.name
                st.session_state.dicom = dicom
                st.session_state.file_signature = file_signature

                st.success(
                    f"Loaded: {uploaded_file.name}"
                )

            except Exception as error:

                st.error(
                    "Could not load image."
                )

                st.code(
                    str(error)
                )

    # ========================================================
    # PROCESSING
    # ========================================================

    if st.session_state.image is not None:

        st.divider()

        st.header("⚙️ Processing")

        pipeline = st.radio(
            "Select method:",
            [
                "Contrast & Edge Enhancement",
                "Spatial Filtering / Masking",
                "Noise Reduction / Smoothing",
                "Endo Preset",
                "Perio / Bone Preset",
                "Endo Sharp",
                "X-Ray CLAHE",
            ],
            index=0,
        )

        # ====================================================
        # CONTRAST & EDGE
        # ====================================================

        if pipeline == "Contrast & Edge Enhancement":

            st.caption(
                "CLAHE → Unsharp Masking"
            )

            clahe_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1, 10.0, 2.0, 0.1,
            )

            clahe_tile_size = st.slider(
                "CLAHE tile size",
                2, 32, 8, 1,
            )

            st.markdown(
                "**Unsharp Masking**"
            )

            usm_sigma = st.slider(
                "USM Gaussian sigma",
                0.1, 5.0, 1.0, 0.1,
            )

            usm_amount = st.slider(
                "USM amount",
                0.0, 5.0, 1.0, 0.1,
            )

            usm_threshold = st.slider(
                "USM threshold",
                0, 50, 5, 1,
            )

        # ====================================================
        # SPATIAL MASKING
        # ====================================================

        elif pipeline == "Spatial Filtering / Masking":

            st.caption(
                "Standard convolution masking"
            )

            masking_strength = st.slider(
                "Masking strength",
                0.0, 2.0, 1.0, 0.1,
            )

        # ====================================================
        # MEDIAN FILTER
        # ====================================================

        elif pipeline == "Noise Reduction / Smoothing":

            st.caption(
                "Median Filter"
            )

            median_kernel = st.selectbox(
                "Median kernel",
                [3, 5, 7, 9],
                index=0,
            )

        # ====================================================
        # ENDO PRESET
        # ====================================================

        elif pipeline == "Endo Preset":

            st.subheader(
                "🦷 Endo Preset"
            )

            st.info(
                "Στόχος: ανάδειξη λεπτών ενδοδοντικών "
                "δομών, ριζικών σωλήνων και ακρορριζίου."
            )

            endo_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1, 10.0, 2.0, 0.1,
                key="endo_clip",
            )

            endo_tile_size = st.slider(
                "CLAHE tile size",
                2, 32, 8, 1,
                key="endo_tile",
            )

            st.markdown(
                "**USM parameters**"
            )

            endo_sigma = st.slider(
                "USM Gaussian sigma",
                0.80, 1.20, 1.00, 0.05,
                key="endo_sigma",
            )

            endo_amount = st.slider(
                "USM amount",
                2.00, 3.00, 2.50, 0.05,
                key="endo_amount",
            )

            endo_threshold = st.slider(
                "USM threshold",
                1.00, 2.00, 1.50, 0.05,
                key="endo_threshold",
            )

        # ====================================================
        # PERIO / BONE PRESET
        # ====================================================

        elif pipeline == "Perio / Bone Preset":

            st.subheader(
                "🦴 Perio / Bone Preset"
            )

            st.info(
                "Στόχος: ανάδειξη της αρχιτεκτονικής "
                "του οστού και της περιοδοντικής σχισμής."
            )

            perio_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1, 10.0, 2.0, 0.1,
                key="perio_clip",
            )

            perio_tile_size = st.slider(
                "CLAHE tile size",
                2, 32, 8, 1,
                key="perio_tile",
            )

            st.markdown(
                "**USM parameters**"
            )

            perio_sigma = st.slider(
                "USM Gaussian sigma",
                1.50, 2.00, 1.75, 0.05,
                key="perio_sigma",
            )

            perio_amount = st.slider(
                "USM amount",
                1.00, 1.50, 1.25, 0.05,
                key="perio_amount",
            )

            perio_threshold = st.slider(
                "USM threshold",
                3.00, 4.00, 3.50, 0.05,
                key="perio_threshold",
            )

        # ====================================================
        # ENDO SHARP
        # ====================================================

        elif pipeline == "Endo Sharp":

            st.subheader(
                "🦷 Endo Sharp (TV-CLAHE)"
            )

            st.caption(
                "Based on MDPI JCM 14(15) 5554"
            )

            st.info(
                "CLAHE → Spatial Normalization → Total Variation Denoising"
            )

            st.success(
                "Ενισχύει την τοπική αντίθεση και μειώνει τον θόρυβο "
                "χωρίς να προσθέτει τεχνητές δομές (artifacts)."
            )


# ========================================================
        # X-RAY CLAHE
        # ========================================================

        elif pipeline == "X-Ray CLAHE":

            st.subheader("🩻 X-Ray CLAHE")

            st.caption(
                "Dedicated CLAHE option for X-ray enhancement"
            )

            xray_clip_limit = st.slider(
                "X-Ray CLAHE clip limit",
                0.1, 10.0, 2.0, 0.1,
                key="xray_clahe_clip",
            )

            xray_tile_size = st.slider(
                "X-Ray CLAHE tile size",
                2, 32, 8, 1,
                key="xray_clahe_tile",
            )


# ============================================================
# STOP IF NO IMAGE
# ============================================================

if st.session_state.image is None:

    st.info(
        "👆 Upload a dental image using the sidebar."
    )

    st.stop()


# ============================================================
# ORIGINAL IMAGE
# ============================================================

original = st.session_state.image


# ============================================================
# PROCESS IMAGE
# ============================================================

if pipeline == "Contrast & Edge Enhancement":

    processed = contrast_edge_enhancement(
        original,
        clahe_clip_limit,
        clahe_tile_size,
        usm_sigma,
        usm_amount,
        usm_threshold,
    )


elif pipeline == "Spatial Filtering / Masking":

    processed = spatial_filtering_masking(
        original,
        masking_strength,
    )


elif pipeline == "Noise Reduction / Smoothing":

    processed = noise_reduction_smoothing(
        original,
        median_kernel,
    )


elif pipeline == "Endo Preset":

    processed = endo_preset(
        original,
        endo_clip_limit,
        endo_tile_size,
        endo_sigma,
        endo_amount,
        endo_threshold,
    )


elif pipeline == "Perio / Bone Preset":

    processed = perio_bone_preset(
        original,
        perio_clip_limit,
        perio_tile_size,
        perio_sigma,
        perio_amount,
        perio_threshold,
    )


elif pipeline == "Endo Sharp":

    processed = tv_clahe(
        original
    )


elif pipeline == "X-Ray CLAHE":

    processed = xray_clahe_pipeline(
        original,
        clip_limit=xray_clip_limit,
        tile_size=xray_tile_size,
    )


else:

    processed = original.copy()


# ============================================================
# DISPLAY SCALING
# ============================================================

height, width = original.shape[:2]

MAX_WIDTH = 1000

if width > MAX_WIDTH:

    scale = (
        MAX_WIDTH
        / float(width)
    )

else:

    scale = 1.0


display_width = max(
    1,
    int(
        round(
            width * scale
        )
    ),
)

display_height = max(
    1,
    int(
        round(
            height * scale
        )
    ),
)


# ============================================================
# RESIZE FOR DISPLAY ONLY
# ============================================================

original_display = cv2.resize(
    original,
    (
        display_width,
        display_height,
    ),
    interpolation=cv2.INTER_AREA,
)

processed_display = cv2.resize(
    processed,
    (
        display_width,
        display_height,
    ),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# MAIN DISPLAY
# ============================================================

st.header(
    pipeline
)

st.info(
    PIPELINE_DESCRIPTIONS[
        pipeline
    ]
)


# ============================================================
# ORIGINAL / PROCESSED
# ============================================================

left, right = st.columns(2)


with left:

    st.subheader(
        "Original"
    )

    st.image(
        original_display,
        use_container_width=True,
    )


with right:

    st.subheader(
        "Processed"
    )

    st.image(
        processed_display,
        use_container_width=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

output = io.BytesIO()

Image.fromarray(
    processed
).save(
    output,
    format="PNG",
)

pipeline_filename = (
    pipeline
    .lower()
    .replace(
        " ",
        "_",
    )
    .replace(
        "/",
        "_",
    )
    .replace(
        "&",
        "and",
    )
)

st.download_button(
    "⬇️ Download processed image",
    output.getvalue(),
    file_name=(
        f"{Path(st.session_state.filename).stem}"
        f"_{pipeline_filename}.png"
    ),
    mime="image/png",
)


# ============================================================
# DICOM INFORMATION
# ============================================================

if st.session_state.dicom is not None:

    st.divider()

    with st.expander(
        "🏥 DICOM information"
    ):

        ds = st.session_state.dicom

        fields = [
            "PatientID",
            "StudyDate",
            "Modality",
            "Rows",
            "Columns",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "PhotometricInterpretation",
            "PixelSpacing",
            "SliceThickness",
            "WindowCenter",
            "WindowWidth",
            "RescaleSlope",
            "RescaleIntercept",
            "SamplesPerPixel",
            "PlanarConfiguration",
            "NumberOfFrames",
        ]

        metadata = {}

        for field in fields:

            if hasattr(
                ds,
                field,
            ):

                metadata[field] = str(
                    getattr(
                        ds,
                        field,
                    )
                )

        st.json(
            metadata
        )

        st.caption(
            "Η αρχική εικόνα DICOM δεν τροποποιείται. "
            "Η επεξεργασία εφαρμόζεται σε παράγωγη εικόνα."
        )
