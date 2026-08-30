import io
from pathlib import Path

import cv2
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Dental Image Processor",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 Dental Image Processor")
st.caption("Επεξεργασία και βελτίωση οδοντιατρικών ακτινογραφιών")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_to_uint8(array):
    """Convert any grayscale image to uint8."""

    array = np.asarray(array, dtype=np.float32)

    finite = np.isfinite(array)

    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)

    minimum = np.min(array[finite])
    maximum = np.max(array[finite])

    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)

    result = (
        (array - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# DICOM
# ============================================================

def load_dicom(file):
    """Load a DICOM image and return uint8 image + dataset."""

    ds = pydicom.dcmread(file, force=True)

    try:
        image = ds.pixel_array.astype(np.float32)
    except Exception as error:
        raise RuntimeError(
            "This DICOM could not be decoded.\n\n"
            "Compressed DICOM files may require an additional "
            "decoder such as pylibjpeg.\n\n"
            f"Original error:\n{error}"
        ) from error

    # Multi-frame: use first frame
    if image.ndim > 2:
        image = image[0]

    # Rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    # MONOCHROME1 -> invert
    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )
    )

    if photometric == "MONOCHROME1":
        image = np.max(image) - image

    # Windowing, when available
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)

    try:
        if center is not None and width is not None:

            if hasattr(center, "__len__"):
                center = float(center[0])
            else:
                center = float(center)

            if hasattr(width, "__len__"):
                width = float(width[0])
            else:
                width = float(width)

            if width > 1:
                low = center - width / 2.0
                high = center + width / 2.0
                image = np.clip(image, low, high)

    except Exception:
        pass

    return normalize_to_uint8(image), ds


# ============================================================
# REGULAR IMAGE
# ============================================================

def load_regular_image(file):
    """Load PNG/JPG/TIFF/BMP as grayscale."""

    data = file.read()

    image = Image.open(
        io.BytesIO(data)
    ).convert("L")

    return np.asarray(image), None


# ============================================================
# GENERAL IMAGE LOADER
# ============================================================

def load_image(file):

    filename = file.name.lower()

    if filename.endswith(".dcm"):
        return load_dicom(file)

    try:
        file.seek(0)
        return load_regular_image(file)

    except Exception:
        file.seek(0)
        return load_dicom(file)


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image,
    clip_limit=2.0,
    tile_size=8,
):

    image = normalize_to_uint8(image)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(
            int(tile_size),
            int(tile_size),
        ),
    )

    return clahe.apply(image)


# ============================================================
# UNSHARP MASKING
# ============================================================

def apply_usm(
    image,
    sigma=1.0,
    amount=1.0,
    threshold=5.0,
):

    image = normalize_to_uint8(image)

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    image_float = image.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    detail = image_float - blurred_float

    threshold = float(threshold)

    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0

    result = (
        image_float
        + float(amount) * detail
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# STANDARD SHARPENING / MASKING
# ============================================================

def apply_standard_masking(
    image,
    strength=1.0,
):

    image = normalize_to_uint8(image)

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
        * (sharpening_kernel - identity)
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

    image = normalize_to_uint8(image)

    kernel_size = int(kernel_size)

    if kernel_size < 3:
        kernel_size = 3

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(
        image,
        kernel_size,
    )


# ============================================================
# CONTROLLED DETAIL SHARPENING
# ============================================================

def sharpen_detail(
    image,
    sigma=0.8,
    amount=1.8,
    threshold=1.5,
):

    image = normalize_to_uint8(image)

    image_float = image.astype(np.float32)

    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    detail = image_float - blurred

    threshold = float(threshold)

    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0

    result = (
        image_float
        + float(amount) * detail
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# TV-CLAHE / ENDO SHARP
# ============================================================

def tv_clahe(image):

    image = normalize_to_uint8(image)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    clip_limit = 1.5
    tile_grid = (8, 8)

    gaussian_sigma = 50.0
    intensity_floor = 5.0

    # Reduced regularization to preserve anatomy
    tv_weight = 0.03

    # Final sharpening
    sharpening_sigma = 0.8
    sharpening_amount = 1.8
    sharpening_threshold = 1.5

    # --------------------------------------------------------
    # CLAHE object
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid,
    )

    # --------------------------------------------------------
    # 1. Initial CLAHE
    # --------------------------------------------------------

    clahe_image = clahe.apply(image)

    # --------------------------------------------------------
    # 2. Spatial normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normalize [0, 1]
    # --------------------------------------------------------

    minimum = float(normalized.min())
    maximum = float(normalized.max())

    if maximum > minimum:

        normalized = (
            normalized - minimum
        ) / (
            maximum - minimum
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

    # --------------------------------------------------------
    # 3. TV denoising
    # --------------------------------------------------------

    tv_image = denoise_tv_chambolle(
        normalized,
        weight=tv_weight,
        channel_axis=None,
    )

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------

    tv_image = np.clip(
        tv_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 4. Final CLAHE
    # --------------------------------------------------------

    final_image = clahe.apply(
        tv_image
    )

    # --------------------------------------------------------
    # 5. Detail sharpening
    # --------------------------------------------------------

    final_image = sharpen_detail(
        final_image,
        sigma=sharpening_sigma,
        amount=sharpening_amount,
        threshold=sharpening_threshold,
    )

    return final_image


# ============================================================
# PIPELINES
# ============================================================

def contrast_edge_enhancement(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    result = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    return apply_usm(
        result,
        sigma,
        amount,
        threshold,
    )


def endo_preset(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    result = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    return apply_usm(
        result,
        sigma,
        amount,
        threshold,
    )


def perio_preset(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    result = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    return apply_usm(
        result,
        sigma,
        amount,
        threshold,
    )


# ============================================================
# DESCRIPTIONS
# ============================================================

DESCRIPTIONS = {
    "Contrast & Edge Enhancement":
        "CLAHE και Unsharp Masking για βελτίωση τοπικής "
        "αντίθεσης και ανατομικών ακμών.",

    "Spatial Filtering / Masking":
        "Convolution sharpening για ενίσχυση ακμών.",

    "Noise Reduction / Smoothing":
        "Median filtering για μείωση τοπικού θορύβου.",

    "Endo Preset":
        "Ρύθμιση για ανάδειξη λεπτών ενδοδοντικών δομών, "
        "ριζικών σωλήνων και ακρορριζίου.",

    "Perio / Bone Preset":
        "Ρύθμιση για ανάδειξη οστικής αρχιτεκτονικής "
        "και περιοδοντικών δομών.",

    "Endo Sharp":
        "TV-CLAHE με μειωμένη TV regularization και "
        "ελεγχόμενη ενίσχυση λεπτομερειών."
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

    # --------------------------------------------------------
    # Load uploaded image
    # --------------------------------------------------------

    if uploaded_file is not None:

        signature = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if (
            signature
            != st.session_state.file_signature
        ):

            try:

                image, dicom = load_image(
                    uploaded_file
                )

                st.session_state.image = image
                st.session_state.dicom = dicom
                st.session_state.filename = (
                    uploaded_file.name
                )
                st.session_state.file_signature = (
                    signature
                )

                st.success(
                    f"Loaded: {uploaded_file.name}"
                )

            except Exception as error:

                st.error(
                    "Could not load image."
                )

                st.code(str(error))

    # --------------------------------------------------------
    # Processing options
    # --------------------------------------------------------

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
            ],
        )

        # ====================================================
        # CLAHE + USM
        # ====================================================

        if pipeline == "Contrast & Edge Enhancement":

            clahe_clip = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
            )

            clahe_tile = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
            )

            usm_sigma = st.slider(
                "USM sigma",
                0.1,
                5.0,
                1.0,
                0.1,
            )

            usm_amount = st.slider(
                "USM amount",
                0.0,
                5.0,
                1.0,
                0.1,
            )

            usm_threshold = st.slider(
                "USM threshold",
                0,
                50,
                5,
                1,
            )

        # ====================================================
        # MASKING
        # ====================================================

        elif pipeline == "Spatial Filtering / Masking":

            masking_strength = st.slider(
                "Masking strength",
                0.0,
                2.0,
                1.0,
                0.1,
            )

        # ====================================================
        # MEDIAN
        # ====================================================

        elif pipeline == "Noise Reduction / Smoothing":

            median_kernel = st.selectbox(
                "Median kernel",
                [3, 5, 7, 9],
                index=0,
            )

        # ====================================================
        # ENDO
        # ====================================================

        elif pipeline == "Endo Preset":

            st.subheader("🦷 Endo Preset")

            st.caption(
                "Για λεπτές ενδοδοντικές δομές."
            )

            endo_clip = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
                key="endo_clip",
            )

            endo_tile = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
                key="endo_tile",
            )

            endo_sigma = st.slider(
                "USM sigma",
                0.80,
                1.20,
                1.00,
                0.05,
                key="endo_sigma",
            )

            endo_amount = st.slider(
                "USM amount",
                2.00,
                3.00,
                2.50,
                0.05,
                key="endo_amount",
            )

            endo_threshold = st.slider(
                "USM threshold",
                1.00,
                2.00,
                1.50,
                0.05,
                key="endo_threshold",
            )

        # ====================================================
        # PERIO
        # ====================================================

        elif pipeline == "Perio / Bone Preset":

            st.subheader("🦴 Perio / Bone Preset")

            st.caption(
                "Για οστική και περιοδοντική απεικόνιση."
            )

            perio_clip = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
                key="perio_clip",
            )

            perio_tile = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
                key="perio_tile",
            )

            perio_sigma = st.slider(
                "USM sigma",
                1.50,
                2.00,
                1.75,
                0.05,
                key="perio_sigma",
            )

            perio_amount = st.slider(
                "USM amount",
                1.00,
                1.50,
                1.25,
                0.05,
                key="perio_amount",
            )

            perio_threshold = st.slider(
                "USM threshold",
                3.00,
                4.00,
                3.50,
                0.05,
                key="perio_threshold",
            )

        # ====================================================
        # ENDO SHARP
        # ====================================================

        elif pipeline == "Endo Sharp":

            st.subheader("🦷 Endo Sharp")

            st.caption(
                "TV-CLAHE + anatomical detail sharpening"
            )

            st.info(
                "Βελτιστοποιημένο για υψηλότερη οπτική "
                "ευκρίνεια και ανάδειξη λεπτών δομών."
            )


# ============================================================
# NO IMAGE
# ============================================================

if st.session_state.image is None:

    st.info(
        "👆 Upload a dental image using the sidebar."
    )

    st.stop()


# ============================================================
# ORIGINAL
# ============================================================

original = st.session_state.image


# ============================================================
# PROCESSING
# ============================================================

if pipeline == "Contrast & Edge Enhancement":

    processed = contrast_edge_enhancement(
        original,
        clahe_clip,
        clahe_tile,
        usm_sigma,
        usm_amount,
        usm_threshold,
    )

elif pipeline == "Spatial Filtering / Masking":

    processed = apply_standard_masking(
        original,
        masking_strength,
    )

elif pipeline == "Noise Reduction / Smoothing":

    processed = apply_median_filter(
        original,
        median_kernel,
    )

elif pipeline == "Endo Preset":

    processed = endo_preset(
        original,
        endo_clip,
        endo_tile,
        endo_sigma,
        endo_amount,
        endo_threshold,
    )

elif pipeline == "Perio / Bone Preset":

    processed = perio_preset(
        original,
        perio_clip,
        perio_tile,
        perio_sigma,
        perio_amount,
        perio_threshold,
    )

elif pipeline == "Endo Sharp":

    processed = tv_clahe(
        original
    )

else:

    processed = original.copy()


# ============================================================
# DISPLAY
# ============================================================

height, width = original.shape[:2]

MAX_DISPLAY_WIDTH = 1400

scale = min(
    1.0,
    MAX_DISPLAY_WIDTH / float(width),
)

display_width = max(
    1,
    int(round(width * scale)),
)

display_height = max(
    1,
    int(round(height * scale)),
)

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
# RESULTS
# ============================================================

st.header(pipeline)

st.info(
    DESCRIPTIONS[pipeline]
)

left, right = st.columns(2)

with left:

    st.subheader("Original")

    st.image(
        original_display,
        use_container_width=True,
    )

with right:

    st.subheader("Processed")

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

filename = Path(
    st.session_state.filename
).stem

safe_pipeline_name = (
    pipeline
    .lower()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("&", "and")
)

download_name = (
    f"{filename}_{safe_pipeline_name}.png"
)

st.download_button(
    "⬇️ Download processed image",
    data=output.getvalue(),
    file_name=download_name,
    mime="image/png",
)


# ============================================================
# DICOM INFORMATION
# ============================================================

if st.session_state.dicom is not None:

    st.divider()

    with st.expander("🏥 DICOM information"):

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

            if hasattr(ds, field):

                metadata[field] = str(
                    getattr(ds, field)
                )

        st.json(metadata)

        st.caption(
            "Η αρχική εικόνα DICOM δεν τροποποιείται. "
            "Η επεξεργασία εφαρμόζεται σε παράγωγη εικόνα."
        )
