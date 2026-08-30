import io
from pathlib import Path
import cv2
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


# ============================================================
# PAGE CONFIG
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
    Convert arbitrary grayscale data to uint8.
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

    minimum = np.min(
        array[finite]
    )

    maximum = np.max(
        array[finite]
    )

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

    return np.clip(
        normalized,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# DICOM LOADING
# ============================================================

def load_dicom(file):

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
            "If this is a JPEG Lossless DICOM, additional "
            "DICOM decoding libraries may be required.\n\n"
            f"Original error:\n{error}"
        ) from error

    # --------------------------------------------------------
    # Multi-frame DICOM
    # --------------------------------------------------------

    if image.ndim > 2:

        image = image[0]

    # --------------------------------------------------------
    # Rescale slope/intercept
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
    # MONOCHROME1
    # --------------------------------------------------------

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )
    )

    if photometric == "MONOCHROME1":

        image = (
            np.max(image)
            - image
        )

    # --------------------------------------------------------
    # DICOM Window Center / Width
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
                    - width_value / 2
                )

                high = (
                    center
                    + width_value / 2
                )

                image = np.clip(
                    image,
                    low,
                    high,
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = normalize_to_uint8(
        image
    )

    return image, ds


# ============================================================
# REGULAR IMAGE LOADING
# ============================================================

def load_regular_image(file):

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

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    clahe = cv2.createCLAHE(
        clipLimit=float(
            clip_limit
        ),
        tileGridSize=(
            int(tile_size),
            int(tile_size),
        ),
    )

    return clahe.apply(
        image
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

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # Gaussian blur
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    # High-frequency mask
    mask = (
        image.astype(np.float32)
        -
        blurred.astype(np.float32)
    )

    # Threshold
    if float(threshold) > 0:

        mask[
            np.abs(mask)
            < float(threshold)
        ] = 0.0

    # Sharpen
    sharpened = (
        image.astype(np.float32)
        +
        float(amount)
        * mask
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# STANDARD / CONVOLUTION MASKING
# ============================================================

def apply_standard_masking(
    image,
    strength=1.0,
):

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # Standard sharpening / convolution kernel
    sharpening_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # Identity matrix
    identity = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    identity[1, 1] = 1.0

    # Adjustable strength
    kernel = (
        identity
        +
        float(strength)
        *
        (
            sharpening_kernel
            -
            identity
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
    ).astype(
        np.uint8
    )


# ============================================================
# MEDIAN FILTER
# ============================================================

def apply_median_filter(
    image,
    kernel_size=3,
):

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
# TV-CLAHE
#
# THIS IS THE ORIGINAL TV-CLAHE IMPLEMENTATION
# FROM THE PREVIOUS APP.
#
# DO NOT CHANGE THE ALGORITHM OR PARAMETERS.
# ============================================================

def tv_clahe(image):

    """
    Original TV-CLAHE implementation.

    Workflow:
        1. CLAHE
        2. Spatial normalization
        3. Total Variation denoising
           using Chambolle's algorithm
        4. Second CLAHE

    Fixed parameters:
        CLAHE clip limit = 1.5
        CLAHE tile grid = 8 x 8
        Gaussian sigma = 50
        TV weight = 0.1
        TV floor = 5
    """

    # --------------------------------------------------------
    # Make sure the input is uint8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # --------------------------------------------------------
    # Fixed parameters
    # --------------------------------------------------------

    tv_clip_limit = 1.5
    tv_tile_grid = (8, 8)
    tv_sigma = 50.0
    tv_weight = 0.1
    tv_floor = 5.0

    clahe = cv2.createCLAHE(
        clipLimit=tv_clip_limit,
        tileGridSize=tv_tile_grid
    )

    # --------------------------------------------------------
    # 1. FIRST CLAHE
    # --------------------------------------------------------

    first_clahe = clahe.apply(
        image
    )

    # --------------------------------------------------------
    # 2. SPATIAL NORMALIZATION
    #
    # Divide the CLAHE image by a large-scale
    # Gaussian-blurred version of itself.
    #
    # The denominator is floored at 5.
    # --------------------------------------------------------

    image_float = first_clahe.astype(
        np.float32
    )

    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=tv_sigma,
        sigmaY=tv_sigma
    )

    denominator = np.maximum(
        blurred,
        tv_floor
    )

    normalized = (
        image_float
        /
        denominator
    )

    # --------------------------------------------------------
    # Rescale normalized image to [0, 1]
    # --------------------------------------------------------

    norm_min = float(
        np.min(normalized)
    )

    norm_max = float(
        np.max(normalized)
    )

    if norm_max > norm_min:

        normalized = (
            (normalized - norm_min)
            /
            (norm_max - norm_min)
        )

    else:

        normalized = np.zeros_like(
            normalized,
            dtype=np.float32
        )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # 3. TOTAL VARIATION DENOISING
    #
    # Chambolle algorithm
    # weight = 0.1
    # --------------------------------------------------------

    tv_denoised = denoise_tv_chambolle(
        normalized,
        weight=tv_weight,
        channel_axis=None
    )

    tv_denoised = np.clip(
        tv_denoised * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # 4. SECOND CLAHE
    # --------------------------------------------------------

    final_image = clahe.apply(
        tv_denoised
    )

    return final_image


# ============================================================
# PIPELINE 1
# CONTRAST & EDGE ENHANCEMENT
# ============================================================

def contrast_edge_enhancement(
    image
):

    # CLAHE
    clahe_image = apply_clahe(
        image,
        clip_limit=2.0,
        tile_size=8,
    )

    # USM
    final_image = apply_usm(
        clahe_image,
        sigma=1.0,
        amount=1.0,
        threshold=5.0,
    )

    return final_image


# ============================================================
# PIPELINE 2
# SPATIAL FILTERING / MASKING
# ============================================================

def spatial_filtering_masking(
    image
):

    final_image = apply_standard_masking(
        image,
        strength=1.0,
    )

    return final_image


# ============================================================
# PIPELINE 3
# NOISE REDUCTION / SMOOTHING
# ============================================================

def noise_reduction_smoothing(
    image
):

    final_image = apply_median_filter(
        image,
        kernel_size=3,
    )

    return final_image


# ============================================================
# PIPELINE 4
# ENDO PRESET
# ============================================================

def endo_preset(
    image
):

    # --------------------------------------------------------
    # Endodontic preset
    #
    # Recommended:
    # Sigma     = 0.80 - 1.20
    # Amount    = 2.00 - 3.00
    # Threshold = 1.00 - 2.00
    #
    # Default midpoint:
    # Sigma     = 1.00
    # Amount    = 2.50
    # Threshold = 1.50
    # --------------------------------------------------------

    clahe_image = apply_clahe(
        image,
        clip_limit=2.0,
        tile_size=8,
    )

    final_image = apply_usm(
        clahe_image,
        sigma=1.00,
        amount=2.50,
        threshold=1.50,
    )

    return final_image


# ============================================================
# PIPELINE 5
# PERIO / BONE PRESET
# ============================================================

def perio_bone_preset(
    image
):

    # --------------------------------------------------------
    # Perio / Bone preset
    #
    # Recommended:
    # Sigma     = 1.50 - 2.00
    # Amount    = 1.00 - 1.50
    # Threshold = 3.00 - 4.00
    #
    # Default midpoint:
    # Sigma     = 1.75
    # Amount    = 1.25
    # Threshold = 3.50
    # --------------------------------------------------------

    clahe_image = apply_clahe(
        image,
        clip_limit=2.0,
        tile_size=8,
    )

    final_image = apply_usm(
        clahe_image,
        sigma=1.75,
        amount=1.25,
        threshold=3.50,
    )

    return final_image


# ============================================================
# PIPELINE 6
# ENDO SHARP
#
# ORIGINAL TV-CLAHE
# ============================================================

def endo_sharp(
    image
):

    return tv_clahe(
        image
    )


# ============================================================
# PIPELINE DESCRIPTIONS
# ============================================================

PIPELINE_DESCRIPTIONS = {

    "Contrast & Edge Enhancement": (
        "Ενισχύει την τοπική αντίθεση και στη συνέχεια "
        "τονίζει τις ακμές της εικόνας. Το CLAHE βοηθά "
        "στην ανάδειξη λεπτομερειών σε περιοχές με "
        "διαφορετική φωτεινότητα, ενώ το Unsharp Masking "
        "κάνει τις ανατομικές ακμές και τις λεπτές "
        "δομές πιο ευδιάκριτες."
    ),

    "Spatial Filtering / Masking": (
        "Χρησιμοποιεί convolution masking για την ενίσχυση "
        "των ακμών και των τοπικών μεταβολών έντασης. "
        "Μπορεί να κάνει τις λεπτές γραμμές, τα όρια "
        "των δομών και τις ακτινοσκιερές περιοχές "
        "πιο ευδιάκριτα."
    ),

    "Noise Reduction / Smoothing": (
        "Το Median Filter χρησιμοποιείται για τη μείωση "
        "μικρού τοπικού θορύβου και μεμονωμένων ακραίων "
        "pixel, διατηρώντας παράλληλα σχετικά καλά "
        "τις σημαντικές ακμές της εικόνας."
    ),

    "Endo Preset": (
        "Ειδικά ρυθμισμένο για ενδοδοντική απεικόνιση. "
        "Στόχος είναι η καλύτερη ανάδειξη των "
        "ενδοδοντικών εργαλείων, του ακρορριζίου και "
        "των στενών ριζικών σωλήνων. Χρησιμοποιεί "
        "ισχυρότερο Unsharp Masking με χαμηλό threshold "
        "ώστε να αναδεικνύονται ακόμη και λεπτές "
        "διαφορές πυκνότητας."
    ),

    "Perio / Bone Preset": (
        "Ειδικά ρυθμισμένο για περιοδοντολογία και "
        "απεικόνιση οστού. Στόχος είναι η ανάδειξη της "
        "αρχιτεκτονικής του σπογγώδους οστού και της "
        "περιοδοντικής σχισμής, με πιο ήπια όξυνση ώστε "
        "να περιορίζεται η υπερβολική ενίσχυση "
        "ψηφιακού θορύβου."
    ),

    "Endo Sharp": (
        "Η αρχική TV-CLAHE μέθοδος του προηγούμενου "
        "εργαλείου. Συνδυάζει CLAHE, χωρική κανονικοποίηση, "
        "Total Variation denoising με τον αλγόριθμο "
        "Chambolle και δεύτερο CLAHE. Στόχος είναι η "
        "βελτίωση της αντίθεσης και η μείωση ανεπιθύμητου "
        "θορύβου, διατηρώντας παράλληλα τις σημαντικές "
        "ανατομικές ακμές."
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

    st.header("📂 Εικόνα")

    uploaded_file = st.file_uploader(
        "Σύρετε εδώ την οδοντιατρική εικόνα",
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
    # Load image
    # --------------------------------------------------------

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

                (
                    image,
                    dicom,
                ) = load_image(
                    uploaded_file
                )

                st.session_state.image = (
                    image
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.dicom = (
                    dicom
                )

                st.session_state.file_signature = (
                    file_signature
                )

                st.success(
                    f"Φορτώθηκε: "
                    f"{uploaded_file.name}"
                )

            except Exception as error:

                st.error(
                    "Δεν ήταν δυνατή η φόρτωση της εικόνας."
                )

                st.code(
                    str(error)
                )

    # --------------------------------------------------------
    # Pipeline selection
    # --------------------------------------------------------

    if st.session_state.image is not None:

        st.divider()

        st.header(
            "⚙️ Pipeline"
        )

        pipeline = st.radio(
            "Επιλέξτε μέθοδο:",
            [
                "Contrast & Edge Enhancement",
                "Spatial Filtering / Masking",
                "Noise Reduction / Smoothing",
                "Endo Preset",
                "Perio / Bone Preset",
                "Endo Sharp",
            ],
        )


# ============================================================
# STOP IF NO IMAGE
# ============================================================

if st.session_state.image is None:

    st.info(
        "👆 Ανεβάστε μια οδοντιατρική εικόνα "
        "από το πλαϊνό μενού."
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

original = st.session_state.image


if pipeline == "Contrast & Edge Enhancement":

    processed = contrast_edge_enhancement(
        original
    )


elif pipeline == "Spatial Filtering / Masking":

    processed = spatial_filtering_masking(
        original
    )


elif pipeline == "Noise Reduction / Smoothing":

    processed = noise_reduction_smoothing(
        original
    )


elif pipeline == "Endo Preset":

    processed = endo_preset(
        original
    )


elif pipeline == "Perio / Bone Preset":

    processed = perio_bone_preset(
        original
    )


elif pipeline == "Endo Sharp":

    processed = endo_sharp(
        original
    )


else:

    processed = original.copy()


# ============================================================
# DISPLAY SCALING
# ============================================================

height, width = (
    original.shape[:2]
)

MAX_WIDTH = 1000

if width > MAX_WIDTH:

    scale = (
        MAX_WIDTH
        /
        float(width)
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
# PIPELINE TITLE
# ============================================================

st.header(
    pipeline
)


# ============================================================
# GREEK EXPLANATION
# ============================================================

st.info(
    PIPELINE_DESCRIPTIONS[
        pipeline
    ]
)


# ============================================================
# IMAGE COMPARISON
# ============================================================

left, right = st.columns(
    2
)


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
        "_"
    )
    .replace(
        "/",
        "_"
    )
    .replace(
        "&",
        "and"
    )
)

st.download_button(
    "⬇️ Λήψη επεξεργασμένης εικόνας",
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
        "🏥 DICOM πληροφορίες"
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
```
def normalize_to_uint8(array):
    """
    Convert arbitrary grayscale data to uint8.
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

    minimum = np.min(
        array[finite]
    )

    maximum = np.max(
        array[finite]
    )

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

    return np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# DICOM LOADING
# ============================================================

def load_dicom(file):

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
            "that the current Playground environment "
            "cannot decode.\n\n"
            "If this is a JPEG Lossless DICOM, the "
            "normal Streamlit deployment should use "
            "pylibjpeg/gdcm for decoding.\n\n"
            f"Original error:\n{error}"
        ) from error

    # --------------------------------------------------------
    # Multi-frame DICOM
    # --------------------------------------------------------

    if image.ndim > 2:
        image = image[0]

    # --------------------------------------------------------
    # Rescale slope/intercept
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
    # MONOCHROME1
    # --------------------------------------------------------

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )
    )

    if photometric == "MONOCHROME1":

        image = (
            np.max(image)
            - image
        )

    # --------------------------------------------------------
    # DICOM Window Center / Width
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
                    - width_value / 2
                )

                high = (
                    center
                    + width_value / 2
                )

                image = np.clip(
                    image,
                    low,
                    high,
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = normalize_to_uint8(
        image
    )

    # --------------------------------------------------------
    # Pixel spacing
    # --------------------------------------------------------

    pixel_spacing = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    spacing = None

    if (
        pixel_spacing is not None
        and len(pixel_spacing) >= 2
    ):

        try:

            spacing = (
                float(pixel_spacing[0]),
                float(pixel_spacing[1]),
            )

        except Exception:

            spacing = None

    return (
        image,
        ds,
        spacing,
    )


# ============================================================
# REGULAR IMAGE LOADING
# ============================================================

def load_regular_image(file):

    data = file.read()

    image = Image.open(
        io.BytesIO(data)
    ).convert("L")

    return (
        np.array(image),
        None,
        None,
    )


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

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    clahe = cv2.createCLAHE(
        clipLimit=float(
            clip_limit
        ),
        tileGridSize=(
            int(tile_size),
            int(tile_size),
        ),
    )

    return clahe.apply(
        image
    )


# ============================================================
# UNSHARP MASKING
# ============================================================

def apply_usm(
    image,
    sigma=1.0,
    amount=1.0,
    threshold=5,
):

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # --------------------------------------------------------
    # Gaussian blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    # --------------------------------------------------------
    # High-frequency mask
    # --------------------------------------------------------

    mask = (
        image.astype(np.float32)
        -
        blurred.astype(np.float32)
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if float(threshold) > 0:

        mask[
            np.abs(mask)
            < float(threshold)
        ] = 0.0

    # --------------------------------------------------------
    # Apply USM
    # --------------------------------------------------------

    sharpened = (
        image.astype(np.float32)
        +
        float(amount)
        * mask
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

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # Original convolution sharpening kernel

    sharpening_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    # Identity

    identity = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    identity[1, 1] = 1.0

    # Blend identity with sharpening kernel

    kernel = (
        identity
        +
        float(strength)
        *
        (
            sharpening_kernel
            -
            identity
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

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    kernel = int(
        kernel_size
    )

    if kernel < 3:
        kernel = 3

    if kernel % 2 == 0:
        kernel += 1

    return cv2.medianBlur(
        image,
        kernel,
    )


# ============================================================
# TV-CLAHE
# ORIGINAL ENDO SHARP IMPLEMENTATION
# ============================================================

def tv_clahe_stages(image):
    """
    Original TV-CLAHE / Endo Sharp pipeline.

    Returns every intermediate stage:

        1. Original
        2. First CLAHE
        3. Spatial normalization
        4. TV denoising
        5. Final CLAHE
    """

    if image.dtype != np.uint8:

        image = normalize_to_uint8(
            image
        )

    # --------------------------------------------------------
    # Fixed parameters
    # --------------------------------------------------------

    tv_clip_limit = 1.5
    tv_tile_grid = (8, 8)
    tv_sigma = 50.0
    tv_weight = 0.1
    tv_floor = 5.0

    clahe = cv2.createCLAHE(
        clipLimit=tv_clip_limit,
        tileGridSize=tv_tile_grid,
    )

    # --------------------------------------------------------
    # STAGE 1
    # FIRST CLAHE
    # --------------------------------------------------------

    first_clahe = clahe.apply(
        image
    )

    # --------------------------------------------------------
    # STAGE 2
    # SPATIAL NORMALIZATION
    # --------------------------------------------------------

    image_float = first_clahe.astype(
        np.float32
    )

    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=tv_sigma,
        sigmaY=tv_sigma,
    )

    denominator = np.maximum(
        blurred,
        tv_floor,
    )

    normalized = (
        image_float
        / denominator
    )

    # Normalize to [0, 1]

    norm_min = float(
        np.min(normalized)
    )

    norm_max = float(
        np.max(normalized)
    )

    if norm_max > norm_min:

        normalized = (
            (normalized - norm_min)
            / (norm_max - norm_min)
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
    # STAGE 3
    # TOTAL VARIATION DENOISING
    # --------------------------------------------------------

    tv_denoised = denoise_tv_chambolle(
        normalized,
        weight=tv_weight,
        channel_axis=None,
    )

    tv_denoised = np.clip(
        tv_denoised * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # STAGE 4
    # FINAL CLAHE
    # --------------------------------------------------------

    final_image = clahe.apply(
        tv_denoised
    )

    return {
        "Original": image,
        "First CLAHE": first_clahe,
        "Spatial Normalization": (
            np.clip(
                normalized * 255.0,
                0,
                255,
            ).astype(np.uint8)
        ),
        "TV Denoising": tv_denoised,
        "Final CLAHE": final_image,
    }


# ============================================================
# CONTRAST & EDGE ENHANCEMENT STAGES
# ============================================================

def contrast_edge_stages(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    clahe_image = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    usm_image = apply_usm(
        clahe_image,
        sigma,
        amount,
        threshold,
    )

    return {
        "Original": image,
        "CLAHE": clahe_image,
        "Unsharp Masking": usm_image,
    }


# ============================================================
# ENDO PRESET STAGES
# ============================================================

def endo_preset_stages(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    clahe_image = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    usm_image = apply_usm(
        clahe_image,
        sigma,
        amount,
        threshold,
    )

    return {
        "Original": image,
        "CLAHE": clahe_image,
        "USM": usm_image,
    }


# ============================================================
# PERIO / BONE PRESET STAGES
# ============================================================

def perio_bone_stages(
    image,
    clip_limit,
    tile_size,
    sigma,
    amount,
    threshold,
):

    clahe_image = apply_clahe(
        image,
        clip_limit,
        tile_size,
    )

    usm_image = apply_usm(
        clahe_image,
        sigma,
        amount,
        threshold,
    )

    return {
        "Original": image,
        "CLAHE": clahe_image,
        "USM": usm_image,
    }


# ============================================================
# SPATIAL MASKING STAGES
# ============================================================

def masking_stages(
    image,
    strength,
):

    masked = apply_standard_masking(
        image,
        strength,
    )

    return {
        "Original": image,
        "Convolution Masking": masked,
    }


# ============================================================
# MEDIAN STAGES
# ============================================================

def median_stages(
    image,
    kernel,
):

    filtered = apply_median_filter(
        image,
        kernel,
    )

    return {
        "Original": image,
        "Median Filter": filtered,
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

if "spacing" not in st.session_state:
    st.session_state.spacing = None

if "calibration" not in st.session_state:
    st.session_state.calibration = None

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

                (
                    image,
                    dicom,
                    spacing,
                ) = load_image(
                    uploaded_file
                )

                st.session_state.image = image

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.dicom = dicom

                st.session_state.spacing = spacing

                st.session_state.calibration = None

                st.session_state.file_signature = (
                    file_signature
                )

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

        st.header(
            "⚙️ Processing Pipeline"
        )

        pipeline = st.radio(
            "Select pipeline",
            [
                "Contrast & Edge Enhancement",
                "Spatial Filtering / Masking",
                "Noise Reduction / Smoothing",
                "Endo Preset",
                "Perio / Bone Preset",
                "Endo Sharp",
            ],
            index=0,
        )

        st.divider()

        # ====================================================
        # 1. CONTRAST & EDGE ENHANCEMENT
        # ====================================================

        if pipeline == "Contrast & Edge Enhancement":

            st.subheader(
                "Contrast & Edge Enhancement"
            )

            st.caption(
                "CLAHE → Unsharp Masking"
            )

            clahe_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
            )

            clahe_tile_size = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
            )

            st.markdown(
                "**Unsharp Masking**"
            )

            usm_sigma = st.slider(
                "USM Gaussian sigma",
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
        # 2. SPATIAL FILTERING / MASKING
        # ====================================================

        elif pipeline == "Spatial Filtering / Masking":

            st.subheader(
                "Spatial Filtering / Masking"
            )

            st.caption(
                "Standard Masking / Convolution Masking"
            )

            masking_strength = st.slider(
                "Masking strength",
                0.0,
                2.0,
                1.0,
                0.1,
            )

            st.markdown(
                "**Convolution kernel**"
            )

            st.code(
                "[ 0  -1   0 ]\n"
                "[-1   5  -1 ]\n"
                "[ 0  -1   0 ]"
            )

        # ====================================================
        # 3. MEDIAN
        # ====================================================

        elif pipeline == "Noise Reduction / Smoothing":

            st.subheader(
                "Noise Reduction / Smoothing"
            )

            st.caption(
                "Median Filter (Διάμεσο Φίλτρο)"
            )

            median_kernel = st.selectbox(
                "Median kernel",
                [
                    3,
                    5,
                    7,
                    9,
                ],
                index=0,
            )

        # ====================================================
        # 4. ENDO PRESET
        # ====================================================

        elif pipeline == "Endo Preset":

            st.subheader(
                "🦷 Endo Preset"
            )

            st.caption(
                "Endodontic image enhancement"
            )

            st.info(
                "Στόχος: να φαίνονται οι βελόνες, "
                "το ακρορρίζιο και οι στενωμένοι "
                "ριζικοί σωλήνες."
            )

            st.markdown(
                "**CLAHE**"
            )

            endo_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
                key="endo_clip",
            )

            endo_tile_size = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
                key="endo_tile",
            )

            st.markdown(
                "**USM parameters**"
            )

            endo_sigma = st.slider(
                "USM Gaussian sigma",
                0.80,
                1.20,
                1.00,
                0.05,
                key="endo_sigma",
                help=(
                    "Recommended range: 0.80–1.20"
                ),
            )

            endo_amount = st.slider(
                "USM amount",
                2.00,
                3.00,
                2.50,
                0.05,
                key="endo_amount",
                help=(
                    "Recommended range: 2.00–3.00"
                ),
            )

            endo_threshold = st.slider(
                "USM threshold",
                1.00,
                2.00,
                1.50,
                0.05,
                key="endo_threshold",
                help=(
                    "Recommended range: 1.00–2.00"
                ),
            )

            st.success(
                f"Current Endo settings: "
                f"σ={endo_sigma:.2f}, "
                f"Amount={endo_amount:.2f}, "
                f"Threshold={endo_threshold:.2f}"
            )

        # ====================================================
        # 5. PERIO / BONE PRESET
        # ====================================================

        elif pipeline == "Perio / Bone Preset":

            st.subheader(
                "🦴 Perio / Bone Preset"
            )

            st.caption(
                "Periodontology & bone architecture enhancement"
            )

            st.info(
                "Στόχος: να φαίνεται η αρχιτεκτονική "
                "του σπογγώδους οστού και η περιοδοντική "
                "σχισμή, χωρίς υπερβολικό ψηφιακό θόρυβο."
            )

            st.markdown(
                "**CLAHE**"
            )

            perio_clip_limit = st.slider(
                "CLAHE clip limit",
                0.1,
                10.0,
                2.0,
                0.1,
                key="perio_clip",
            )

            perio_tile_size = st.slider(
                "CLAHE tile size",
                2,
                32,
                8,
                1,
                key="perio_tile",
            )

            st.markdown(
                "**USM parameters**"
            )

            perio_sigma = st.slider(
                "USM Gaussian sigma",
                1.50,
                2.00,
                1.75,
                0.05,
                key="perio_sigma",
                help=(
                    "Recommended range: 1.50–2.00"
                ),
            )

            perio_amount = st.slider(
                "USM amount",
                1.00,
                1.50,
                1.25,
                0.05,
                key="perio_amount",
                help=(
                    "Recommended range: 1.00–1.50"
                ),
            )

            perio_threshold = st.slider(
                "USM threshold",
                3.00,
                4.00,
                3.50,
                0.05,
                key="perio_threshold",
                help=(
                    "Recommended range: 3.00–4.00"
                ),
            )

            st.success(
                f"Current Perio/Bone settings: "
                f"σ={perio_sigma:.2f}, "
                f"Amount={perio_amount:.2f}, "
                f"Threshold={perio_threshold:.2f}"
            )

        # ====================================================
        # 6. ENDO SHARP
        # ====================================================

        elif pipeline == "Endo Sharp":

            st.subheader(
                "🦷 Endo Sharp"
            )

            st.caption(
                "Original TV-CLAHE"
            )

            st.info(
                "CLAHE → Spatial Normalization → "
                "Chambolle TV Denoising → CLAHE"
            )

            st.markdown(
                "**Fixed parameters**"
            )

            st.write(
                "CLAHE clip limit: 1.5"
            )

            st.write(
                "CLAHE tile grid: 8 × 8"
            )

            st.write(
                "Gaussian sigma: 50"
            )

            st.write(
                "TV weight: 0.1"
            )

            st.write(
                "TV floor: 5"
            )

        # ====================================================
        # MEASUREMENT
        # ====================================================

        st.divider()

        st.header(
            "📏 Measurement"
        )

        if st.session_state.spacing:

            sx, sy = (
                st.session_state.spacing
            )

            st.success(
                f"Row spacing: "
                f"{sx:.6f} mm/pixel\n\n"
                f"Column spacing: "
                f"{sy:.6f} mm/pixel"
            )

        else:

            st.warning(
                "No DICOM PixelSpacing detected."
            )

        st.subheader(
            "Manual calibration"
        )

        known_distance = st.number_input(
            "Reference distance (mm)",
            min_value=0.001,
            value=10.0,
            step=0.5,
        )

        if st.button(
            "Clear calibration"
        ):

            st.session_state.calibration = None

            st.rerun()


# ============================================================
# STOP IF NO IMAGE
# ============================================================

if st.session_state.image is None:

    st.info(
        "👆 Upload a dental image using the sidebar."
    )

    st.stop()


# ============================================================
# DEFAULT VALUES
# ============================================================

clahe_clip_limit = 2.0
clahe_tile_size = 8

usm_sigma = 1.0
usm_amount = 1.0
usm_threshold = 5

masking_strength = 1.0

median_kernel = 3


# ============================================================
# CREATE ALL PIPELINE STAGES
# ============================================================

original = (
    st.session_state.image
)


if pipeline == "Contrast & Edge Enhancement":

    stages = contrast_edge_stages(
        original,
        clahe_clip_limit,
        clahe_tile_size,
        usm_sigma,
        usm_amount,
        usm_threshold,
    )


elif pipeline == "Spatial Filtering / Masking":

    stages = masking_stages(
        original,
        masking_strength,
    )


elif pipeline == "Noise Reduction / Smoothing":

    stages = median_stages(
        original,
        median_kernel,
    )


elif pipeline == "Endo Preset":

    stages = endo_preset_stages(
        original,
        endo_clip_limit,
        endo_tile_size,
        endo_sigma,
        endo_amount,
        endo_threshold,
    )


elif pipeline == "Perio / Bone Preset":

    stages = perio_bone_stages(
        original,
        perio_clip_limit,
        perio_tile_size,
        perio_sigma,
        perio_amount,
        perio_threshold,
    )


elif pipeline == "Endo Sharp":

    stages = tv_clahe_stages(
        original
    )


else:

    stages = {
        "Original": original
    }


# ============================================================
# DISPLAY SCALING
# ============================================================

height, width = (
    original.shape[:2]
)

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
# CREATE DISPLAY IMAGES
# ============================================================

display_stages = {}

for stage_name, stage_image in stages.items():

    if stage_image.dtype != np.uint8:

        stage_image = normalize_to_uint8(
            stage_image
        )

    display_stages[stage_name] = cv2.resize(
        stage_image,
        (
            display_width,
            display_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# FINAL IMAGE
# ============================================================

final_image = list(
    stages.values()
)[-1]


# ============================================================
# TABS
# ============================================================

comparison_tab, measurement_tab, dicom_tab = (
    st.tabs(
        [
            "🖼️ Pipeline",
            "📏 Measurement",
            "🏥 DICOM",
        ]
    )
)


# ============================================================
# PIPELINE TAB
# ============================================================

with comparison_tab:

    st.subheader(
        pipeline
    )

    # --------------------------------------------------------
    # Pipeline flow
    # --------------------------------------------------------

    stage_names = list(
        stages.keys()
    )

    st.markdown(
        " ** → ** ".join(
            stage_names
        )
    )

    st.divider()

    # --------------------------------------------------------
    # DISPLAY ALL STAGES
    # --------------------------------------------------------

    number_of_stages = len(
        display_stages
    )

    # Two columns for comparison

    if number_of_stages == 2:

        columns = st.columns(2)

    elif number_of_stages == 3:

        columns = st.columns(3)

    elif number_of_stages == 4:

        columns = st.columns(2)

    else:

        columns = st.columns(3)

    for index, (
        stage_name,
        stage_image,
    ) in enumerate(
        display_stages.items()
    ):

        column = columns[
            index % len(columns)
        ]

        with column:

            st.markdown(
                f"### {stage_name}"
            )

            st.image(
                stage_image,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Processing parameters
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Processing parameters"
    )

    if pipeline == "Contrast & Edge Enhancement":

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "CLAHE clip limit",
                f"{clahe_clip_limit:.1f}",
            )

        with col2:
            st.metric(
                "USM sigma",
                f"{usm_sigma:.2f}",
            )

        with col3:
            st.metric(
                "USM amount",
                f"{usm_amount:.2f}",
            )

        st.write(
            f"CLAHE tile size: "
            f"{clahe_tile_size} × {clahe_tile_size}"
        )

        st.write(
            f"USM threshold: "
            f"{usm_threshold}"
        )

    elif pipeline == "Spatial Filtering / Masking":

        st.write(
            f"Masking strength: "
            f"{masking_strength:.2f}"
        )

        st.write(
            "Kernel:"
        )

        st.code(
            "[ 0  -1   0 ]\n"
            "[-1   5  -1 ]\n"
            "[ 0  -1   0 ]"
        )

    elif pipeline == "Noise Reduction / Smoothing":

        st.write(
            f"Median kernel: "
            f"{median_kernel} × {median_kernel}"
        )

    elif pipeline == "Endo Preset":

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "USM sigma",
                f"{endo_sigma:.2f}",
            )

        with col2:
            st.metric(
                "USM amount",
                f"{endo_amount:.2f}",
            )

        with col3:
            st.metric(
                "USM threshold",
                f"{endo_threshold:.2f}",
            )

        st.write(
            f"CLAHE clip limit: "
            f"{endo_clip_limit:.1f}"
        )

        st.write(
            f"CLAHE tile size: "
            f"{endo_tile_size} × {endo_tile_size}"
        )

    elif pipeline == "Perio / Bone Preset":

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "USM sigma",
                f"{perio_sigma:.2f}",
            )

        with col2:
            st.metric(
                "USM amount",
                f"{perio_amount:.2f}",
            )

        with col3:
            st.metric(
                "USM threshold",
                f"{perio_threshold:.2f}",
            )

        st.write(
            f"CLAHE clip limit: "
            f"{perio_clip_limit:.1f}"
        )

        st.write(
            f"CLAHE tile size: "
            f"{perio_tile_size} × {perio_tile_size}"
        )

    elif pipeline == "Endo Sharp":

        st.write(
            "CLAHE clip limit: 1.5"
        )

        st.write(
            "CLAHE tile grid: 8 × 8"
        )

        st.write(
            "Gaussian sigma: 50"
        )

        st.write(
            "TV weight: 0.1"
        )

        st.write(
            "TV floor: 5"
        )

    # --------------------------------------------------------
    # DOWNLOAD FINAL IMAGE
    # --------------------------------------------------------

    st.divider()

    output = io.BytesIO()

    Image.fromarray(
        final_image
    ).save(
        output,
        format="PNG",
    )

    st.download_button(
        "⬇️ Download final processed image",
        output.getvalue(),
        file_name=(
            f"{Path(st.session_state.filename).stem}"
            f"_{pipeline.lower().replace(' ', '_').replace('/', '_')}"
            ".png"
        ),
        mime="image/png",
    )


# ============================================================
# MEASUREMENT TAB
# ============================================================

with measurement_tab:

    st.subheader(
        "📏 Point-to-point measurement"
    )

    st.write(
        "Draw a line between two points on the final "
        "pipeline image."
    )

    canvas = st_canvas(
        fill_color=(
            "rgba(255, 0, 0, 0.1)"
        ),
        stroke_width=3,
        stroke_color="#ff0000",
        background_image=Image.fromarray(
            display_stages[
                list(
                    display_stages.keys()
                )[-1]
            ]
        ),
        drawing_mode="line",
        height=int(
            display_height
        ),
        width=int(
            display_width
        ),
        update_streamlit=True,
        key=f"measurement_canvas_{pipeline}",
    )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    if canvas.json_data is not None:

        objects = (
            canvas.json_data.get(
                "objects",
                [],
            )
        )

        if len(objects) > 0:

            line = objects[-1]

            x1 = float(
                line.get(
                    "x1",
                    0,
                )
            )

            y1 = float(
                line.get(
                    "y1",
                    0,
                )
            )

            x2 = float(
                line.get(
                    "x2",
                    0,
                )
            )

            y2 = float(
                line.get(
                    "y2",
                    0,
                )
            )

            scale_x = float(
                line.get(
                    "scaleX",
                    1,
                )
            )

            scale_y = float(
                line.get(
                    "scaleY",
                    1,
                )
            )

            dx_display = (
                (x2 - x1)
                * scale_x
            )

            dy_display = (
                (y2 - y1)
                * scale_y
            )

            display_distance = math.sqrt(
                dx_display ** 2
                +
                dy_display ** 2
            )

            if scale > 0:

                source_distance = (
                    display_distance
                    / scale
                )

            else:

                source_distance = 0.0

            st.metric(
                "Distance",
                f"{source_distance:.2f} pixels",
            )

            # ------------------------------------------------
            # DICOM physical distance
            # ------------------------------------------------

            if st.session_state.spacing:

                sx, sy = (
                    st.session_state.spacing
                )

                dx = (
                    dx_display
                    / scale
                )

                dy = (
                    dy_display
                    / scale
                )

                distance_mm = math.sqrt(
                    (dx * sy) ** 2
                    +
                    (dy * sx) ** 2
                )

                st.metric(
                    "Physical distance",
                    f"{distance_mm:.3f} mm",
                )

            # ------------------------------------------------
            # Manual calibration
            # ------------------------------------------------

            elif (
                st.session_state.calibration
            ):

                mm_per_pixel = float(
                    st.session_state.calibration
                )

                distance_mm = (
                    source_distance
                    *
                    mm_per_pixel
                )

                st.metric(
                    "Calibrated distance",
                    f"{distance_mm:.3f} mm",
                )

            # ------------------------------------------------
            # Calibration
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Calibration"
            )

            st.write(
                "For a non-DICOM image, draw the line "
                "over an object with a known physical length."
            )

            if st.button(
                "Set calibration from this line"
            ):

                if source_distance > 0:

                    mm_per_pixel = (
                        float(
                            known_distance
                        )
                        /
                        float(
                            source_distance
                        )
                    )

                    st.session_state.calibration = (
                        mm_per_pixel
                    )

                    st.success(
                        "Calibration: "
                        f"{mm_per_pixel:.8f} mm/pixel"
                    )

                    st.rerun()

                else:

                    st.warning(
                        "Please draw a valid line first."
                    )


# ============================================================
# DICOM TAB
# ============================================================

with dicom_tab:

    st.subheader(
        "🏥 DICOM metadata"
    )

    ds = (
        st.session_state.dicom
    )

    if ds is None:

        st.info(
            "The uploaded image is not a DICOM file."
        )

    else:

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

        st.warning(
            "Original DICOM pixel data is not "
            "overwritten. Processing is applied "
            "to a derived image."
        )
