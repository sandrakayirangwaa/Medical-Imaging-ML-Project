import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------

st.set_page_config(
    page_title="Eye Health Check",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("aptos_mobilenetv2.keras")


model = load_model()


class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

class_explanations = {
    "No DR":
        "The model did not find signs of diabetic retinopathy in this image.",

    "Mild":
        "The model classified this image as mild diabetic retinopathy.",

    "Moderate":
        "The model classified this image as moderate diabetic retinopathy.",

    "Severe":
        "The model classified this image as severe diabetic retinopathy.",

    "Proliferative DR":
        "The model classified this image as proliferative diabetic retinopathy."
}


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


if st.session_state.dark_mode:

    background = "#182014"
    sidebar = "#202b19"
    card = "#273321"
    card_hover = "#2e3b27"
    main_text = "#e7ecd9"
    secondary_text = "#b9c5a4"
    border = "#536442"
    olive = "#829b58"
    olive_dark = "#60763e"
    soft_olive = "#34442a"

else:

    background = "#f1f4e7"
    sidebar = "#e3e9d3"
    card = "#f8f9f2"
    card_hover = "#eef2df"
    main_text = "#30421f"
    secondary_text = "#687653"
    border = "#c5cfad"
    olive = "#687f43"
    olive_dark = "#506633"
    soft_olive = "#e3e9d3"


st.markdown(
    f"""
    <style>

    /* Whole page */
    .stApp {{
        background-color: {background};
        color: {main_text};
        transition: background-color 0.4s ease;
    }}

    /* Streamlit top bar */
    header[data-testid="stHeader"] {{
        background-color: {background} !important;
    }}

    header[data-testid="stHeader"] * {{
        color: {main_text} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar};
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {main_text};
    }}

    /* Main area */
    .block-container {{
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Title */
    .title {{
        color: {main_text};
        font-size: 34px;
        font-weight: 650;
        margin-bottom: 4px;
        animation: fadeIn 0.7s ease;
    }}

    /* Description */
    .description {{
        color: {secondary_text};
        font-size: 16px;
        margin-bottom: 28px;
        animation: fadeIn 0.9s ease;
    }}

    /* Sidebar title */
    .sidebar-title {{
        color: {main_text};
        font-size: 23px;
        font-weight: 650;
        margin-bottom: 5px;
    }}

    .sidebar-description {{
        color: {secondary_text};
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 25px;
    }}

    /* Sidebar divider */
    .sidebar-line {{
        height: 1px;
        background-color: {border};
        margin: 20px 0;
    }}

    /* Upload card */
    .upload-section {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 22px;
        animation: fadeUp 0.7s ease;
        transition: background-color 0.4s ease;
    }}

    .upload-title {{
        color: {main_text};
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 6px;
    }}

    .upload-description {{
        color: {secondary_text};
        font-size: 14px;
    }}

    /* Result card */
    .result-section {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        animation: fadeUp 0.7s ease;
        transition: background-color 0.4s ease;
    }}

    .result-heading {{
        color: {secondary_text};
        font-size: 14px;
        margin-bottom: 5px;
    }}

    .prediction {{
        color: {olive};
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .explanation {{
        color: {secondary_text};
        font-size: 15px;
        line-height: 1.5;
    }}

    .confidence-heading {{
        color: {main_text};
        font-size: 15px;
        font-weight: 600;
        margin-top: 22px;
    }}

    .confidence {{
        color: {olive};
        font-size: 25px;
        font-weight: 700;
    }}

    /* Information box */
    .info-box {{
        background-color: {soft_olive};
        border-left: 4px solid {olive};
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
        color: {secondary_text};
        font-size: 14px;
        line-height: 1.5;
    }}

    /* About */
    .about-section {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 22px;
        margin-top: 28px;
        color: {secondary_text};
        font-size: 14px;
        line-height: 1.6;
        animation: fadeIn 1s ease;
    }}

    .about-title {{
        color: {main_text};
        font-size: 19px;
        font-weight: 600;
        margin-bottom: 8px;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {olive};
        color: white;
        border: none;
        border-radius: 10px;
        width: 100%;
        padding: 10px;
        transition: all 0.25s ease;
    }}

    .stButton > button:hover {{
        background-color: {olive_dark};
        transform: translateY(-2px);
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {card};
        border-radius: 12px;
    }}

    /* Uploaded image */
    [data-testid="stImage"] {{
        animation: fadeUp 0.7s ease;
    }}

    /* Alerts */
    .stAlert {{
        background-color: {soft_olive};
        border: 1px solid {border};
        color: {main_text};
    }}

    /* Animations */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    @keyframes fadeUp {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Eye Health Check</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-description">
            A simple retinal image classification project.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Home")

    st.markdown(
        """
        <div class="sidebar-description">
            Upload a retinal image and view the model prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Appearance")

    if st.session_state.dark_mode:

        if st.button("Switch to light mode"):
            st.session_state.dark_mode = False
            st.rerun()

    else:

        if st.button("Switch to dark mode"):
            st.session_state.dark_mode = True
            st.rerun()

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown("### About")

    st.markdown(
        """
        <div class="sidebar-description">
            This project uses the APTOS 2019 retinal image dataset
            and a MobileNetV2 model to classify diabetic retinopathy
            severity.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="title">Eye Health Check</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">Upload a retinal image to see the model prediction.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-section">
        <div class="upload-title">Upload a retinal image</div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )


    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = np.argmax(predictions[0])

    confidence = float(
        predictions[0][predicted_class]
    )

    predicted_label = class_names[predicted_class]

    st.markdown(
        '<div class="result-section">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-heading">Model prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="prediction">{predicted_label}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="explanation">{class_explanations[predicted_label]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="confidence-heading">Confidence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">{confidence:.2%}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Confidence message
    # -----------------------------------------------------

    if confidence < 0.60:

        st.warning(
            "The model is not very sure about this result. "
            "Please do not rely on this prediction."
        )

    else:

        st.info(
            "This is the model's confidence in its prediction. "
            "It is not a medical diagnosis."
        )


# ---------------------------------------------------------
# Medical safety note
# ---------------------------------------------------------

st.markdown(
    """
    <div class="info-box">
        <strong>Important:</strong><br>
        This is a student coursework project. It is not a medical
        diagnostic tool and should not be used to make health decisions.
        A qualified healthcare professional should review any medical concern.
    </div>
    """,
    unsafe_allow_html=True
)


