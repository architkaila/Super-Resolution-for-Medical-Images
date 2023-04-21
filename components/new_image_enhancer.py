## Library imports
import streamlit as st
from PIL import Image

## Local imports
from model.run_model import run_inference, prepare_image
from model.model_config import DEVICE


def new_image_enhancer_UI(model):
    """
    The main UI function to display the recommended recipes page
    """
    st.divider()
    st.subheader("Choose an X-Ray Image To Enhance ...")

    ## Image uploader
    image_upload = st.file_uploader("Upload A Low Resolution X-Ray Image (Png)", type="png")

    if image_upload is not None:
        ## Load image
        input_image = Image.open(image_upload).convert('RGB')
        input_image_for_model = prepare_image(image_upload, is_hr_image=True)
        output_image = run_inference(input_image_for_model, model, device=DEVICE)

        results = st.empty()
        with results.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Low Resolution Input Image")
                st.image(input_image_for_model, use_column_width=True)
            with col2:
                st.subheader("Super Resolution Output Image")
                st.image(output_image, use_column_width=True)