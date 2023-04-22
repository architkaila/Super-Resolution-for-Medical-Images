## Library imports
import streamlit as st
from PIL import Image

## Local imports
from model.run_inference import run_model_inference, prepare_image
from model.model_config import DEVICE


def new_image_enhancer_UI(model):
    """
    The main UI function to display the recommended recipes page
    """

    ## Pager subheader
    st.divider()
    st.subheader("Choose an X-Ray Image To Enhance ...")

    ## Image uploader
    image_upload = st.file_uploader("Upload A Low Resolution X-Ray Image (Png)", type="png")

    ## If image is uploaded
    if image_upload is not None:
        ## Load image
        input_image = Image.open(image_upload).convert('RGB')

        ## Prepare the image for model
        input_image_for_model = prepare_image(image_upload, is_hr_image=True)

        ## Run the model
        output_image = run_model_inference(input_image_for_model, model, device=DEVICE)

        ## Place holder container for the results
        results = st.empty()

        ## Display the results
        with results.container():
            col1, col2 = st.columns(2)

            ## Display the low resolution image
            with col1:
                st.subheader("Low Resolution Input Image")
                st.image(input_image_for_model, use_column_width=True)

            ## Display the super resolution image
            with col2:
                st.subheader("Super Resolution Output Image")
                st.image(output_image, use_column_width=True)