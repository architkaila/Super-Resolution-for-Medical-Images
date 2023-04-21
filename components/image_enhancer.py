## Library imports
import streamlit as st
from PIL import Image

## Local imports
from config import PAGES
from model.run_model import run_inference, prepare_image
from model.model_config import DEVICE
from config import SAMPLE_IMAGES


def image_enhancer_UI(model):
    """
    The main UI function to display the Image Super Resolution page UI for webapp
    """
    st.divider()
    st.subheader("Choose Patient X-Ray Image To Enhance")

    ## Plot the samples
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    ## Display the top recipes for the user to select
    count = 0
    for sample_image in SAMPLE_IMAGES[:4]:
        with columns[count]:
            ## Display the recipe image
            st.image(sample_image['image_path'], use_column_width=True)

            ## Display the recipe title and checkbox
            sample_image["selected"] = st.checkbox(sample_image["image_id"])
            
            ## Increment the column count
            count += 1
            if count >= 4:
                count = 0

    st.divider()
    results = st.empty()

    col7, col5, col6 = st.columns([1, 1.5, 1])
    with col5:
        if st.button('Generate SuperRes Image', use_container_width=True):
            
            ## Get the recipe ids for recippies selected by user
            seleted_sample_image = [sample_image["image_path"] for sample_image in SAMPLE_IMAGES if sample_image["selected"]]
            
            ## Fetch the recommended recipies from the API
            with st.spinner('Please Wait ...'):
                ## Load the original image
                input_image = Image.open(seleted_sample_image[0]).convert('RGB')
                input_image_for_model = prepare_image(seleted_sample_image[0], is_hr_image=True)
                output_image = run_inference(input_image_for_model, model, device=DEVICE)
            if st.button("Clear Selection", use_container_width=True):
                with results.container():
                    results.empty()
            
            with results.container():
                col8, col9 = st.columns(2)
                with col8:
                    st.subheader("Low Resolution Input Image")
                    st.image(input_image_for_model, use_column_width=True)
                with col9:
                    st.subheader("Super Resolution Output Image")
                    st.image(output_image, use_column_width=True)
            
    results = st.empty()

                
                
            
            
    # ## Image uploader
    # image_upload = st.file_uploader("Upload A Low Resolution X-Ray Image (Png)", type="png")

    # if image_upload is not None:
    #     ## Load image
    #     input_image = Image.open(image_upload)

    #     input_image = prepare_image(image_upload, is_hr_image=True)

    #     output_image = run_inference(input_image, model, device=DEVICE)
