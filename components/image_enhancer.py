## Library imports
import streamlit as st
from PIL import Image

## Local imports
from config import PAGES
from model.run_inference import run_model_inference, prepare_image
from model.model_config import DEVICE
from config import SAMPLE_IMAGES


def image_enhancer_UI(model):
    """
    The main UI function to display the Image Super Resolution page UI for webapp
    """

    ## Pager subheader
    st.divider()
    st.subheader("Choose Patient X-Ray Image To Enhance")

    ## Plot the chest x-ray samples
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    ## Display xray image samples for the user to select
    count = 0
    for sample_image in SAMPLE_IMAGES[:4]:
        with columns[count]:
            ## Display the image
            st.image(sample_image['image_path'], use_column_width=True)

            ## Display the checkbox
            sample_image["selected"] = st.checkbox(sample_image["image_id"])
            
            ## Increment the column count
            count += 1
            if count >= 4:
                count = 0

    ## Place holder container for the results
    st.divider()
    results = st.empty()

    ## Generate the super resolution image
    col7, col5, col6 = st.columns([1, 1.5, 1])
    with col5:
        if st.button('Generate SuperRes Image', use_container_width=True):
            
            ## Get the selected image
            seleted_sample_image = [sample_image["image_path"] for sample_image in SAMPLE_IMAGES if sample_image["selected"]]
            
            ## Fetch the super resolution image
            with st.spinner('Please Wait ...'):
                ## Load the original image
                input_image = Image.open(seleted_sample_image[0]).convert('RGB')

                ## Prepare the image for model
                input_image_for_model = prepare_image(seleted_sample_image[0], is_hr_image=True)

                ## Run the model
                output_image = run_model_inference(input_image_for_model, model, device=DEVICE)
            
            ## Button to clear the selection
            if st.button("Clear Selection", use_container_width=True):
                with results.container():
                    results.empty()
            
            ## Display the results
            with results.container():
                col8, col9 = st.columns(2)

                ## Display the Low Resolution Image
                with col8:
                    st.subheader("Low Resolution Input Image")
                    st.image(input_image_for_model, use_column_width=True)

                ## Display the Super Resolution Image
                with col9:
                    st.subheader("Super Resolution Output Image")
                    st.image(output_image, use_column_width=True)