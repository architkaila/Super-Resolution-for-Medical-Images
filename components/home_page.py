import logging
import numpy as np
import pandas as pd
import streamlit as st

## Initiate logging
logger = logging.getLogger(__name__)

def home_page_UI():
    """
    The main UI function to display the Home page UI for webapp
    """
    
    ## Display details on super resolution
    st.divider()
    st.subheader("What is Super Resolution? 🩻")
    st.write("The process of recovering and reconstructing the resolution of a noisy low-quality image into a very high-quality and high-resolution image is known as image Super Resolution")
    
    ## Display details on project
    st.subheader("About the Project ⭐")
    st.write("""
            This project aims to enhance the resolution and quality of medical X-ray images using state-of-the-art Generative Adversarial Networks (GANs). 
            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality X-ray images.
        """)
    
    ## Display details on model performance
    st.divider()
    st.subheader("Model Performance 🧨")
    st.write("""
            The resulting GAN model was evaluated and compared against ground truths using different metrics like PSNR and SSIM.
            Compared to PSNR, SSIM is often considered a more perceptually accurate metric, as it takes into account the human visual system's sensitivity to changes in luminance, contrast, and structure
            """)
    
    # Display PSNR
    row_1_col1, row_1_col2 = st.columns(2)
    with row_1_col1:
        st.success('Peak Signal-to-Noise Ratio: 41.66 db', icon="👀")
        st.write("""
            - It's a metric used to measure the quality of an image.
            - It measures difference b/w two images by comparing their pixel values and computing the ratio between the maximum pixel value and the mean squared error.
            - Higher the PSNR, lesser tehe difference b/w two images, indicating a higher quality.
        """)
    
    # Display SSIM
    with row_1_col2:
        st.info('Structural Similarity Index: 0.96', icon="🎯")
        st.write("""
            - It's a metric used to measure the similarity between two images.
            - It takes into account structural information of the images and computes a similarity score between the two based on structural factors.
            - Higher SSIM score indicates a higher similarity between the two images.
        """)

    ## Display details on risks and limitations
    st.divider()
    st.subheader("Risks and Limitations ⚠️")
    st.write("""
            - Generative networks may struggle to accurately capture important details in extremely low resolution medical X-ray images (< 128 x128), which could negatively impact the generated high quality images. 
            - The netowrk may generate features that dont exist.
            - The use of generative networks in medical imaging raises ethical concerns around issues such as bias, accountability, and transparency.
            """)
    st.caption("Minimizing the Risk ✅")
    st.write("""To minimize the risk of bias, the model was trained on a diverse dataset of X-ray images. 
            Furthermore, addtion of perceptual loss to the model helps to ensure that the generated images are similar to the original images and no new features are generated while enhancing the resolution.
            """)

    ## Display details on about me
    st.divider()
    st.subheader("About me")
    st.write("""
    I am doing this project as a part of our core curriculam at Duke University for Masters in Artificial Intelligence (Course: AIPI 540: Deep Learning Applications)
    """)