## Library imports
import streamlit as st
from PIL import Image

def about_us_UI():
    """
    The main UI function to display the About US page UI.
    """

    ## Display about the project section
    st.divider()
    st.subheader("About the Project")
    st.write("""
            This project aims to enhance the resolution and quality of medical X-ray images using state-of-the-art Generative Adversarial Networks (GANs). 
            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality X-ray images.
        """)
    
    ## Display dataset details
    st.subheader("Dataset")
    st.write("""
            The  Dataset used to train the Super Resolution model is [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data). It comprised of 112,120 X-ray images.
            The dataset has High resolution images (1024 x 1024). 
            Even though this dataset is primally for lung disease identification, I striped off the labels for each image and instead use these high-resolution X-ray images for training my super-resolution GAN.
            """)
    ## Display Model Architecture
    st.divider()
    st.subheader("Model Architecture")
    st.image("./data/images/network_architecture.png", caption="Model Architecture")
    st.write("""
            I have implemented the [Swift-SRGAN](https://arxiv.org/pdf/2111.14320.pdf) model architecture to enhance the resolution of low-quality X-ray images.
            The Generative network was trained on a proposed dataset of Chest X-rays.
            Given an input image of size 256 x 256, the generator generates a high-resolution image of size 1024 x 1024.
            """)
    
    # Details of Generator
    st.caption("Genertor architecture")
    st.write("""
            The generator consists of Depthwise Separable Convolutional layers, which are used to reduce the number of parameters and computation in the network.
            The major part of network is created of 16 residual blocks. Each block has a Depthwise Convolution followed by Batch Normalization, PReLU activation, another Depthwise Convolution, and Batch Normalizationa and finally a skip connection.
            After the residual blocks, the images is passsed thourgh upsample blocks and finally thourgh a convolution layer to generate the final output image.
            """)
    
    # Details of Discriminator
    st.caption("Discriminator architecture")
    st.write("""
            The discriminator consists of 8 Depthwise Separable Convolutional blocks.
            Each block has a Depthwise Convolution followed by Batch Normalization and LeakyReLU activation. 
            After the 8 blocks the images is passed through Avg Pooling and a fully connected layer to generate the final output.
            The objective of the discriminator is to classify super-resolution images generated by the geenrator as fake and original high-resolution images as real.     
            """)
    
    ## Display details on Loss functions used
    st.divider()
    st.subheader("Loss functions")
    st.write("""
            The loss function for the generator is a combination of multiple losses. The main one is Perceptual Loss which is a combination of Adversarial Loss and Content Loss.
            """)
    code = '''Total_Loss = Image_Loss + Perception_Loss + TV_Loss

Perceptual_Loss = Adversarial_Loss + Content_Loss
    '''
    st.code(code, language='python')

    # Details of Image loss
    st.caption("Image loss")
    st.write("""
            This is a naive loss functionn whihc calculates the Mean Squared Error b/w the generated image and the original high res image pixels.
            """)
    
    # Details of Perceptual loss
    st.caption("Content loss")
    st.write("""
            It represents the information that is lost or distorted during the processing of an image.
            The image generated by the generator and the original high res image are passed though the MobileNetV2 network to compute the feature vectors of both the images. 
            Content loss is calculated as the euclidean distance b/w the feature vectors of the original image and the generated image.
            """)
    
    # Details of Adversarial loss
    st.caption("Adversarial loss")
    st.write("""
            Used to train the generator network by providing a signal on how to improve the quality of the generated images.
            THis is calculated based on the discriminator's output, which indicates how well it can distinguish between the real and fake images.
            Generator tries to minimize this loss, by trying to generate images that the discriminator cannot distinguish.
            """)
    
    # Details of Total Variation loss
    st.caption("Total Variation loss")
    st.write("""
            It measures the variation or changes in intensity or color between adjacent pixels in an image. 
            It is defined as the sum of the absolute differences between neighboring pixels in both the horizontal and vertical directions in an image.
            """)
    
    ## Display details on About me
    st.divider()
    st.subheader("About me")

    st.write("""
    I am doing this project as a part of our core curriculam at Duke University for Masters in Artificial Intelligence (Course: AIPI 540: Deep Learning Applications)
    """)