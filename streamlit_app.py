## Library imports
import pandas as pd 
import numpy as np 
import streamlit as st
from streamlit import runtime
import pickle

## Local imports
from components import home_page
from components import image_enhancer
from components import new_image_enhancer
from components import about_us
from config import PAGES

@st.cache_data
def load_data():
    """ 
    Loads the required sample image paths into the session state

    Args:
        None

    Returns:
        None
    """

    ## Load some sample chest x-ray images
    with open('./data/val_images.pkl', 'rb') as f:
        val_data_list = pickle.load(f)

    ## Save the sample image paths into the session state
    st.session_state.val_images = val_data_list

## Set the page tab title
st.set_page_config(page_title="Medical Image Enhancer", page_icon="🤖", layout="wide")

## Load the initial app data
load_data()

## Landing page UI
def run_UI():
    """
    The main UI function to display the UI for the webapp
    """

    ## Set the page title and navigation bar
    st.sidebar.title('Select Menu')
    if st.session_state["page"]:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state["page"])
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=0)
    st.experimental_set_query_params(page=page)


    ## Display the page selected on the navigation bar
    if page == 'Home':
        st.sidebar.write("""
            ## About
            
            The project aims to provide personalized recipe recommendations to users based on their preferences and available ingredients.
            
            Users will select their favorite recipes, and the system will use content-based filtering to suggest similar and popular recipes.
            
            The recommendations will also the user's available ingredients.
        """)
        st.title("Home")
        home_page.home_page_UI()

    elif page == 'Image Enhancer Example':
        st.sidebar.write("""
            ## About
            
            The project aims to provide personalized recipe recommendations to users based on their preferences and available ingredients.
            
            Users will select their favorite recipes, and the system will use content-based filtering to suggest similar and popular recipes.
            
            The recommendations will also the user's available ingredients.
        """)
        st.title("Image Enhancing Examples")
        image_enhancer.image_enhancer_UI()
    
    elif page == 'Try Your Own Image':
        st.sidebar.write("""
            ## About
            
            The project aims to provide personalized recipe recommendations to users based on their preferences and available ingredients.
            
            Users will select their favorite recipes, and the system will use content-based filtering to suggest similar and popular recipes.
            
            The recommendations will also the user's available ingredients.
        """)
        st.title("Try Your Own Image")
        new_image_enhancer.new_image_enhancer_UI()

    else:
        st.sidebar.write("""
            ## About
            
            The project aims to provide personalized recipe recommendations to users based on their preferences and available ingredients.
            
            Users will select their favorite recipes, and the system will use content-based filtering to suggest similar and popular recipes.
            
            The recommendations will also the user's available ingredients.
        """)
        st.title("About Us")
        about_us.about_us_UI()


if __name__ == '__main__':
    ## Load the streamlit app with "Recipe Recommender" as default page
    if runtime.exists():

        ## Get the page name from the URL
        url_params = st.experimental_get_query_params()
        if len(url_params.keys()) == 0 or "page" not in st.session_state:
            st.session_state.page = 1

        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                ## Set the default page as "Home"
                st.experimental_set_query_params(page='Home')
                url_params = st.experimental_get_query_params()
                st.session_state.page = PAGES.index(url_params['page'][0])
        
        ## Call the main UI function
        run_UI()