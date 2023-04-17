## Library imports
import streamlit as st
import requests

from config import PAGES

def image_enhancer_UI():
    """
    The main UI function to display the Image Super Resolution page UI for webapp
    """
    st.divider()
    st.write("Image Super Resolution")
