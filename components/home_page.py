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
    st.write("Homepage")