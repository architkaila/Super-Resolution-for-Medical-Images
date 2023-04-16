## Library imports
import streamlit as st
from PIL import Image

def about_us_UI():
    """
    The main UI function to display the About US page UI.
    """

    st.write("""
        The project aims to provide personalized recipe recommendations to users based on their preferences and available ingredients. Users will select their favorite recipes, and the system will use content-based filtering to suggest similar and popular recipes. The recommendations will also the user's available ingredients.
        
        We are doing this project as a part of our core curriculam at Duke University for Masters in Artificial Intelligence (Course: AIPI 540: Deep Learning Applications)
        """)

    st.markdown("""---""")
    st.subheader("The Team")

    ## Displays the team members
    row_1_col1, row_1_col2, row_1_col3 = st.columns(3)
    with row_1_col1:
        image = Image.open('data/images/archit.jpeg')
        st.image(image, caption="Archit")