# Use this to predefine the stylesheet for the GUI
# Create separate functions for each text style with unsafe_allow_html=True then call from the main script
import streamlit as st
def max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


# Button numbering guide
# Button_A_B_C where A is the column number, B is the tab number, C is the button index