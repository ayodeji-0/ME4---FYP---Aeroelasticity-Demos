import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# file structure
# AeroViz
# ├── AeroViz.py
# |── pages
# |   ├── 1_CBT.py
# |   ├── 2_Flutter.py
# |   ├── 3_LinearCascade.py
# |--Modules
# |   ├── CBT_Flutter.py


# import the module from the module folder
# This is the correct way to import the module
from modules.CBT_Flutter import *

st.set_page_config(layout="wide", page_icon="./icon.ico")


st.markdown(
                """
                <style>
                div[data-baseweb="base-input"] > markdown {
                    text-align: center;
                    },
                </style>
                <style>
                div[data-baseweb="base-input"] > textarea {
                    min-height: 1px;
                    min-width: 1px;
                    padding: 0;
                    resize: none;
                    -webkit-text-fill-color: black;
                    text-align: center;
                    font-size: 20px;
                    background-color: white;
                },
                </style>
                """,
                unsafe_allow_html=True,
            )

st.header('AeroViz')
