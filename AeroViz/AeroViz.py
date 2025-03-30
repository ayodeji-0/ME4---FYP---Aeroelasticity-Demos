import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# file structure
# AeroViz
# ├── AeroViz.py
# ├── stylesheet.py/css # stylesheet file
# ├── requirements.txt # requirements file
# ├── README.md # readme file
# ├── .streamlit
# │   └── config.toml # streamlit config file
# ├── chatbot
# │   ├── __init__.py
# │   ├── retrieval.py
# │   ├── summarizer.py
# │   ├── data
# │   │   ├── embedded_papers.json # json file with embedded papers
# │   │   └── faiss.index # faiss index file
# │   └── models
# │       ├── minilm # MiniLM model 
# │       └── mpnet # MPNet model
# ├── icons
# │   └── icon.ico # icon file
# ├── modules
# │   ├── Blisks.py
# │   └── CBT.py
# ├── pages
# │   ├── 1_CBT.py
# │   ├── 2_Blisks.py
# │   └── 3_Assistant.py




st.set_page_config(layout="wide", page_icon="./icons/icon.ico")


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

## Page Title
st.header('AeroViz')

st.subheader("Motivational Objectives")
st.write('Aeroelasticity Visualisation Tool')

# nice pics

# final use case will be links etc

# consider scaling animations - really small displacements dont show, waste of computational power
