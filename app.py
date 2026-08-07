import streamlit as st
import pandas as pd
import joblib
import requests
import os


MODEL_URL = "https://huggingface.co/spaces/RSYD21/Kelayakan_Kredit/resolve/main/random_forest_model.pkl"


MODEL_PATH = "random_forest_model.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        response = requests.get(MODEL_URL)

        with open(
            MODEL_PATH,
            "wb"
        ) as file:

            file.write(response.content)


    model = joblib.load(MODEL_PATH)

    return model



model = load_model()
