import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import cv2
import copy
import plotly.express as px
import nbformat
from attention.utils.output_plot import *
import requests



st.markdown("<h1 style='text-align: center;'>Le Wagon Demo Day</h1>", unsafe_allow_html=True)
st.write("#")

st.markdown("<h3 style='text-align: center;'>Your Attention Curve</h3>", unsafe_allow_html=True)
st.write("#")

# current_directory = os.getcwd()
# data_directory = os.path.join(current_directory, "attention_data")

data_url = st.secrets['data_url']
data_csv = requests.get(data_url).content
df = pd.read_csv(data_csv).sort_values(by=['timestamp', 'face_idx']).reset_index(drop=True)



fig = plot_attention_curve(df, 44)
st.plotly_chart(fig)

st.write("#")


col1, col2, col3 , col4, col5, col6, col7 = st.columns(7)

with col1:
    pass
with col2:
    attentive = st.button("Very Attentive")
with col3:
    pass
with col5:
    pass
with col4 :
    pass
with col6:
    non_attentive = st.button("Not Attentive")
with col7:
    pass


st.write("#")

st.write("#")


# if attentive:
#     st.image(os.path.join(data_directory, 'demo', 'attentive.png'))
# if  non_attentive:
#     st.image(os.path.join(data_directory, 'demo', 'inattentive.png'))
