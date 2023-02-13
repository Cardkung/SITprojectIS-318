import streamlit as st
import pandas as pd
import pathlib


st.markdown("# SIT Independent Study Visualization")

st.markdown("## Upload the CSV File")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

