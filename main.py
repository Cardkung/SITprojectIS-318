import streamlit as st
import pandas as pd
import pathlib

st.title("Cardkung AI Visualization")
st.markdown("## SIT Independent Study Visualization")

st.markdown("Upload the CSV File")

#uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True, type=["csv"])
#st.write("filename:", uploaded_files.name)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
