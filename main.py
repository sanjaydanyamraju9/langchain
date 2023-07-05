import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

st.title("Designer Blend") 

os.environ["OPENAI_API_KEY"] = openai_key
input_text = st.text_input("")
llm = OpenAI(temperature=0.8)

st.header('Upload an Art Sketch')
uploaded_file = st.file_uploader('Upload an Image')


st.header('Upload the Image of a Model')
uploaded_file = st.file_uploader('Upload a file')

# temperature - balanced answer

if input_text:
    # response = llm.generate(input_text)
    st.write(llm(input_text))