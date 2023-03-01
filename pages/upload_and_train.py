########################################
import os.path
from chatbot_core.train import train
import streamlit as st

########################################

st.title('Upload and Train Your Chatbot in a GO')
file = st.file_uploader('upload your data')
if file is not None:
    with open(os.path.join(os.getcwd(), 'uploads', file.name), 'wb') as f:
        f.write(file.getbuffer())

st.write('To train your model')
model_name = st.text_area('your model name: ')
if st.button('Train'):
    data_path = os.path.join(os.getcwd(), 'uploads', file.name)
    train(data_path,model_name)
