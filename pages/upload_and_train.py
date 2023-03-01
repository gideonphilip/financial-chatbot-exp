########################################
import os.path
from chatbot_core.models import models
import streamlit as st

########################################

st.title('Upload and Train Your Chatbot in a GO')
file = st.file_uploader('upload your data')
if file is not None:
    with open(os.path.join(os.getcwd(), 'uploads', file.name), 'wb') as f:
        f.write(file.getbuffer())
st.selectbox('Data Already Available', os.listdir(os.path.join(os.getcwd(), 'uploads')))
st.write('To train your model')
model_name = st.text_area('your model name: ')
train_model_name = st.selectbox('Select Our Nlp Model for Training', [i for i in models.keys()])
basemodel = models[train_model_name]
if st.button('Train'):
    data_path = os.path.join(os.getcwd(), 'uploads', file.name)
    basemodel.train(data_path, model_name)
