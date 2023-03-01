#############################
import os

import streamlit as st
from chatbot_core.inference import chatbot_response
import tensorflow as tf


############################
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


data_name = st.selectbox('select data', os.listdir(os.path.join(os.getcwd(), 'uploads')))
data_path = os.path.join(os.getcwd(), 'uploads', data_name)
model_name = st.selectbox('select inference model', os.listdir(os.path.join(os.getcwd(), 'saved_models')))
model_path = os.path.join(os.getcwd(), 'saved_models', model_name)
loaded_model = load_model(model_path)
user_input = st.text_area('You: ', "Hi there!")
response = chatbot_response(user_input, data_path, loaded_model)
st.text_area('bot', response)
