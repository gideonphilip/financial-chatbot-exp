###############################
import streamlit as st
import os
import pandas as pd

###############################

models = os.listdir(os.path.join(os.getcwd(), 'saved_models'))
dataframe = pd.DataFrame(models, columns=['Models'])
st.title('All the models trained so far')
st.dataframe(dataframe, use_container_width=True)
