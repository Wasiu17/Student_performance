import joblib
import streamlit as st

model = joblib.load('Best_model.jb')


st.title('STUDENT PERFORMANCE PREDICTOR')