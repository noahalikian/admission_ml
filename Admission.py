# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

password_guess = st.text_input("What is the Password?")
if password_guess != st.secrets["password"]:
    st.stop()
st.title('Graduate Admission Predictor')
st.image('admission.jpg')

st.text('This app uses multiple inputs to predict the probability of admission to graduate school')

reg_pickle = open('reg_admission.pickle', 'rb')
clf = pickle.load(reg_pickle) 
reg_pickle.close()

st.sidebar.header('Enter Your Profile Details')

gre = st.sidebar.number_input('GRE Score', value = 320)
toefl = st.sidebar.number_input('TOEFL Score', value = 100)
gpa = st.sidebar.number_input('CGPA', value = 8.00)
re = st.sidebar.selectbox('Research Experience', ['Yes', 'No'])
ur = st.sidebar.slider('University Rating', min_value = 1.00, max_value = 5.00, step = 0.5, value = 3.00)
sop = st.sidebar.slider('Statement of Purpose (SOP)', min_value = 1.00, max_value = 5.00, step = 0.5, value = 3.50)
lor = st.sidebar.slider('Letter of Recommendation', min_value = 1.00, max_value = 5.00, step = 0.5, value = 3.50)

st.sidebar.subheader('Enter Significance Level (Range: 0-1)')
sig_lev = st.sidebar.number_input('Significance level', value = .1)
butt = st.sidebar.button('Predict')

re_Yes, re_No = 0, 0
if re == 'Yes':
    re_Yes = 1
elif re == 'No':
    re_No = 1

if butt:
    st.subheader('Prediction Admission Chance...')

    values = [[gre, toefl, ur, sop, lor, gpa, re_No, re_Yes]]
    alpha = sig_lev

    y_pred, y_pi = clf.predict(values, alpha=alpha)

    st.metric('Predicted Admission Probability',f"{(y_pred[0] * 100).round(3)}%", border = True)

    st.text(f"With a {(1-alpha)*100}% confidence level:")

    int = [f"{round(float(y_pi[0][0]*100), 3)}%", f"{round(float(y_pi[0][1]*100), 3)}%"]
    st.text(f'Prediction Interval: {int}')

    st.subheader('Model Insights')
    tab1, tab2, tab3, tab4 = st.tabs(['Feature Importance', 'Histogram of Residuals', 'Predicted vs. Actual', 'Coverage Plot'])
    with tab1:
        st.write('Feature Importance')
        st.image('feat_fig.svg')
        st.caption('Relative importance of features in prediction')
    with tab2:
        st.write('Histogram of Residuals')
        st.image('hist_fig.svg')
        st.caption('Distribution of residuals to evaluate prediction quality')
    with tab3:
        st.write('Plot of Predicted vs. Actual')
        st.image('scatter_fig.svg')
        st.caption('Visual comparison of predicted and actual values')
    with tab4:
        st.write('Coverage Plot')
        st.image('cov_fig.svg')
        st.caption('Range of predictions with confidence intervals')