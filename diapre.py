# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 00:52:41 2024

@author: JOSH
"""
import numpy as np
import sklearn 
import pickle
import streamlit as st
pickle_in = open("C:/Users/JOSH/Downloads/deploymentAI/classifier.pki", "rb")
loaded_model = pickle.load(pickle_in)



def diabetes_prediction(data):
    data_array = np.asarray(data).ravel()
    reshaped = data_array.reshape(1,-1)
    predicted = loaded_model.predict(reshaped)
    if predicted[0] == 0:
        return 'This person does not have diabetes'
    else:
        return 'This person has diabetes'
def main():
    st.title("Diabetes Prediction Web App")
    st.write("""
  Welcome to the Diabetes Prediction App! This application utilizes a machine learning model to predict the likelihood of an individual having diabetes based on various health-related features.

### Purpose:
- The primary goal of this app is to provide users with a simple tool for diabetes risk assessment.

### Model Information:
- The underlying model is a Random Forest Classifier, trained on a dataset containing information about diabetes patients.

### How to Use:
1. Input  health-related data in the sidebar.
2. Click the "Predict" button to see the prediction results.
3. Explore the model details and classification report to better understand the prediction accuracy.

Feel free to explore the app and gain insights into the factors influencing diabetes predictions. Remember that this tool is not a substitute for professional medical advice, and any predictions should be interpreted cautiously.

Let's get started! Input your data on the sidebar and click the "Predict" button to see your diabetes prediction.
""")
    Pregnancies = st.text_input("Enter patient Pregnancies Status: ")
    Glucose = st.text_input("Enter patient Glucose level: ")
    BloodPressure = st.text_input("Enter patient Blood pressure status: ")
    SkinThickness = st.text_input("Enter SkinThickness status: ")
    Insulin = st.text_input("Enter Insulin Status: ")
    BMI = st.text_input("Enter BMI status: ")
    DiabetesPedigreeFunction = st.text_input("Enter DF Diabete: ")
    Age = st.text_input("Enter Patient age: ")
    diagnosis = " "
    
    if st.button("Diabetes test result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)



if __name__ == '__main__':
    main()