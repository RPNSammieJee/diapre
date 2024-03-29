import sys
import numpy as np
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
diabetes_data = pd.read_excel("diabetes.xlsx")
x = diabetes_data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y = diabetes_data["Outcome"]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=1)
#Fitting the train data into the logistics model

# Create a Logistic Regression model
model = LogisticRegression(fit_intercept=False)

# Fit the model to the training data
model.fit(x_train, y_train)
t = model.predict(x_test)
file_name = open("modellings.pki", "wb")
pickle.dump(model, file_name)
file_name.close()

#C:\Users\JOSH\PycharmProjects\diabetes_prediction\
pickle_in = open(r"modellings.pki", "rb")
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
    Pregnancies = st.number_input("Enter patient Pregnancies Status: ")
    Glucose = st.number_input("Enter patient Glucose level: ")
    BloodPressure = st.number_input("Enter patient Blood pressure status: ")
    SkinThickness = st.number_input("Enter SkinThickness status: ")
    Insulin = st.number_input("Enter Insulin Status: ")
    BMI = st.number_input("Enter BMI status:", min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Enter DF Diabete:", min_value=0.0, step=0.1)
    Age = st.number_input("Enter Patient age: ")
    diagnosis = " "

    if st.button("Diabetes test result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
