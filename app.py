import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

#Trained Model
model = load_model('Upmodel.h5')

#Encoders and Scalers
with open('label2.pkl', 'rb') as f:
    label2 = pickle.load(f)

with open('label.pkl', 'rb') as f:
    label = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Churn Predictor: Analyze & Forecast Customer Retention')

#Input
geography = st.selectbox('Geography', label2.categories_[0])
gender = st.selectbox('Gender', label.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = label2.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label2.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Red Flag: Potential Customer Loss Detected!.')
else:
    st.write('All Clear: Customer Retention Likely!')