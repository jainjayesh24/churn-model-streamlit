# Churn Predictor: Analyze & Forecast Customer Retention

This project uses an Artificial Neural Network (ANN) model to predict customer churn based on multiple features such as geography, age, balance, and credit score. The goal is to help businesses identify at-risk customers to take steps to improve customer retention.

## Project Overview

The project offers a user-friendly web interface built with **Streamlit** that allows users to input customer data and get predictions on the likelihood of customer churn. The backend is powered by a machine learning model trained on a customer churn dataset.

### Key Features:
- **Input Features**: Users can input customer data like Geography, Gender, Age, Balance, Credit Score, Estimated Salary, Tenure, Number of Products, etc.
- **Churn Probability**: The model outputs a churn probability score based on the inputs.
- **Red Flag Alerts**: The interface displays a warning if the customer is at high risk of churn, enabling proactive measures.

## Dataset

The model is trained on customer data in the `Churn_Modelling.csv` file, which includes:

- **Geography**: The country of the customer.
- **Gender**: Male or Female.
- **Age**: Age of the customer.
- **Balance**: Bank account balance of the customer.
- **Credit Score**: Customer's credit score.
- **Estimated Salary**: Customer's estimated salary.
- **Tenure**: Number of years the customer has been with the company.
- **Number of Products**: Number of products used by the customer.
- **Has Credit Card**: Binary value indicating if the customer owns a credit card.
- **Is Active Member**: Binary value indicating if the customer is an active member.

## Technologies Used

- **Artificial Neural Network (ANN)**: A deep learning model that predicts customer churn based on the input features.
    - **Keras/TensorFlow**: The ANN model is built using the Keras API with TensorFlow as the backend. The model architecture and weights are stored in `model.h5`.
- **Preprocessing**:
    - **Label Encoding**: Categorical features like Geography and Gender are label encoded. The encoding files are stored in `label.pkl` and `label2.pkl`.
    - **Scaling**: Numerical data is scaled using a scaler, which is saved in `scaler.pkl`.
- **Streamlit**: The web interface is built with **Streamlit**, a fast and simple way to create data-driven web applications using Python.
- **Jupyter Notebooks**: The model training and prediction logic are contained in the `model.ipynb` and `prediction.ipynb` notebooks.
