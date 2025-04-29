import streamlit as st
import pandas as pd
import pickle
import openai
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Please check OPENAI_API_KEY in the .env file.")
    st.stop()

# Load the dataset (to calculate means for replacement)
data = pd.read_csv("data/diabetes.csv")

# Load the trained model and scaler
model = pickle.load(open("models/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# OpenAI functions (using gpt-3.5-turbo)
def explain_prediction(prediction, input_data):
    prompt = f"""
    A diabetes prediction model returned {prediction} (0 means no diabetes, 1 means diabetes) for a patient with:
    Age: {input_data['Age']}, Glucose: {input_data['Glucose']}, BMI: {input_data['BMI']}.
    Explain this result in simple terms for a patient, including possible next steps.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def diabetes_chatbot(question):
    prompt = f"""
    You are a medical assistant specializing in diabetes. Answer the following question in simple language:
    Question: {question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant specializing in diabetes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# Streamlit application
st.title("Diabetes Risk Prediction")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Blood Glucose Level (mg/dL)", 0, 200, 0)
blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 0)
skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 0)
insulin = st.number_input("Insulin Level (µU/mL)", 0, 900, 0)
bmi = st.number_input("BMI (kg/m²)", 0.0, 60.0, 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.0)
age = st.number_input("Age (years)", 18, 100, 18)

# Prepare the input data
input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
input_df = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Replace zero values with the mean
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    if input_df[column][0] == 0:
        input_df[column] = data[column].mean()

# Normalize the input data
input_data_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]  # Use input_df with feature names
    st.write("Result:", "Diabetes Present" if prediction == 1 else "No Diabetes")

    # Explanation using OpenAI
    input_dict = {'Age': age, 'Glucose': glucose, 'BMI': bmi}
    try:
        explanation = explain_prediction(prediction, input_dict)
        st.write("Explanation:", explanation)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")

# Chatbot functionality
st.subheader("Ask a Question About Diabetes")
question = st.text_input("Your Question:")
if st.button("Get Answer"):
    try:
        answer = diabetes_chatbot(question)
        st.write("Answer:", answer)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")