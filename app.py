import streamlit as st
import pandas as pd
import numpy as np
import pickle
import openai
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Please check OPENAI_API_KEY in the .env file.")
    st.stop()

# Load datasets (to calculate means for replacement)
diabetes_data = pd.read_csv("data/diabetes.csv")
heart_data = pd.read_csv("data/heart_disease_uci.csv")
mass_train = pd.read_csv("data/mass_case_description_train_set.csv")

# Load the trained models and scaler
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
breast_model = pickle.load(open("models/breast_cancer_model.pkl", "rb"))
pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")
brain_tumor_model = tf.keras.models.load_model("models/brain_tumor_model.h5")

# Load separate scalers for each model
scaler_diabetes = pickle.load(open("models/scaler_diabetes.pkl", "rb"))
scaler_heart = pickle.load(open("models/scaler_heart.pkl", "rb"))
scaler_breast = pickle.load(open("models/scaler_breast.pkl", "rb"))

# OpenAI functions for explanations and chatbot (using old API method)
def explain_prediction(disease, prediction, input_data):
    prompt = f"""
    A {disease} prediction model returned {prediction} for a patient with:
    Age: {input_data['Age']}, Glucose: {input_data.get('Glucose', 'N/A')}, BMI: {input_data.get('BMI', 'N/A')}.
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
    return response.choices[0].message['content'].strip()

def medical_chatbot(question):
    prompt = f"""
    You are a medical assistant specializing in various diseases. Answer the following question in simple language:
    Question: {question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant specializing in various diseases."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message['content'].strip()

# Streamlit application
st.title("MedAI: Universal Medical Diagnosis System")

# Select prediction type
prediction_type = st.selectbox("Select Prediction Type", ["Tabular Data", "Image-Based"])

if prediction_type == "Tabular Data":
    # Select disease to predict
    disease = st.selectbox("Select Disease to Predict", ["Diabetes", "Heart Disease", "Breast Cancer"])

    if disease == "Diabetes":
        st.subheader("Enter Patient Data for Diabetes Prediction")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", 0, 200, 0)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 0)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 0)
        insulin = st.number_input("Insulin Level (µU/mL)", 0, 900, 0)
        bmi = st.number_input("BMI (kg/m²)", 0.0, 60.0, 0.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.0)
        age = st.number_input("Age (years)", 18, 100, 18)

        # Prepare the input data for diabetes
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        input_df = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Replace zero values with the mean
        for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            if input_df[column][0] == 0:
                input_df[column] = diabetes_data[column].mean()

        # Normalize the input data
        input_data_scaled = scaler_diabetes.transform(input_df)

        # Prediction for diabetes
        if st.button("Predict Diabetes Risk"):
            prediction = diabetes_model.predict(input_data_scaled)[0]
            st.write("Result:", "Diabetes Present" if prediction == 1 else "No Diabetes")

            # Explanation using OpenAI
            input_dict = {'Age': age, 'Glucose': glucose, 'BMI': bmi}
            try:
                explanation = explain_prediction("diabetes", prediction, input_dict)
                st.write("Explanation:", explanation)
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")

    elif disease == "Heart Disease":
        st.subheader("Enter Patient Data for Heart Disease Prediction")
        age = st.number_input("Age (years)", 18, 100, 18)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 0, 200, 0)
        chol = st.number_input("Cholesterol (mg/dL)", 0, 600, 0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", 0, 220, 0)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 0.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

        # Check if required columns are present in heart_data
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        missing_columns = [col for col in required_columns if col not in heart_data.columns]
        if missing_columns:
            st.error(f"Missing columns in heart data: {', '.join(missing_columns)}")
            st.stop()

        # Prepare the input data for heart disease
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        input_df = pd.DataFrame(input_data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        # Replace zero values with the mean
        for column in ['trestbps', 'chol', 'thalach']:
            if input_df[column][0] == 0:
                input_df[column] = heart_data[column].mean()

        # Normalize the input data
        input_data_scaled = scaler_heart.transform(input_df)

        # Prediction for heart disease
        if st.button("Predict Heart Disease Risk"):
            prediction = heart_model.predict(input_data_scaled)[0]
            prediction_label = ["No Heart Disease", "Mild", "Moderate", "Severe", "Critical"][prediction]
            st.write("Result:", prediction_label)

            # Explanation using OpenAI
            input_dict = {'Age': age, 'Glucose': 'N/A', 'BMI': 'N/A'}
            try:
                explanation = explain_prediction("heart disease", prediction_label, input_dict)
                st.write("Explanation:", explanation)
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")

    elif disease == "Breast Cancer":
        st.subheader("Enter Patient Data for Breast Cancer Prediction")
        breast_density = st.number_input("Breast Density", 1, 4, 1)
        mass_shape = st.selectbox("Mass Shape", ["ROUND", "OVAL", "LOBULATED", "IRREGULAR"], index=0)
        mass_margin = st.selectbox("Mass Margin", ["CIRCUMSCRIBED", "OBSCURED", "MICROLOBULATED", "ILL_DEFINED", "SPICULATED"], index=0)

        # Convert categorical variables to numerical
        mass_shape_mapping = {"ROUND": 0, "OVAL": 1, "LOBULATED": 2, "IRREGULAR": 3}
        mass_margin_mapping = {"CIRCUMSCRIBED": 0, "OBSCURED": 1, "MICROLOBULATED": 2, "ILL_DEFINED": 3, "SPICULATED": 4}
        mass_shape_numeric = mass_shape_mapping[mass_shape]
        mass_margin_numeric = mass_margin_mapping[mass_margin]

        # Prepare the input data for breast cancer
        input_data = [[breast_density, mass_shape_numeric, mass_margin_numeric]]
        input_df = pd.DataFrame(input_data, columns=['breast density', 'mass shape', 'mass margin'])

        # Map input feature names to match those used during scaler training
        input_df.rename(columns={
            'breast density': 'breast_density',
            'mass margin': 'mass margins'
        }, inplace=True)

        # Normalize the input data
        input_data_scaled = scaler_breast.transform(input_df)

        # Prediction for breast cancer
        if st.button("Predict Breast Cancer Risk"):
            prediction = breast_model.predict(input_data_scaled)[0]
            st.write("Result:", "Malignant (Cancer Present)" if prediction == 1 else "Benign (No Cancer)")

            # Explanation using OpenAI
            input_dict = {'Age': 'N/A', 'Glucose': 'N/A', 'BMI': 'N/A'}
            try:
                explanation = explain_prediction("breast cancer", prediction, input_dict)
                st.write("Explanation:", explanation)
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")

else:  # Image-Based
    # Select disease to predict
    image_disease = st.selectbox("Select Disease to Predict", ["Pneumonia (Chest X-Ray)", "Brain Tumor (MRI)"])

    # Upload image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        if image_array.shape[-1] != 3:  # Ensure the image has 3 channels (RGB)
            image_array = np.stack([image_array[:, :, 0]] * 3, axis=-1)
        image_array = image_array / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        if image_disease == "Pneumonia (Chest X-Ray)":
            # Prediction for pneumonia
            prediction = pneumonia_model.predict(image_array)[0][0]
            prediction_label = "Pneumonia Present" if prediction > 0.5 else "No Pneumonia"
            st.write("Result:", prediction_label)

            # Explanation using OpenAI
            input_dict = {'Age': 'N/A', 'Glucose': 'N/A', 'BMI': 'N/A'}
            try:
                explanation = explain_prediction("pneumonia", "1" if prediction > 0.5 else "0", input_dict)
                st.write("Explanation:", explanation)
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")

        elif image_disease == "Brain Tumor (MRI)":
            # Prediction for brain tumor
            prediction = brain_tumor_model.predict(image_array)[0]
            classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            predicted_class = classes[np.argmax(prediction)]
            st.write("Result:", predicted_class)

            # Explanation using OpenAI
            input_dict = {'Age': 'N/A', 'Glucose': 'N/A', 'BMI': 'N/A'}
            try:
                explanation = explain_prediction("brain tumor", predicted_class, input_dict)
                st.write("Explanation:", explanation)
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")

# Chatbot functionality
st.subheader("Ask a Medical Question")
question = st.text_input("Your Question:")
if st.button("Get Answer"):
    try:
        answer = medical_chatbot(question)
        st.write("Answer:", answer)
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")