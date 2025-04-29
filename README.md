Diabetes Risk Prediction Project
Project Overview
This project aims to develop an AI model to predict the risk of Type 2 diabetes based on patient data. The project utilizes the PIMA Indians Diabetes Dataset and employs the Random Forest algorithm to train the model. The trained model is integrated into a user-friendly web application built with Streamlit, allowing users to input their data and receive a diabetes risk prediction. Additionally, the OpenAI API is used to provide easy-to-understand explanations of the prediction results and to power a chatbot that answers diabetes-related questions.
Objectives

Primary Objective: Predict the risk of Type 2 diabetes based on patient medical indicators (e.g., blood glucose level, BMI, age, etc.).
Additional Objectives:
Create an intuitive interface for doctors and patients.
Provide clear explanations of predictions using OpenAI in simple language.
Integrate a chatbot to answer diabetes-related questions.



Technologies Used

Programming Language: Python
Dataset: PIMA Indians Diabetes Dataset
Libraries:
pandas and numpy: For data analysis and manipulation.
scikit-learn: For training the Random Forest model and hyperparameter tuning.
streamlit: For building the web application interface.
openai: For generating explanations and powering the chatbot.
python-dotenv: For managing sensitive information (e.g., API keys).


Algorithm: Random Forest (optimized with GridSearchCV)

Work Done
The project was completed in several stages:

Data Preparation:

Loaded the PIMA Indians Diabetes Dataset (768 rows, 9 columns).
Cleaned the data: Replaced zero values in columns like Glucose, Insulin, and BMI with their respective means.
Normalized the data using StandardScaler for better model performance.


Model Training (train_model.py):

Split the data into 80% training and 20% testing sets.
Used the Random Forest algorithm with hyperparameter tuning via GridSearchCV.
Achieved a model accuracy of approximately 75-76%.
Saved the trained model (diabetes_model.pkl) and the scaler (scaler.pkl) for later use.


Web Application Development (app.py):

Built a simple interface using Streamlit.
Enabled users to input patient data and receive diabetes risk predictions.
Integrated the OpenAI API to provide explanations of predictions in simple language.
Added a chatbot feature to answer diabetes-related questions.


Testing and Validation:

Tested the application with sample data from the dataset.
Analyzed prediction results and OpenAI-generated explanations for accuracy and clarity.



Project Structure
diabetes_prediction/
│
├── data/
│   └── diabetes.csv              # PIMA Indians Diabetes Dataset
├── models/
│   ├── diabetes_model.pkl        # Trained model
│   └── scaler.pkl                # Scaler for normalization
├── .env                          # OpenAI API key
├── requirements.txt              # Required libraries
├── train_model.py                # Script for training the model
└── app.py                        # Streamlit application

Usage Instructions
1. Setup

Clone the repository:git clone <repository-url>
cd diabetes_prediction


Create and activate a virtual environment:python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac


Install the required libraries:pip install -r requirements.txt


Configure the OpenAI API key in the .env file:OPENAI_API_KEY=sk-your-openai-api-key

You can obtain your API key from the OpenAI platform: https://platform.openai.com/account/api-keys

2. Train the Model
To train the model, run the following command:
python train_model.py

This script analyzes the dataset, trains the Random Forest model, and saves it.
3. Run the Application
To launch the Streamlit application, run:
streamlit run app.py

The app will open in your browser at: http://localhost:8501
4. Using the Application

Predict Diabetes Risk:
Enter patient data into the fields (e.g., blood glucose level, BMI, age).
Click the "Predict" button.
View the result ("Diabetes Present" or "No Diabetes") and an explanation provided by OpenAI.


Chatbot Feature:
In the "Ask a Question About Diabetes" section, type your question (e.g., "What are the symptoms of diabetes?").
Click "Get Answer" to receive a response from OpenAI.



5. Sample Test Cases
Sample 1 (Diabetes Present):

Number of Pregnancies: 6
Blood Glucose Level: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin Level: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
Expected Result: "Diabetes Present"

Sample 2 (No Diabetes):

Number of Pregnancies: 1
Blood Glucose Level: 85
Blood Pressure: 66
Skin Thickness: 29
Insulin Level: 0
BMI: 26.6
Diabetes Pedigree Function: 0.351
Age: 31
Expected Result: "No Diabetes"

Future Plans

Improve model accuracy by experimenting with algorithms like XGBoost or Gradient Boosting.
Collect additional data to enhance the model's performance.
Optimize OpenAI explanations for better clarity and localization.
Develop a mobile app version (Android/iOS).

Notes

When using the OpenAI API, ensure your API key is active and monitor usage costs.
Do not upload the .env file to public repositories (e.g., GitHub) as it contains sensitive information.

Author

Marufjon (Project Author)
For questions or assistance, feel free to contact me at email@example.com.

