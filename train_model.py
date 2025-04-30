import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

# 1. Train Tabular Data Models (Diabetes, Heart Disease, Breast Cancer)

# Load datasets
diabetes_data = pd.read_csv("data/diabetes.csv")
heart_data = pd.read_csv("data/heart_disease_uci.csv")
calc_test = pd.read_csv("data/calc_case_description_test_set.csv")
calc_train = pd.read_csv("data/calc_case_description_train_set.csv")
mass_test = pd.read_csv("data/mass_case_description_test_set.csv")
mass_train = pd.read_csv("data/mass_case_description_train_set.csv")

# Clean diabetes data
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_replace:
    diabetes_data[column] = diabetes_data[column].replace(0, diabetes_data[column].mean())

# Clean heart disease data
heart_columns_to_replace = ['chol', 'trestbps', 'thalch']
for column in heart_columns_to_replace:
    heart_data[column] = heart_data[column].replace(0, heart_data[column].mean())

# Drop the 'id' column from heart_data to avoid including it in training
heart_data = heart_data.drop('id', axis=1, errors='ignore')
# Drop the 'dataset' column from heart_data to avoid invalid data types
heart_data = heart_data.drop('dataset', axis=1, errors='ignore')

# Prepare breast cancer data (combine calc and mass datasets)
calc_data = pd.concat([calc_train, calc_test], ignore_index=True)
mass_data = pd.concat([mass_train, mass_test], ignore_index=True)
# Assuming 'pathology' is the target column (MALIGNANT/BENIGN)
# Select relevant features (you may need to adjust based on your dataset)
breast_features = ['breast_density', 'mass shape', 'mass margins']  # Example features
breast_data = mass_data[breast_features + ['pathology']]
# Drop rows with NaN values in the entire breast_data DataFrame
breast_data = breast_data.dropna()
# Ensure 'pathology' column contains only valid values and drop invalid rows
breast_data = breast_data[breast_data['pathology'].isin(['MALIGNANT', 'BENIGN'])]
# Map 'pathology' values to numeric labels
breast_data['pathology'] = breast_data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0})

# Encode categorical columns in breast_data
breast_data['mass shape'] = breast_data['mass shape'].astype('category').cat.codes
breast_data['mass margins'] = breast_data['mass margins'].astype('category').cat.codes

# Normalize data
scaler_diabetes = StandardScaler()
scaler_heart = StandardScaler()
scaler_breast = StandardScaler()

# Diabetes
diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
diabetes_data[diabetes_features] = scaler_diabetes.fit_transform(diabetes_data[diabetes_features])

# Heart Disease
heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
# Encode categorical columns in heart_data
heart_data['sex'] = heart_data['sex'].map({'Male': 1, 'Female': 0})
heart_data['cp'] = heart_data['cp'].astype('category').cat.codes
heart_data['restecg'] = heart_data['restecg'].astype('category').cat.codes
heart_data['slope'] = heart_data['slope'].astype('category').cat.codes
heart_data['thal'] = heart_data['thal'].astype('category').cat.codes
heart_data[heart_features] = scaler_heart.fit_transform(heart_data[heart_features])

# Breast Cancer
breast_data[breast_features] = scaler_breast.fit_transform(breast_data[breast_features])

# Split data
X_diabetes = diabetes_data.drop('Outcome', axis=1)
y_diabetes = diabetes_data['Outcome']
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

X_heart = heart_data.drop('num', axis=1)
y_heart = heart_data['num']
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

X_breast = breast_data.drop('pathology', axis=1)
y_breast = breast_data['pathology']
X_train_breast, X_test_breast, y_train_breast, y_test_breast = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

# Train models
diabetes_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
diabetes_model.fit(X_train_diabetes, y_train_diabetes)
y_pred_diabetes = diabetes_model.predict(X_test_diabetes)
print("Diabetes Model Accuracy:", accuracy_score(y_test_diabetes, y_pred_diabetes))
print(classification_report(y_test_diabetes, y_pred_diabetes))

heart_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
heart_model.fit(X_train_heart, y_train_heart)
y_pred_heart = heart_model.predict(X_test_heart)
print("Heart Disease Model Accuracy:", accuracy_score(y_test_heart, y_pred_heart))
print(classification_report(y_test_heart, y_pred_heart))

breast_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
breast_model.fit(X_train_breast, y_train_breast)
y_pred_breast = breast_model.predict(X_test_breast)
print("Breast Cancer Model Accuracy:", accuracy_score(y_test_breast, y_pred_breast))
print(classification_report(y_test_breast, y_pred_breast))

# Save models
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(diabetes_model, f)
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(heart_model, f)
with open("models/breast_cancer_model.pkl", "wb") as f:
    pickle.dump(breast_model, f)

# Save separate scalers for each dataset
with open("models/scaler_diabetes.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

with open("models/scaler_heart.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

with open("models/scaler_breast.pkl", "wb") as f:
    pickle.dump(scaler_breast, f)

# 2. Train Image-Based Models (Pneumonia and Brain Tumor)

# Pneumonia Detection (Chest X-Ray)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/images/chest_xray/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'data/images/chest_xray/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build CNN model for pneumonia
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
pneumonia_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

pneumonia_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
pneumonia_model.fit(train_generator, validation_data=val_generator, epochs=5)
pneumonia_model.save("models/pneumonia_model.h5")

# Brain Tumor Classification
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'data/images/testing',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build CNN model for brain tumor
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)  # 4 classes: glioma, meningioma, notumor, pituitary
brain_tumor_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

brain_tumor_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
brain_tumor_model.fit(train_generator, epochs=5)
brain_tumor_model.save("models/brain_tumor_model.h5")

print("All models trained and saved successfully!")