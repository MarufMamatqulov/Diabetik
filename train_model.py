import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Datasetni yuklash
data = pd.read_csv("data/diabetes.csv")

# 0 qiymatlarni o‘rtacha bilan almashtirish
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_replace:
    data[column] = data[column].replace(0, data[column].mean())

# Ma'lumotlarni normalizatsiya qilish
scaler = StandardScaler()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data[features] = scaler.fit_transform(data[features])

# Ma'lumotlarni bo‘lish
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelni o‘qitish (hyperparametrlarni sozlash bilan)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Eng yaxshi modelni tanlash
best_model = grid_search.best_estimator_
print("Eng yaxshi parametrlar:", grid_search.best_params_)

# Test qilish
y_pred = best_model.predict(X_test)
print("Random Forest aniqligi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Model va scaler’ni saqlash
with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model va scaler saqlandi!")