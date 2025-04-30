import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the Random Forest and XGBoost models
@st.cache_resource
def load_rf_model():
    with open('random_forest_model_titanic.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_xgb_model():
    with open('xgboost_model_titanic.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Titanic Survival Prediction App")

# Model selection
model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, step=1.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Prediction button and logic
if st.button("Predict"):
    if age == 0.0:
        st.error("Please provide a non-zero value for age.")
    elif fare == 0.0:
        st.error("Please provide a non-zero value for fare.")
    else:
        # Map the 'Sex' and 'Embarked' columns to numeric values
        sex = 1 if sex == "female" else 0
        embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame(
            [[pclass, sex, age, sibsp, parch, fare, embarked]],
            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        )

        # Load the selected model and make prediction
        if model_option == "Random Forest":
            model = load_rf_model()
        else:
            model = load_xgb_model()

        # Make the prediction
        prediction = model.predict(input_data)

        # Display the prediction
        if prediction[0] == 1:
            st.write("The passenger survived!")
        else:
            st.write("The passenger did not survive.")
