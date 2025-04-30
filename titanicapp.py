import pickle
import numpy as np
import streamlit as st

# Load Random Forest and XGBoost models
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
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    input_df = pd.DataFrame(input_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    # Preprocessing the input data here, encode 'Sex' and 'Embarked' and handle missing values if necessary

    # Load the selected model and make prediction
    if model_option == "Random Forest":
        model = load_rf_model()
    else:
        model = load_xgb_model()

    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.write("The passenger survived!")
    else:
        st.write("The passenger did not survive.")
