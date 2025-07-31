import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Heart Failure Analysis Dashboard")

# --- Global Data Loading ---
# Use Streamlit's cache to load data only once
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

try:
    data = load_data("data/heart_failure_clinical_records_dataset.csv")
except FileNotFoundError:
    st.error("Error: The data file 'data/heart_failure_clinical_records_dataset.csv' was not found. Please make sure it's in the correct directory.")
    st.stop()


# ==============================================================================
# --- App Sidebar for Model Loading and Navigation ---
# ==============================================================================
st.sidebar.header("Model Selection")


@st.cache_resource
def load_model_from_mlflow(dst_path="./app/models"):
    model = mlflow.pyfunc.load_model(dst_path)
    return model

model = load_model_from_mlflow()

if model:
    st.sidebar.success("Model loaded successfully.")
else:
    st.sidebar.warning("No model loaded. Prediction tool will be disabled.")

# Navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose a Section", ["Exploratory Data Analysis", "Interactive Prediction"])


# ==============================================================================
# --- Section 1: Exploratory Data Analysis (EDA) ---
# ==============================================================================
if app_mode == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis of Heart Failure Data")

    st.header("Dataset Overview")
    st.dataframe(data.head())
    st.write(f"**Shape of the data:** {data.shape}")

    st.header("Target Variable Distribution (DEATH_EVENT)")
    col1, col2 = st.columns([1, 2])
    with col1:
        len_live = len(data["DEATH_EVENT"][data.DEATH_EVENT == 0])
        len_death = len(data["DEATH_EVENT"][data.DEATH_EVENT == 1])
        st.metric(label="Living Cases", value=len_live)
        st.metric(label="Deceased Cases", value=len_death)
        st.write("The dataset is imbalanced, with fewer deceased cases.")
    with col2:
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie([len_live, len_death], labels=['Living', 'Died'], autopct='%1.1f%%', startangle=90, explode=[0.1, 0], shadow=True)
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig_pie)

    st.header("Feature Distributions and Correlations")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age Distribution")
        fig_age, ax_age = plt.subplots()
        sns.histplot(data["age"], kde=True, ax=ax_age)
        st.pyplot(fig_age)

    with col4:
        st.subheader("Diabetes vs. Mortality")
        diabetes_data = data.groupby(['diabetes', 'DEATH_EVENT']).size().unstack(fill_value=0)
        fig_diab, ax_diab = plt.subplots()
        diabetes_data.plot(kind='bar', stacked=True, ax=ax_diab, colormap='viridis')
        ax_diab.set_xticklabels(['No Diabetes', 'Has Diabetes'], rotation=0)
        st.pyplot(fig_diab)

    st.header("Feature Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(15, 12))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    st.info("This heatmap shows the Pearson correlation between features. 'time', 'serum_creatinine', and 'ejection_fraction' have the strongest correlation with the 'DEATH_EVENT'.")

# ==============================================================================
# --- Section 2: Interactive Prediction ---
# ==============================================================================
elif app_mode == "Interactive Prediction":
    st.title("Interactive Heart Failure Risk Prediction")

    if not model:
        st.warning("Please load a model using a valid Run ID in the sidebar to use the prediction tool.")
        st.stop()

    st.header("Enter Patient Data")
    # Create a form for better UX
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider('Age', 40, 95, 60)
            anaemia = st.selectbox('Anaemia', [0, 1], help="0 = No, 1 = Yes", format_func=lambda x: "Yes" if x == 1 else "No")
            creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', min_value=23, max_value=7861, value=582)
            diabetes = st.selectbox('Diabetes', [0, 1], help="0 = No, 1 = Yes", format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            ejection_fraction = st.slider('Ejection Fraction (%)', 14, 80, 38)
            high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1], help="0 = No, 1 = Yes", format_func=lambda x: "Yes" if x == 1 else "No")
            platelets = st.number_input('Platelets (kiloplatelets/mL)', min_value=25000.0, max_value=850000.0, value=263358.0, format="%.1f")
            serum_creatinine = st.slider('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.1, 0.1)

        with col3:
            serum_sodium = st.slider('Serum Sodium (mEq/L)', 113, 148, 136)
            sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            smoking = st.selectbox('Smoking', [0, 1], help="0 = No, 1 = Yes", format_func=lambda x: "Yes" if x == 1 else "No")
            time = st.slider('Follow-up period (days)', 4, 285, 130)

        # Submit button for the form
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        # Construct the input DataFrame based on the model's expected features
        # This is robust and handles any model logged with a signature
        try:
            feature_names = model.metadata.get_input_schema().input_names()
            input_dict = {
                'age': float(age), 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
                'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure,
                'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
                'sex': sex, 'smoking': smoking, 'time': time
            }
            # Ensure the DataFrame columns are in the exact order the model expects
            input_df = pd.DataFrame([input_dict])[feature_names]

            # The `model.predict()` call works on RAW, UNSCALED data.
            # The pipeline loaded from MLflow handles scaling internally if it exists.
            prediction = model.predict(input_df)
            print(f"Prediction: {prediction}")
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("Prediction: High risk of a mortality event.")
            else:
                st.success("Prediction: Low risk of a mortality event.")
            
            st.info("Disclaimer: This is an AI-generated prediction based on a model and should not be used for actual medical diagnosis. Consult a healthcare professional.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the model loaded has a valid signature and was trained on the expected features.")