import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("clinical-trial-app/model/random_forest.pkl")
preprocessor = joblib.load("clinical-trial-app/model/preprocessor.pkl")

st.set_page_config(page_title="Clinical Trial Outcome Predictor", layout="wide")
st.title("🔬 Clinical Trial Outcome Prediction")
st.markdown("Use this app to predict the success of a clinical trial based on study parameters.")

# UI inputs
st.sidebar.header("Enter Trial Information")

phase = st.sidebar.selectbox("Trial Phase", options=[2, 3, 4])
sponsor_type = st.sidebar.selectbox("Sponsor Type", options=["INDUSTRY", "NIH", "OTHER"])
gender = st.sidebar.selectbox("Gender", options=["MALE", "FEMALE", "ALL"])
condition = st.sidebar.selectbox("Condition", options=[
    "Cardiovascular Diseases", "Coronary Disease", "Diabetes Mellitus", 
    "Diabetes Mellitus, Type 2", "Hypercholesterolemia", 
    "Hypertension", "Prostate Cancer", "others"
])
location = st.sidebar.selectbox("Trial Location", options=["United States", "Canada"])
enrollment = st.sidebar.number_input("Enrollment (Number of Participants)", min_value=1)
duration = st.sidebar.number_input("Trial Duration (in days)", min_value=1)

# Predict button
if st.sidebar.button("Predict Outcome"):
    # Prepare input DataFrame
    input_df = pd.DataFrame.from_dict({
        "phase": [phase],
        "sponsor_type": [sponsor_type],
        "gender": [gender],
        "condition": [condition],
        "location": [location],
        "enrollment": [enrollment],
        "duration": [duration]
    })

    try:
        # Encode inputs using saved preprocessor
        transformed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_input)[0]

        # Display result with custom styling
        if prediction == 1:
            st.markdown(
                """<div style='background-color: #d4edda; padding: 15px; border-radius: 10px; color: #155724; font-weight: bold;'>
                ✅ Predicted Clinical Trial Outcome: Success
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div style='background-color: #f8d7da; padding: 15px; border-radius: 10px; color: #721c24; font-weight: bold;'>
                ❌ Predicted Clinical Trial Outcome: Failure
                </div>""",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")

# Styling tweaks
st.markdown("""
<style>
    .stApp { background-color: #f7f9fb; }
    .stButton > button {
        background-color: #0072C6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
