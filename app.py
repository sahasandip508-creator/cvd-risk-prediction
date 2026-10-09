
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CVD Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "/content/cvd_xgb_model.pkl"
FEATURE_PATH = "/content/cvd_feature_order.pkl"

final_model = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURE_PATH)

# ============================================================
# RISK LEVEL FUNCTION
# ============================================================

def get_risk_level(probability):

    if probability < 0.20:
        return "Low Risk"

    elif probability < 0.40:
        return "Moderate Risk"

    else:
        return "High Risk"


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_cvd_risk(user_data):

    user_df = pd.DataFrame([user_data])

    # Ensure exact feature order
    user_df = user_df[feature_order]

    # Predict probability of CVD = 1
    probability = final_model.predict_proba(user_df)[0, 1]

    risk_level = get_risk_level(probability)

    return {
        "CVD Probability": round(float(probability), 4),
        "Risk Level": risk_level
    }


# ============================================================
# HEADER
# ============================================================

st.title("❤️ CVD Risk Prediction")

st.write(
    "Enter the following information to estimate cardiovascular "
    "disease (CVD) risk using the trained machine learning model."
)

st.info(
    "This is a research/educational prediction tool and is not "
    "a medical diagnosis."
)

# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("1. Personal Information")

age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=120,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

race = st.selectbox(
    "Race / Ethnicity",
    [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Non-Hispanic Asian",
        "Other / Multiracial"
    ]
)

education = st.selectbox(
    "Highest level of education",
    [
        "Less than 9th grade",
        "9-11th grade",
        "High school graduate / GED",
        "Some college or Associate degree",
        "College graduate",
        "College graduate or above"
    ]
)

income_ratio = st.number_input(
    "Income-to-Poverty Ratio",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="This is the NHANES income-to-poverty ratio, not annual income."
)

# ============================================================
# BODY MEASUREMENTS
# ============================================================

st.header("2. Body Measurements")

weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0,
    step=0.1
)

height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=230.0,
    value=175.0,
    step=0.1
)

# Automatic BMI calculation
height_m = height / 100

if height_m > 0:
    bmi = weight / (height_m ** 2)
else:
    bmi = np.nan

st.write(f"**Calculated BMI:** {bmi:.2f} kg/m²")

waist = st.number_input(
    "Waist circumference (cm)",
    min_value=40.0,
    max_value=200.0,
    value=80.0,
    step=0.1
)

# ============================================================
# BLOOD PRESSURE
# ============================================================

st.header("3. Blood Pressure")

systolic = st.number_input(
    "Systolic blood pressure (mmHg)",
    min_value=70.0,
    max_value=250.0,
    value=120.0,
    step=1.0
)

diastolic = st.number_input(
    "Diastolic blood pressure (mmHg)",
    min_value=40.0,
    max_value=180.0,
    value=80.0,
    step=1.0
)

if diastolic > systolic:
    st.error(
        "Diastolic blood pressure cannot be greater than "
        "systolic blood pressure."
    )

# ============================================================
# HEALTH HISTORY
# ============================================================

st.header("4. Health History")

diabetes = st.radio(
    "Has a doctor or health professional ever told you that you have diabetes?",
    ["No", "Yes"]
)

smoking = st.radio(
    "Have you smoked at least 100 cigarettes in your entire life?",
    ["No", "Yes"]
)

# ============================================================
# PHYSICAL ACTIVITY
# ============================================================

st.header("5. Physical Activity")

paq605 = st.radio(
    "Does your work involve vigorous physical activity?",
    ["No", "Yes"]
)

paq620 = st.radio(
    "Does your work involve moderate physical activity?",
    ["No", "Yes"]
)

paq635 = st.radio(
    "Do you walk or bicycle for transportation for at least 10 minutes continuously?",
    ["No", "Yes"]
)

paq650 = st.radio(
    "Do you do vigorous sports, fitness or recreational activities?",
    ["No", "Yes"]
)

paq665 = st.radio(
    "Do you do moderate sports, fitness or recreational activities?",
    ["No", "Yes"]
)

pad680 = st.number_input(
    "Minutes spent sitting on a typical day",
    min_value=0.0,
    max_value=1440.0,
    value=360.0,
    step=10.0
)

# ============================================================
# LABORATORY INFORMATION
# ============================================================

st.header("6. Laboratory Information")

st.caption(
    "If you do not know a laboratory value, you may leave it unavailable."
)

hba1c_option = st.radio(
    "HbA1c (%)",
    ["Enter value", "Not available"],
    horizontal=True
)

if hba1c_option == "Enter value":
    hba1c = st.number_input(
        "HbA1c value",
        min_value=3.0,
        max_value=20.0,
        value=5.2,
        step=0.1
    )
else:
    hba1c = np.nan


chol_option = st.radio(
    "Total Cholesterol (mg/dL)",
    ["Enter value", "Not available"],
    horizontal=True
)

if chol_option == "Enter value":
    total_cholesterol = st.number_input(
        "Total cholesterol value",
        min_value=50.0,
        max_value=600.0,
        value=180.0,
        step=1.0
    )
else:
    total_cholesterol = np.nan


hdl_option = st.radio(
    "HDL Cholesterol (mg/dL)",
    ["Enter value", "Not available"],
    horizontal=True
)

if hdl_option == "Enter value":
    hdl = st.number_input(
        "HDL cholesterol value",
        min_value=10.0,
        max_value=200.0,
        value=55.0,
        step=1.0
    )
else:
    hdl = np.nan


# ============================================================
# CONVERT ANSWERS TO NHANES CODES
# ============================================================

sex_code = {
    "Male": 1,
    "Female": 2
}[sex]


race_code = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other / Multiracial": 7
}[race]


education_code = {
    "Less than 9th grade": 1,
    "9-11th grade": 2,
    "High school graduate / GED": 3,
    "Some college or Associate degree": 4,
    "College graduate": 5,
    "College graduate or above": 5
}[education]


diabetes_code = {
    "No": 2,
    "Yes": 1
}[diabetes]


smoking_code = {
    "No": 2,
    "Yes": 1
}[smoking]


yes_no_code = {
    "No": 2,
    "Yes": 1
}


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Predict CVD Risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if diastolic > systolic:

        st.error(
            "Please correct the blood pressure values before prediction."
        )

    else:

        # ----------------------------------------------------
        # Create exactly 22 model features
        # ----------------------------------------------------

        user_data = {

            "RIDAGEYR": age,
            "RIAGENDR": sex_code,
            "RIDRETH3": race_code,
            "DMDEDUC2": education_code,
            "INDFMPIR": income_ratio,

            "BMXWT": weight,
            "BMXHT": height,
            "BMXBMI": bmi,
            "BMXWAIST": waist,

            "BPXOSY1": systolic,
            "BPXODI1": diastolic,

            "DIQ010": diabetes_code,
            "SMQ020": smoking_code,

            "PAQ605": yes_no_code[paq605],
            "PAQ620": yes_no_code[paq620],
            "PAQ635": yes_no_code[paq635],
            "PAQ650": yes_no_code[paq650],
            "PAQ665": yes_no_code[paq665],

            "PAD680": pad680,

            "LBXGH": hba1c,
            "LBXTC": total_cholesterol,
            "LBDHDD": hdl
        }

        # ----------------------------------------------------
        # Verify feature count
        # ----------------------------------------------------

        if len(user_data) != 22:

            st.error(
                f"Expected 22 features, but received {len(user_data)}."
            )

        else:

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = predict_cvd_risk(user_data)

            probability = result["CVD Probability"]
            risk = result["Risk Level"]

            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            st.subheader("Prediction Result")

            st.metric(
                "Estimated CVD Probability",
                f"{probability * 100:.2f}%"
            )

            if risk == "Low Risk":

                st.success(f"🟢 {risk}")

            elif risk == "Moderate Risk":

                st.warning(f"🟡 {risk}")

            else:

                st.error(f"🔴 {risk}")

            st.caption(
                "The prediction is generated by the trained XGBoost "
                "pipeline used in this research project."
            )

            st.info(
                "This result is for research/educational purposes only "
                "and should not be used as a medical diagnosis."
            )
