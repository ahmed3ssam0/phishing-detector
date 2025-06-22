import streamlit as st
import numpy as np
import joblib
from elm_model import ELMClassifier

# Load models
elm_model = joblib.load("elm_model.pkl")
svc_model = joblib.load("svc_model.pkl")
nb_model = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature list
FEATURE_NAMES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
    'Statistical_report'
]

# Sample values
phishingSample = [-1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1,  0,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1, -1]
legitSample = [1,  0, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  0,  0, -1, 1,  1,  0, -1,  1, -1,  1, -1, -1,  0, -1,  1,  1,  1]

# Init session state
if 'input_values' not in st.session_state:
    st.session_state.input_values = [0] * 30  # default to all 0

# Set page config
st.set_page_config(page_title="Phishing Website Detector", layout="wide")
st.title("🛡️ Phishing Website Detection")
st.write("Enter the 30 features of a website and choose a model to classify it.")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose the model:", ["ELM", "SVC", "Naive Bayes"])

# Autofill buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Fill Phishing Sample"):
        st.session_state.input_values = phishingSample
with col2:
    if st.button("Fill Legitimate Sample"):
        st.session_state.input_values = legitSample

# Input form
input_data = []
with st.form("feature_form"):
    st.subheader("Feature Inputs")
    cols = st.columns(3)
    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            val = st.selectbox(
                label=feature,
                options=[-1, 0, 1],
                index=[-1, 0, 1].index(st.session_state.input_values[i]),
                key=feature
            )
            input_data.append(val)
    submitted = st.form_submit_button("Predict")

# Predict on submission
if submitted:
    input_array = np.array(input_data).reshape(1, -1)
    scaled = scaler.transform(input_array)
    try:
        if model_choice == "ELM":
            prediction = elm_model.ELMPredict(scaled)
        elif model_choice == "SVC":
            prediction = svc_model.predict(scaled)
        else:
            prediction = nb_model.predict(scaled)

        label = "Legitimate Website" if prediction[0] == 1 else "Phishing Website"
        st.success(f"**Prediction:** {label}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
