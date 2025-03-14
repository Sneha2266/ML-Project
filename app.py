

import streamlit as st
import pickle
import base64

# Function to set background image and adjust text colors
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        b64_encoded = base64.b64encode(data).decode()
    
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_encoded}");
        background-size: cover;
        color: white; /* Adjust text color to be visible on background */
    }}
    .stTextInput, .stSelectbox, .stNumberInput, .stButton {{
        color: black !important; /* Ensuring form elements are readable */
        background-color: rgba(255, 255, 255, 0.8) !important; /* Light background for contrast */
        border-radius: 5px;
    }}
    label {{
        background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent background for labels */
        color: white !important; /* White text for visibility */
        padding: 5px;
        border-radius: 5px;
    }}
    .stButton>button {{
        font-size: 18px !important;
        font-weight: bold !important;
        color: white !important;
        background-color: #4CAF50 !important; /* Green background */
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
    }}
    .predicted-yield {{
        font-size: 24px !important;
        font-weight: bold !important;
        color: black !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Load the trained model
with open('KNNMODEL.sav', 'rb') as f:
    model = pickle.load(f)

# Load encoders and scaler
le_crop = pickle.load(open('le_crop.sav', 'rb'))
le_season = pickle.load(open('le_season.sav', 'rb'))
le_state = pickle.load(open('le_state.sav', 'rb'))
scaler = pickle.load(open('SD.sav', 'rb'))

def predict_yield(state, season, crop, crop_year, area, production, annual_rainfall, fertilizer, pesticide):
    state_encoded = le_state.transform([[state]])[0]
    season_encoded = le_season.transform([[season]])[0]
    crop_encoded = le_crop.transform([[crop]])[0]
    
    features = [[state_encoded, season_encoded, crop_encoded, crop_year, area, production, annual_rainfall, fertilizer, pesticide]]
    
    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    prediction = model.predict(features_scaled)[0]
    return prediction

def main():
    st.title(":rainbow[CROP YIELD PREDICTION]")
    set_bg("C:/Users/Lenovo/OneDrive/Desktop/PROJECT 2/crop 2.jpeg")  # Set background image

    crop = st.selectbox("🌾 **Crop**", le_crop.classes_)
    crop_year = st.number_input("📅 **Crop Year**", min_value=1997, max_value=2025, step=1)
    season = st.selectbox("🍂 **Season**", le_season.classes_)
    state = st.selectbox("📍 **State**", le_state.classes_)
    area = st.number_input("🌱 **Area (in hectares)**", min_value=0.0)
    production = st.number_input("📦 **Production (in tonnes)**", min_value=0.0)
    annual_rainfall = st.number_input("🌧️ **Annual Rainfall (in mm)**", min_value=0.0)
    fertilizer = st.number_input("🧪 **Fertilizer usage (in kg/ha)**", min_value=0.0)
    pesticide = st.number_input("☠️ **Pesticide usage (in kg/ha)**", min_value=0.0)
    
    if st.button("Predict Yield"):
        try:
            predicted_yield = predict_yield(state, season, crop, crop_year, area, production, annual_rainfall, fertilizer, pesticide)
            st.markdown(f"<div class='predicted-yield'>🌾 Predicted Yield: {predicted_yield:.2f}</div>", unsafe_allow_html=True)
        except ValueError as ve:
            st.error(f"🚨 Invalid input: {ve}. Please enter valid numerical values.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

main()
