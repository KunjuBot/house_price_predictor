import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Predictor")
st.write("""
This app predicts the **Median House Value** in a California block group based on various features.
Adjust the sliders below to see how different factors affect the estimated price!
""")

# Load the saved model and feature names
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        st.error("Model file not found. Please run `train.py` first to generate the model.")
        return None, None

model, feature_names = load_model()

if model:
    st.sidebar.header("Specify House/Neighborhood Details")
    
    # Create input sliders for the user
    def user_input_features():
        MedInc = st.sidebar.slider('Median Income (in tens of thousands)', 0.5, 15.0, 3.5)
        HouseAge = st.sidebar.slider('House Age (years)', 1.0, 55.0, 28.0)
        AveRooms = st.sidebar.slider('Average Rooms per Household', 1.0, 15.0, 5.0)
        AveBedrms = st.sidebar.slider('Average Bedrooms per Household', 0.5, 5.0, 1.0)
        Population = st.sidebar.slider('Population of Block Group', 10.0, 10000.0, 1400.0)
        AveOccup = st.sidebar.slider('Average Occupants per Household', 1.0, 10.0, 3.0)
        Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0)
        Longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -119.0)
        
        data = {
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("User Input Parameters")
    st.write(input_df)

    # Predict
    st.subheader("Predicted House Value")
    
    # The model was trained on targets representing 100,000s of dollars
    prediction = model.predict(input_df)
    predicted_value = prediction[0] * 100000 
    
    st.metric(label="Estimated Price", value=f"${predicted_value:,.2f}")
    
    st.write("---")
    st.write("### How it works")
    st.write("This tool uses a Machine Learning model (Random Forest Regressor) trained on the classic California Housing Dataset. It looks at patterns between neighborhood demographics/house characteristics and historical prices to make its predictions.")
