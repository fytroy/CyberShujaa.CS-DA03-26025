import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Function to load the model
@st.cache_data
def load_model(model_path):
    """
    Loads the saved model from the given path.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# Main function to run the Streamlit app
def main():
    """
    The main function that runs the Streamlit application.
    """
    # Set the title and a small description
    st.title("California Housing Price Prediction")
    st.markdown("This app predicts the price of a house in California based on several features.")

    # Load the model
    model = load_model('california_knn_pipeline.pkl')

    if model:
        # Create the user interface for input
        st.sidebar.header("Input Features")

        # Define the input fields for the user
        med_inc = st.sidebar.slider("Median Income", 1.0, 15.0, 3.87)
        house_age = st.sidebar.slider("House Age", 1.0, 52.0, 28.6)
        ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 15.0, 5.4)
        ave_bedrms = st.sidebar.slider("Average Bedrooms", 1.0, 5.0, 1.1)
        population = st.sidebar.slider("Population", 3.0, 37000.0, 1425.0)
        ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
        latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.6)
        longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.5)

        # Create a dictionary from the user's input
        user_input = {
            'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude
        }

        input_df = pd.DataFrame([user_input])

        # Create a button to make a prediction
        if st.sidebar.button("Predict"):
            # Make a prediction
            prediction = model.predict(input_df)

            # Display the prediction
            st.subheader("Prediction")
            st.write(f"The predicted house price is: ${prediction[0]:,.2f}")

            # SHAP Visualization
            st.subheader("Prediction Explanation (SHAP)")
            
            # Extract the preprocessor and the model from the pipeline
            preprocessor = model.named_steps['preprocessor']
            knn_model = model.named_steps['knn']

            # and a background dataset for the explainer
            housing = fetch_california_housing()
            X_train = pd.DataFrame(housing.data, columns=housing.feature_names).iloc[:100] # Using a subset for speed
            
            # transform the background data
            X_train_transformed = preprocessor.transform(X_train)
            
            # Create a SHAP explainer
            explainer = shap.KernelExplainer(knn_model.predict, X_train_transformed)
            
            # Transform the user input
            input_transformed = preprocessor.transform(input_df)

            # Calculate SHAP values for the user input
            shap_values = explainer.shap_values(input_transformed)
            
            # Create a SHAP force plot
            st.write("The plot below shows how each feature contributes to the prediction.")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], input_df.iloc[0,:]))

    # A placeholder for future visualizations
    st.sidebar.markdown("---")
    st.sidebar.header("Visualizations")
    st.sidebar.info("Feature importance and other data insights will be displayed here in future versions.")

if __name__ == '__main__':
    main()
