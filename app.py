
"""
F1 Predictions 2025 - Streamlit App
Interactive dashboard for F1 race predictions using machine learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modular components
try:
    from app.data.loader import setup_cache, load_race_session, get_lap_times, create_qualifying_data, merge_race_data
    from app.data.weather import get_race_weather_conditions
    from app.models.predictor import create_basic_model, create_advanced_model, predict_race_results
    from app.ui.styles import get_custom_css
    from app.ui.components import display_header, display_welcome_screen, display_podium, display_metrics, display_results_table, display_footer
    from app.visualization.plots import display_visualizations
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="F1 AI Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Display header
display_header()

# Sidebar
st.sidebar.title("🏎️ Race Settings")

# Race selection
races_2025 = {
    "Australian Grand Prix": {"year": 2024, "round": 3},
    "Miami Grand Prix": {"year": 2024, "round": 6},
    "Monaco Grand Prix": {"year": 2024, "round": 8},
    "British Grand Prix": {"year": 2024, "round": 12},
    "Hungarian Grand Prix": {"year": 2024, "round": 13},
    "Belgian Grand Prix": {"year": 2024, "round": 14},
    "Italian Grand Prix": {"year": 2024, "round": 16},
    "Singapore Grand Prix": {"year": 2024, "round": 18},
    "United States Grand Prix": {"year": 2024, "round": 19},
    "Brazilian Grand Prix": {"year": 2024, "round": 21}
}

selected_race = st.sidebar.selectbox("Select Race", list(races_2025.keys()))
race_info = races_2025[selected_race]

# Model selection
model_type = st.sidebar.radio("Model Type", ["Basic", "Advanced"])

# Feature weights (for advanced model)
feature_weights = {}
if model_type == "Advanced":
    st.sidebar.markdown("### Feature Weights")
    st.sidebar.markdown("Adjust the importance of each feature in the prediction model:")

    feature_weights["qualifying_weight"] = st.sidebar.slider(
        "Qualifying Performance", 
        min_value=0.5, 
        max_value=1.5, 
        value=1.0, 
        step=0.1,
        help="How much qualifying performance affects race results"
    )

    feature_weights["team_weight"] = st.sidebar.slider(
        "Team Performance", 
        min_value=0.5, 
        max_value=1.5, 
        value=1.0, 
        step=0.1,
        help="Impact of team's overall performance"
    )

    feature_weights["weather_weight"] = st.sidebar.slider(
        "Weather Adaptation", 
        min_value=0.5, 
        max_value=1.5, 
        value=1.0, 
        step=0.1,
        help="Driver's ability to adapt to weather conditions"
    )

# Run prediction button
run_prediction = st.sidebar.button("🚀 Run Prediction", type="primary")

# Main content area
if run_prediction:
    with st.spinner("Loading F1 data and training model..."):
        try:
            # Setup cache
            setup_cache()

            # Load race data
            session_2024 = load_race_session(race_info['year'], race_info['round'], "R")
            laps_2024 = get_lap_times(session_2024)

            if laps_2024.empty:
                st.warning("Lap time data for the selected race is not available.")
                st.stop()

            # Create qualifying data from real FastF1 data
            qualifying_2025 = create_qualifying_data(race_info['year'], race_info['round'])

            # Merge data
            merged_data = merge_race_data(laps_2024, qualifying_2025)

            # Create and train model
            if model_type == "Basic":
                model = create_basic_model(merged_data)
                weather_data = None
            else:
                # Get weather data
                weather_data = get_race_weather_conditions(selected_race.lower())
                model = create_advanced_model(merged_data, weather_data)

            # Generate predictions
            results_df = predict_race_results(qualifying_2025, model_type.lower(), model, weather_data)

            # Store in session state
            st.session_state.predictions_data = results_df
            st.session_state.model_trained = True
            st.session_state.model_object = model

        except Exception as e:
            st.error(f"Error running prediction: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.predictions_data is not None:
    results_df = st.session_state.predictions_data

    # Display podium
    display_podium(results_df)

    # Display metrics
    display_metrics(results_df, model_type)

    # Display results table
    display_results_table(results_df)

    # Display visualizations
    display_visualizations(results_df, st.session_state.model_object)
else:
    # Welcome screen
    display_welcome_screen()

# Footer
display_footer()

