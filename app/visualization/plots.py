"""
Visualization functions for F1 predictions
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_qualifying_vs_predicted(results_df):
    """
    Create scatter plot of qualifying times vs predicted race times
    
    Args:
        results_df (pandas.DataFrame): Race prediction results
        
    Returns:
        None: Displays the plot directly in Streamlit
    """
    st.subheader("Predicted vs Qualifying Times")
    
    fig = px.scatter(
        results_df,
        x='QualifyingTime',
        y='PredictedRaceTime',
        hover_data=['Driver', 'Team'],
        title="Qualifying Time vs Predicted Race Time",
        labels={
            'QualifyingTime': 'Qualifying Time (s)',
            'PredictedRaceTime': 'Predicted Race Time (s)'
        }
    )
    fig.update_traces(marker=dict(size=10, color='#e10600'))
    st.plotly_chart(fig, use_container_width=True)


def plot_position_changes(results_df):
    """
    Create bar chart of position changes from qualifying to race
    
    Args:
        results_df (pandas.DataFrame): Race prediction results
        
    Returns:
        None: Displays the plot directly in Streamlit
    """
    st.subheader("Position Changes")
    
    # Check if we have the necessary columns
    if 'Position' not in results_df.columns:
        st.warning("Position data not available for visualization")
        return
        
    # Create a copy and add qualifying position if not present
    results_df_copy = results_df.copy()
    
    # Sort by qualifying time to get qualifying position
    if 'QualifyingTime' in results_df_copy.columns:
        # Create qualifying position based on qualifying time
        qualifying_order = results_df_copy.sort_values('QualifyingTime').reset_index(drop=True)
        qualifying_position_map = {driver: pos+1 for pos, driver in enumerate(qualifying_order['Driver'])}
        results_df_copy['QualifyingPosition'] = results_df_copy['Driver'].map(qualifying_position_map)
        
        # Use actual position as predicted position
        results_df_copy['PredictedPosition'] = results_df_copy['Position']
        
        # Calculate position changes
        results_df_copy['PositionChange'] = results_df_copy['QualifyingPosition'] - results_df_copy['PredictedPosition']
    else:
        # If no qualifying time, we can't calculate position change
        st.warning("Qualifying time data not available for position change visualization")
        return
    
    fig = px.bar(
        results_df_copy,
        x='Driver',
        y='PositionChange',
        title="Predicted Position Change from Qualifying",
        labels={'PositionChange': 'Position Change'},
        color='PositionChange',
        color_continuous_scale=['red', 'white', 'green']
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(model, model_type):
    """
    Create bar chart of feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        model_type (str): Type of model ("Basic" or "Advanced")
        
    Returns:
        None: Displays the plot directly in Streamlit
    """
    if not hasattr(model, 'feature_importances_'):
        return
    
    st.subheader("🔍 Feature Importance")
    
    # Get feature names
    if model_type == "Basic":
        feature_names = ['Qualifying Time']
    else:
        feature_names = ['Qualifying Time', 'Team Performance', 'Weather Impact']
    
    importances = model.feature_importances_[:len(feature_names)]
    
    fig = px.bar(
        x=importances,
        y=feature_names,
        orientation='h',
        title="Model Feature Importance",
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig.update_traces(marker_color='#0090ff')
    st.plotly_chart(fig, use_container_width=True)


def display_visualizations(results_df, model):
    """
    Display all visualizations
    
    Args:
        results_df (pandas.DataFrame): Race prediction results
        model: Trained model
        
    Returns:
        None: Displays visualizations directly in Streamlit
    """
    st.header("📈 Analysis & Visualizations")
    
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        plot_qualifying_vs_predicted(results_df)
    
    with viz_cols[1]:
        plot_position_changes(results_df)
    
    # Feature importance (for advanced model)
    if hasattr(model, 'feature_importances_'):
        model_type = "Advanced" if len(model.feature_importances_) > 1 else "Basic"
        plot_feature_importance(model, model_type)
