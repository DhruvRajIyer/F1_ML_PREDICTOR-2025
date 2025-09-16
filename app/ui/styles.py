"""
CSS styles for F1 predictions Streamlit app
"""

def get_custom_css():
    """
    Returns custom CSS for styling the Streamlit app
    
    Returns:
        str: CSS styles as a string
    """
    return """
    <style>
        .main-header {
            background: linear-gradient(90deg, #e10600 0%, #0090ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: rgba(21, 21, 30, 0.9);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .podium-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem;
            text-align: center;
        }
        
        .gold { border: 2px solid #ffd700; }
        .silver { border: 2px solid #c0c0c0; }
        .bronze { border: 2px solid #cd7f32; }
    </style>
    """
