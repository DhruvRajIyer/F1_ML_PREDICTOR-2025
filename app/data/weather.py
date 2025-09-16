"""
Weather data handling for F1 predictions
"""

def get_race_weather_conditions(race_name):
    """
    Get historical weather conditions for a race based on past seasons
    
    Args:
        race_name (str): Name of the race location
        
    Returns:
        dict: Weather conditions including temperature and rain probability
    """
    # Historical weather data based on actual race conditions from recent years
    # Temperature in Celsius, rain_probability as percentage chance based on historical races
    weather_conditions = {
        'australian grand prix': {'temp': 23, 'rain_probability': 0.15},  # Melbourne average in April
        'miami grand prix': {'temp': 28, 'rain_probability': 0.30},       # Miami average in May
        'monaco grand prix': {'temp': 22, 'rain_probability': 0.25},      # Monaco average in May
        'british grand prix': {'temp': 19, 'rain_probability': 0.55},     # Silverstone average in July
        'hungarian grand prix': {'temp': 27, 'rain_probability': 0.20},   # Budapest average in July
        'belgian grand prix': {'temp': 20, 'rain_probability': 0.45},     # Spa average in August
        'italian grand prix': {'temp': 25, 'rain_probability': 0.10},     # Monza average in September
        'singapore grand prix': {'temp': 30, 'rain_probability': 0.60},   # Singapore average in September
        'united states grand prix': {'temp': 24, 'rain_probability': 0.25}, # Austin average in October
        'brazilian grand prix': {'temp': 26, 'rain_probability': 0.35},   # Sao Paulo average in November
    }
    
    # Default values if race not found
    default_weather = {'temp': 24, 'rain_probability': 0.30}
    
    # Try to match by full name first
    if race_name.lower() in weather_conditions:
        return weather_conditions[race_name.lower()]
    
    # Try to match by partial name
    for key in weather_conditions:
        if any(location in race_name.lower() for location in key.split()):
            return weather_conditions[key]
    
    return default_weather
