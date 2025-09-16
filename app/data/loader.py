"""
Data loading and processing functions for F1 predictions
"""

import fastf1
import pandas as pd


def timedelta_to_seconds(timedelta_series):
    """
    Convert a pandas Timedelta series to seconds.
    
    Args:
        timedelta_series (pandas.Series): Series containing timedelta values
        
    Returns:
        pandas.Series: Time in seconds
    """
    return timedelta_series.dt.total_seconds()


def setup_cache(cache_dir="f1_cache"):
    """
    Enable FastF1 caching to store session data.
    
    Args:
        cache_dir (str): Directory to store cache (default: 'f1_cache')
    """
    fastf1.Cache.enable_cache(cache_dir)


def get_available_races(year=2024):
    """
    Retrieve the full event schedule for the season.
    
    Args:
        year (int): F1 season year (default: 2024)
        
    Returns:
        pandas.DataFrame: Event schedule
    """
    return fastf1.get_event_schedule(year)


def load_race_session(year, round_num, session_type="R"):
    """
    Load a race session (e.g., qualifying, race).
    
    Args:
        year (int): Season year
        round_num (int): Race round number
        session_type (str): Session type (e.g., 'Q', 'R', 'FP1')
        
    Returns:
        fastf1.core.Session: Loaded session object
    """
    session = fastf1.get_session(year, round_num, session_type)
    session.load()
    return session


def get_lap_times(session):
    """
    Get clean lap times for a session, excluding pit in/out laps and using FastF1's accuracy filters.
    
    Args:
        session (fastf1.core.Session): Loaded session
        
    Returns:
        pandas.DataFrame: Clean lap times with 'Driver' and 'LapTime'
    """
    # Ensure laps data is loaded
    if session.laps.empty:
        raise ValueError("Lap data is not available for this session. Try another race.")
    
    # Check required columns and handle potential missing data
    required_columns = ["Driver", "LapTime"]
    for col in required_columns:
        if col not in session.laps.columns:
            print(f"WARNING: '{col}' column not found in session.laps. Available columns: {session.laps.columns.tolist()}")
            # Create a minimal DataFrame with the required structure
            return pd.DataFrame(columns=["Driver", "LapTime"])
    
    try:
        # Get only accurate laps (excludes outliers)
        # Use try-except as pick_accurate() might fail on some sessions
        try:
            accurate_laps = session.laps.pick_accurate()
            if accurate_laps.empty:
                print("WARNING: No accurate laps found. Using all laps instead.")
                accurate_laps = session.laps.copy()
        except Exception as e:
            print(f"WARNING: Error using pick_accurate(): {e}. Using all laps instead.")
            accurate_laps = session.laps.copy()
        
        # Exclude pit in/out laps if those columns exist
        if 'PitInTime' in accurate_laps.columns and 'PitOutTime' in accurate_laps.columns:
            clean_laps = accurate_laps.loc[
                (~accurate_laps['PitInTime'].notna()) &  # Not a pit in lap
                (~accurate_laps['PitOutTime'].notna())   # Not a pit out lap
            ]
        else:
            clean_laps = accurate_laps.copy()
        
        # Select required columns and drop any remaining NaN values
        result_laps = clean_laps[["Driver", "LapTime"]].copy()
        result_laps = result_laps.dropna()
        
        # Log the number of laps after cleaning
        print(f"Filtered {len(session.laps)} total laps to {len(result_laps)} clean race laps")
        
        if result_laps.empty:
            print("WARNING: No valid lap times found after filtering. Using a minimal DataFrame.")
            return pd.DataFrame(columns=["Driver", "LapTime"])
            
        return result_laps
        
    except Exception as e:
        print(f"ERROR processing lap times: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=["Driver", "LapTime"])


def create_qualifying_data(year=2024, round_num=21):
    """
    Load real qualifying data for a given race using FastF1.
    
    Args:
        year (int): Season year (default: 2024)
        round_num (int): Race round number (default: 21 for Brazilian GP)
        
    Returns:
        pandas.DataFrame: Drivers with their best qualifying time and team
    """
    try:
        # Use FastF1's get_session to load qualifying session data
        session = fastf1.get_session(year, round_num, "Q")
        session.load()
        
        # Check if session loaded successfully
        if not hasattr(session, 'laps') or session.laps.empty:
            print(f"WARNING: No qualifying data available for year={year}, round={round_num}")
            # Create a minimal DataFrame with simulated data
            return create_simulated_qualifying_data()

        # Get all laps from the qualifying session
        all_laps = session.laps.copy()
        
        # Check if we have lap times
        if 'LapTime' not in all_laps.columns:
            print(f"WARNING: 'LapTime' column not found in qualifying session data")
            return create_simulated_qualifying_data()
        
        # Pick each driver's best lap (fastest time)
        try:
            best_laps = all_laps.pick_fastest().reset_index(drop=True)
            
            # Check if best_laps is empty
            if best_laps.empty:
                print("WARNING: No fastest laps found in qualifying data")
                return create_simulated_qualifying_data()
                
        except Exception as e:
            print(f"ERROR picking fastest laps: {e}")
            # Use all laps instead
            best_laps = all_laps.copy()
        
        # Convert lap times to seconds - LapTime is confirmed to be pandas Timedelta type
        try:
            if hasattr(best_laps['LapTime'], 'dt'):
                best_laps['QualifyingTime'] = best_laps['LapTime'].dt.total_seconds()
            else:
                # Try direct conversion for individual Timedelta objects
                best_laps['QualifyingTime'] = best_laps['LapTime'].apply(
                    lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
                )
        except Exception as e:
            print(f"ERROR converting qualifying lap times: {e}")
            # Create fallback qualifying times
            best_laps['QualifyingTime'] = [80 + i/10 for i in range(len(best_laps))]
        
        # Create a DataFrame with required columns
        result_df = pd.DataFrame()
        result_df['Driver'] = best_laps['Driver']
        result_df['QualifyingTime'] = best_laps['QualifyingTime']
        
        # Get team names from session data and map to standardized 2024 names
        team_mapping_2024 = {
            'Red Bull Racing Honda RBPT': 'Red Bull Racing',
            'Red Bull Racing': 'Red Bull Racing',
            'Oracle Red Bull Racing': 'Red Bull Racing',
            'Mercedes': 'Mercedes',
            'Mercedes-AMG Petronas': 'Mercedes',
            'Ferrari': 'Ferrari',
            'Scuderia Ferrari': 'Ferrari',
            'McLaren Mercedes': 'McLaren',
            'McLaren': 'McLaren',
            'Aston Martin Aramco Mercedes': 'Aston Martin',
            'Aston Martin': 'Aston Martin',
            'Alpine Renault': 'Alpine',
            'Alpine': 'Alpine',
            'Williams Mercedes': 'Williams',
            'Williams': 'Williams',
            'RB Honda RBPT': 'RB',
            'RB': 'RB',
            'Visa Cash App RB Honda': 'RB',
            'Haas Ferrari': 'Haas F1 Team',
            'Haas': 'Haas F1 Team',
            'MoneyGram Haas F1 Team': 'Haas F1 Team',
            'Kick Sauber Ferrari': 'Sauber',
            'Sauber': 'Sauber',
            'Stake F1 Team Kick Sauber': 'Sauber'
        }
        
        # Get team names
        teams = []
        for drv in best_laps['Driver']:
            try:
                drv_info = session.get_driver(drv)
                team_name = drv_info.get('TeamName', 'Unknown')
                # Map to standardized team name if available
                standardized_team = team_mapping_2024.get(team_name, team_name)
                teams.append(standardized_team)
            except Exception as e:
                print(f"ERROR getting team for driver {drv}: {e}")
                teams.append('Unknown')
        
        result_df['Team'] = teams
        
        # Return DataFrame with ['Driver', 'Team', 'QualifyingTime']
        return result_df[['Driver', 'Team', 'QualifyingTime']]
        
    except Exception as e:
        print(f"ERROR in create_qualifying_data: {e}")
        return create_simulated_qualifying_data()


def create_simulated_qualifying_data():
    """
    Create simulated qualifying data when real data is unavailable
    
    Returns:
        pandas.DataFrame: Simulated qualifying data
    """
    print("Creating simulated qualifying data as fallback")
    
    # Create a DataFrame with simulated data for top F1 drivers
    drivers = ['VER', 'LEC', 'NOR', 'PIA', 'SAI', 'HAM', 'RUS', 'PER', 'ALO', 'STR', 
              'GAS', 'OCO', 'HUL', 'MAG', 'ALB', 'SAR', 'TSU', 'ZHO', 'BOT', 'LAW']
    
    teams = ['Red Bull Racing', 'Ferrari', 'McLaren', 'McLaren', 'Ferrari', 'Mercedes', 
            'Mercedes', 'Red Bull Racing', 'Aston Martin', 'Aston Martin', 
            'Alpine', 'Alpine', 'Haas F1 Team', 'Haas F1 Team', 'Williams', 
            'Williams', 'RB', 'Sauber', 'Sauber', 'RB']
    
    # Create qualifying times with realistic gaps (80-85 seconds range)
    base_time = 80.0
    quali_times = [base_time + i*0.2 + (0.1 * (i % 3)) for i in range(len(drivers))]
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'Driver': drivers[:len(quali_times)],
        'Team': teams[:len(quali_times)],
        'QualifyingTime': quali_times
    })
    
    return result_df


def merge_race_data(laps_data, qualifying_data):
    """
    Merge average race lap time per driver into qualifying data.
    
    Args:
        laps_data (pandas.DataFrame): Full lap time data from race session
        qualifying_data (pandas.DataFrame): DataFrame with real qualifying data
        
    Returns:
        pandas.DataFrame: Data used for model training with 'Driver', 'Team', 'QualifyingTime', 'RaceTime'
    """
    try:
        # Check if laps_data has the required structure
        if laps_data.empty:
            print("WARNING: Race lap data is empty. Using simulated race times instead.")
            # Create simulated race times based on qualifying times
            merged = qualifying_data.copy()
            # Apply a typical multiplier to simulate race times (about 1.3x qualifying time per lap * ~60 laps)
            merged['RaceTime'] = merged['QualifyingTime'] * 78.0
            return merged
            
        # Check if LapTime column exists
        if 'LapTime' not in laps_data.columns:
            print(f"WARNING: 'LapTime' column not found in race data. Available columns: {laps_data.columns.tolist()}")
            # Create simulated race times
            merged = qualifying_data.copy()
            merged['RaceTime'] = merged['QualifyingTime'] * 78.0
            return merged
        
        # Drop rows with missing lap times
        laps_data = laps_data.dropna(subset=["LapTime"])
        
        if laps_data.empty:
            print("WARNING: No valid lap times found after filtering. Using simulated race times.")
            merged = qualifying_data.copy()
            merged['RaceTime'] = merged['QualifyingTime'] * 78.0
            return merged
        
        # Convert lap times to seconds - LapTime is confirmed to be pandas Timedelta type
        try:
            # Ensure we're working with Timedelta objects
            if not laps_data.empty and hasattr(laps_data["LapTime"], 'dt'):
                laps_data["RaceTime"] = laps_data["LapTime"].dt.total_seconds()
            elif not laps_data.empty:
                # If for some reason it's not a Series with dt accessor
                print(f"WARNING: LapTime is not a Series with dt accessor. Type: {type(laps_data['LapTime'])}")
                # Try direct conversion if it's a single Timedelta object
                laps_data["RaceTime"] = laps_data["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, 'total_seconds') else float(x)
                )
        except Exception as e:
            print(f"ERROR converting lap times to seconds: {e}")
            print(f"LapTime type: {type(laps_data['LapTime'].iloc[0]) if not laps_data.empty else 'N/A'}")
            # Fall back to simulated data
            merged = qualifying_data.copy()
            merged['RaceTime'] = merged['QualifyingTime'] * 78.0
            return merged
        
        # Calculate average race time per driver
        avg_race_time = laps_data.groupby("Driver")["RaceTime"].mean().reset_index()
        
        # Track drivers before merge
        qualifying_drivers = set(qualifying_data["Driver"].unique())
        race_drivers = set(avg_race_time["Driver"].unique())
        
        # Merge with qualifying data
        merged = pd.merge(qualifying_data, avg_race_time, on="Driver", how="inner")
        
        # If merge resulted in empty DataFrame, use left join instead
        if merged.empty:
            print("WARNING: Inner join resulted in empty DataFrame. Using left join with simulated race times.")
            merged = pd.merge(qualifying_data, avg_race_time, on="Driver", how="left")
            # Fill missing race times with simulated values
            merged['RaceTime'] = merged['RaceTime'].fillna(merged['QualifyingTime'] * 78.0)
        
        # Check which drivers were dropped after merging
        merged_drivers = set(merged["Driver"].unique())
        dropped_qualifying = qualifying_drivers - merged_drivers
        dropped_race = race_drivers - merged_drivers
        
        # Print warnings for dropped drivers
        if dropped_qualifying:
            print(f"WARNING: {len(dropped_qualifying)} drivers from qualifying data were dropped after merging:")
            print(", ".join(sorted(dropped_qualifying)))
        
        if dropped_race:
            print(f"WARNING: {len(dropped_race)} drivers from race data were dropped after merging:")
            print(", ".join(sorted(dropped_race)))
        
        # Print summary of merge results
        print(f"Successfully merged data for {len(merged)} drivers")
        
        # Ensure the DataFrame contains the required columns
        required_columns = ['Driver', 'Team', 'QualifyingTime', 'RaceTime']
        for col in required_columns:
            if col not in merged.columns:
                print(f"WARNING: Required column '{col}' missing after merge. Adding default values.")
                if col == 'RaceTime' and 'QualifyingTime' in merged.columns:
                    merged[col] = merged['QualifyingTime'] * 78.0
                else:
                    merged[col] = 0.0
                
        return merged
        
    except Exception as e:
        print(f"ERROR in merge_race_data: {e}")
        # Return a minimal DataFrame with the required structure
        if not qualifying_data.empty:
            result = qualifying_data.copy()
            result['RaceTime'] = result['QualifyingTime'] * 78.0
            return result
        else:
            # Create an empty DataFrame with the required columns
            return pd.DataFrame(columns=['Driver', 'Team', 'QualifyingTime', 'RaceTime'])
