"""
Diagnostic script to investigate the LapTime error in FastF1 data
"""

import fastf1
import pandas as pd
import sys

def setup_cache(cache_dir="f1_cache"):
    """Enable FastF1 caching to store session data."""
    fastf1.Cache.enable_cache(cache_dir)

def inspect_lap_data(year=2024, round_num=3, session_type="R"):
    """
    Inspect lap data structure and types for a specific session
    
    Args:
        year (int): Season year
        round_num (int): Race round number
        session_type (str): Session type (Q=qualifying, R=race)
    """
    print(f"\n{'='*50}")
    print(f"INSPECTING LAP DATA: Year={year}, Round={round_num}, Session={session_type}")
    print(f"{'='*50}")
    
    try:
        # Load session
        session = fastf1.get_session(year, round_num, session_type)
        session.load()
        
        print(f"\n1. SESSION INFO:")
        print(f"   Event name: {session.event['EventName']}")
        print(f"   Session name: {session.name}")
        print(f"   Date: {session.date}")
        
        # Check if laps data is available
        if session.laps.empty:
            print("\n❌ ERROR: Lap data is empty for this session")
            return
            
        print(f"\n2. LAPS DATA STRUCTURE:")
        print(f"   Total laps: {len(session.laps)}")
        print(f"   Columns: {session.laps.columns.tolist()}")
        
        # Check if LapTime column exists
        if 'LapTime' not in session.laps.columns:
            print("\n❌ ERROR: 'LapTime' column not found in session.laps")
            print(f"   Available columns: {session.laps.columns.tolist()}")
            return
            
        # Get a sample of lap times
        sample_laps = session.laps.head(3)
        
        print(f"\n3. SAMPLE LAP TIMES:")
        for idx, lap in sample_laps.iterrows():
            print(f"   Driver: {lap['Driver']}, Lap: {lap['LapNumber']}, Time: {lap['LapTime']}")
            
        # Check LapTime data type
        laptime_type = type(session.laps['LapTime'].iloc[0])
        print(f"\n4. LAPTIME DATA TYPE: {laptime_type}")
        
        # Try different conversion methods
        print(f"\n5. CONVERSION TESTS:")
        
        # Method 1: dt.total_seconds()
        try:
            test1 = session.laps['LapTime'].dt.total_seconds()
            print(f"   ✅ Method 1 (dt.total_seconds): Success - First value: {test1.iloc[0]}")
        except Exception as e:
            print(f"   ❌ Method 1 (dt.total_seconds): Failed - {e}")
            
        # Method 2: pandas to_numeric
        try:
            test2 = pd.to_numeric(session.laps['LapTime'], errors='coerce')
            print(f"   ✅ Method 2 (pd.to_numeric): Success - First value: {test2.iloc[0]}")
        except Exception as e:
            print(f"   ❌ Method 2 (pd.to_numeric): Failed - {e}")
            
        # Method 3: str conversion then parsing
        try:
            test3 = session.laps['LapTime'].astype(str)
            print(f"   ✅ Method 3 (str conversion): Success - First value: {test3.iloc[0]}")
        except Exception as e:
            print(f"   ❌ Method 3 (str conversion): Failed - {e}")
            
        # Method 4: direct access to components if it's a Timedelta
        try:
            if hasattr(session.laps['LapTime'].iloc[0], 'total_seconds'):
                test4 = session.laps['LapTime'].iloc[0].total_seconds()
                print(f"   ✅ Method 4 (direct total_seconds): Success - Value: {test4}")
            else:
                print(f"   ❌ Method 4 (direct total_seconds): Failed - No total_seconds attribute")
        except Exception as e:
            print(f"   ❌ Method 4 (direct total_seconds): Failed - {e}")
            
        # Check accurate laps
        try:
            accurate_laps = session.laps.pick_accurate()
            print(f"\n6. ACCURATE LAPS:")
            print(f"   Total accurate laps: {len(accurate_laps)}")
            if not accurate_laps.empty:
                print(f"   First accurate lap time: {accurate_laps['LapTime'].iloc[0]}")
        except Exception as e:
            print(f"   ❌ Error getting accurate laps: {e}")
            
        # Check for NaN values
        nan_count = session.laps['LapTime'].isna().sum()
        print(f"\n7. NaN VALUES:")
        print(f"   NaN values in LapTime: {nan_count} ({nan_count/len(session.laps)*100:.1f}%)")
        
        # Test our full data processing pipeline
        print(f"\n8. TESTING FULL PIPELINE:")
        try:
            # Get only accurate laps
            accurate_laps = session.laps.pick_accurate()
            
            # Exclude pit in/out laps
            clean_laps = accurate_laps.loc[
                (~accurate_laps['PitInTime'].notna()) &  # Not a pit in lap
                (~accurate_laps['PitOutTime'].notna())    # Not a pit out lap
            ]
            
            # Select required columns and drop any remaining NaN values
            result_laps = clean_laps[["Driver", "LapTime"]].copy()
            result_laps = result_laps.dropna()
            
            # Convert lap times to seconds
            result_laps["RaceTime"] = result_laps["LapTime"].dt.total_seconds()
            
            print(f"   ✅ Full pipeline successful!")
            print(f"   Original laps: {len(session.laps)}")
            print(f"   Accurate laps: {len(accurate_laps)}")
            print(f"   Clean laps: {len(clean_laps)}")
            print(f"   Final result laps: {len(result_laps)}")
            print(f"   Sample result:")
            print(result_laps.head(3))
            
        except Exception as e:
            print(f"   ❌ Full pipeline failed: {e}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    # Setup cache
    setup_cache()
    
    # Default values
    year = 2024
    round_num = 3  # Australian GP
    session_type = "R"  # Race
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    if len(sys.argv) > 2:
        round_num = int(sys.argv[2])
    if len(sys.argv) > 3:
        session_type = sys.argv[3]
        
    # Run inspection
    inspect_lap_data(year, round_num, session_type)
    
    # Try a different session for comparison
    if len(sys.argv) <= 1:  # Only if no custom args were provided
        print("\n\nTrying a different session for comparison...")
        inspect_lap_data(2024, 6, "R")  # Monaco GP
