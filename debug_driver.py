"""
Diagnostic script to investigate the 'Driver' error in F1 prediction pipeline
"""

import fastf1
import pandas as pd
import numpy as np
import sys
import traceback
from app.data.loader import create_qualifying_data, get_lap_times, merge_race_data
from app.models.predictor import predict_race_results

def setup_cache(cache_dir="f1_cache"):
    """Enable FastF1 caching to store session data."""
    fastf1.Cache.enable_cache(cache_dir)

def inspect_data_pipeline(year=2024, round_num=3):
    """
    Inspect data structure at each step of the pipeline
    
    Args:
        year (int): Season year
        round_num (int): Race round number
    """
    print(f"\n{'='*50}")
    print(f"INSPECTING DATA PIPELINE: Year={year}, Round={round_num}")
    print(f"{'='*50}")
    
    # Step 1: Load qualifying data
    print("\n1. QUALIFYING DATA:")
    try:
        qualifying_data = create_qualifying_data(year, round_num)
        print(f"   Shape: {qualifying_data.shape}")
        print(f"   Columns: {qualifying_data.columns.tolist()}")
        print(f"   First 3 rows:")
        print(qualifying_data.head(3))
        print(f"   'Driver' column exists: {'Driver' in qualifying_data.columns}")
        print(f"   'Driver' column type: {qualifying_data['Driver'].dtype if 'Driver' in qualifying_data.columns else 'N/A'}")
        print(f"   Number of unique drivers: {qualifying_data['Driver'].nunique() if 'Driver' in qualifying_data.columns else 'N/A'}")
    except Exception as e:
        print(f"   ERROR in qualifying data: {e}")
        print(f"   {traceback.format_exc()}")
        qualifying_data = pd.DataFrame()
    
    # Step 2: Load race session
    print("\n2. RACE SESSION:")
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load()
        print(f"   Event name: {session.event['EventName']}")
        print(f"   Session name: {session.name}")
        print(f"   Number of drivers: {len(session.drivers)}")
        print(f"   Drivers: {session.drivers}")
    except Exception as e:
        print(f"   ERROR loading race session: {e}")
        print(f"   {traceback.format_exc()}")
        session = None
    
    # Step 3: Get lap times
    print("\n3. LAP TIMES DATA:")
    try:
        if session:
            laps_data = get_lap_times(session)
            print(f"   Shape: {laps_data.shape}")
            print(f"   Columns: {laps_data.columns.tolist()}")
            print(f"   First 3 rows:")
            print(laps_data.head(3))
            print(f"   'Driver' column exists: {'Driver' in laps_data.columns}")
            print(f"   'Driver' column type: {laps_data['Driver'].dtype if 'Driver' in laps_data.columns else 'N/A'}")
            print(f"   Number of unique drivers: {laps_data['Driver'].nunique() if 'Driver' in laps_data.columns else 'N/A'}")
        else:
            print("   No race session available")
            laps_data = pd.DataFrame()
    except Exception as e:
        print(f"   ERROR getting lap times: {e}")
        print(f"   {traceback.format_exc()}")
        laps_data = pd.DataFrame()
    
    # Step 4: Merge race data
    print("\n4. MERGED DATA:")
    try:
        if not qualifying_data.empty and not laps_data.empty:
            merged_data = merge_race_data(laps_data, qualifying_data)
            print(f"   Shape: {merged_data.shape}")
            print(f"   Columns: {merged_data.columns.tolist()}")
            print(f"   First 3 rows:")
            print(merged_data.head(3))
            print(f"   'Driver' column exists: {'Driver' in merged_data.columns}")
            print(f"   'Driver' column type: {merged_data['Driver'].dtype if 'Driver' in merged_data.columns else 'N/A'}")
            print(f"   Number of unique drivers: {merged_data['Driver'].nunique() if 'Driver' in merged_data.columns else 'N/A'}")
        else:
            print("   No data available for merging")
            merged_data = pd.DataFrame()
    except Exception as e:
        print(f"   ERROR merging data: {e}")
        print(f"   {traceback.format_exc()}")
        merged_data = pd.DataFrame()
    
    # Step 5: Predict race results
    print("\n5. PREDICTION RESULTS:")
    try:
        if not qualifying_data.empty:
            # Test with both model types
            for model_type in ['basic', 'advanced']:
                print(f"\n   Model type: {model_type}")
                results = predict_race_results(qualifying_data, model_type=model_type)
                print(f"   Shape: {results.shape}")
                print(f"   Columns: {results.columns.tolist()}")
                print(f"   First 3 rows:")
                print(results.head(3))
                print(f"   'Driver' column exists: {'Driver' in results.columns}")
                print(f"   'Confidence' column exists: {'Confidence' in results.columns}")
        else:
            print("   No qualifying data available for prediction")
    except Exception as e:
        print(f"   ERROR predicting results: {e}")
        print(f"   {traceback.format_exc()}")
    
    # Step 6: Test fallback paths
    print("\n6. TESTING FALLBACK PATHS:")
    try:
        # Create empty DataFrame to test fallback
        empty_df = pd.DataFrame()
        print("\n   A. Empty qualifying data fallback:")
        results = predict_race_results(empty_df)
        print(f"   Shape: {results.shape}")
        print(f"   Columns: {results.columns.tolist()}")
        print(f"   'Driver' column exists: {'Driver' in results.columns}")
        print(f"   'Confidence' column exists: {'Confidence' in results.columns}")
        
        # Create minimal DataFrame with just Driver column
        print("\n   B. Minimal qualifying data fallback:")
        minimal_df = pd.DataFrame({'Driver': ['VER', 'HAM', 'LEC']})
        try:
            results = predict_race_results(minimal_df)
            print(f"   Shape: {results.shape}")
            print(f"   Columns: {results.columns.tolist()}")
            print(f"   'Driver' column exists: {'Driver' in results.columns}")
            print(f"   'Confidence' column exists: {'Confidence' in results.columns}")
        except Exception as e:
            print(f"   ERROR with minimal data: {e}")
            print(f"   {traceback.format_exc()}")
            
    except Exception as e:
        print(f"   ERROR testing fallbacks: {e}")
        print(f"   {traceback.format_exc()}")

if __name__ == "__main__":
    # Setup cache
    setup_cache()
    
    # Default values
    year = 2024
    round_num = 3  # Australian GP
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    if len(sys.argv) > 2:
        round_num = int(sys.argv[2])
        
    # Run inspection
    inspect_data_pipeline(year, round_num)
