# 2025_f1_predictions

# 🏎️ F1 Predictions 2025 - Machine Learning Model

Welcome to the **F1 Predictions 2025** repository! This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Overview
F1 Predictions 2025 is a Streamlit web app and ML toolkit that predicts race outcomes using:

- FastF1 data (laps, sessions, timing, weather)
- Qualifying performance and engineered features (team priors, grid effects, weather, interactions)
- Ensemble ML models with robust preprocessing

You can run the interactive app locally, inspect tables/plots, and tune between a Basic and Advanced model.

## ✨ Key Features
- Interactive Streamlit app with F1-themed UI (`app.py`, `run_app.py`)
- Two model modes:
  - Basic: Gradient Boosting on qualifying features
  - Advanced: Ensemble (GradientBoosting + RandomForest + Ridge) with learned weights
- Feature engineering for F1 domain: team priors, qualifying advantage/position, grid advantage, weather complexity/adaptation, interactions
- Deterministic confidence scores that reflect field tightness and team reliability
- Robust fallbacks and safe clipping for realistic predictions

## 🧱 Project Structure
```
app/
  data/               # data loaders, weather
  models/             # predictor core (DataProcessor, FeatureEngineer, F1Predictor)
  ui/                 # components and styles for Streamlit
  visualization/      # plot helpers
app.py                # Streamlit app entry
run_app.py            # App runner (python run_app.py)
prediction*.py        # Standalone experiments per race
requirements.txt      # Python dependencies
README.md             # You are here
```

## ⚙️ Quickstart
1) Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the app:
```bash
python run_app.py -p 8502
```
Open your browser at http://localhost:8502

## 🔍 Predictor Internals (app/models/predictor.py)
- `DataProcessor`: imputes (median/mean/KNN) and scales (Robust/Standard/Quantile) features.
- `FeatureEngineer`:
  - Normalizes team names (e.g., "Red Bull" → "Red Bull Racing", "Racing Bulls" → "RB")
  - Builds features: team priors, qualifying advantage/position, grid advantage, normalized qual time, weather complexity/adaptation, interactions
  - Selects features per mode (Basic vs Advanced)
- `F1Predictor`:
  - Trains Basic or Advanced model, learns realistic clipping bounds for race-time predictions from data when possible
  - Predicts and clips to plausible ranges; provides `predict_race_results(...)` with positions and confidence
- `EnsembleF1Predictor` (Advanced):
  - GradientBoosting + RandomForest + Ridge, weighted by adaptive KFold CV scores

## 🧪 Using the library
```python
from app.models.predictor import F1Predictor, predict_race_results

# qualifying_df must have: ['Driver', 'Team', 'QualifyingTime']
predictor = F1Predictor(model_type='advanced')
predictor.train(qualifying_df, weather_data={'rain_probability': 0.2, 'temperature': 24, 'humidity': 55})
pred = predictor.predict(qualifying_df)

results = predict_race_results(qualifying_df, model_type='advanced', model=predictor)
print(results.head())
```

## 🧰 Troubleshooting
- Streamlit not found: activate your venv and `pip install -r requirements.txt`
- macOS Watchdog prompt: optional, speeds up reload (`xcode-select --install`, then `pip install watchdog`)
- FastF1 cache: created automatically at `f1_cache/` (ignored by git). To clear: delete the folder.
- Data merge warnings in app logs: the app will fall back to simulated qualifying data if a session fails to provide expected columns.

## 🗺️ Roadmap
- Track-specific priors (lap counts, SC frequency, overtaking difficulty)
- More strategy-aware features (pit loss, stint degradation)
- Optional model persistence with `save_model` / `load_model`

## 📜 License
MIT

## 🙏 Acknowledgements
We would like to thank the FastF1 API team for providing the data used in this project.

🏎️ **Start predicting F1 races like a data scientist!** 🚀
