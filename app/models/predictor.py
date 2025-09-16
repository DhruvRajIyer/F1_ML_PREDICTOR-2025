
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import joblib

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles data preprocessing, including imputation, scaling, and outlier handling.
    """
    def __init__(self, imputer_strategy='median', scaler_type='robust'):
        self.imputer_strategy = imputer_strategy
        self.scaler_type = scaler_type
        self.imputer = None
        self.scaler = None

    def fit(self, X):
        if self.imputer_strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.imputer_strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif self.imputer_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError("Invalid imputer strategy")
        X = self.imputer.fit_transform(X)

        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError("Invalid scaler type")
        self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.imputer:
            X = self.imputer.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FeatureEngineer:
    """
    Manages feature creation and selection for F1 race prediction.
    """
    def __init__(self):
        self.team_mapping = {
            'Red Bull Racing': {'base': 94, 'consistency': 0.95, 'strategy': 0.92},
            'McLaren': {'base': 91, 'consistency': 0.88, 'strategy': 0.85},
            'Ferrari': {'base': 89, 'consistency': 0.82, 'strategy': 0.88},
            'Mercedes': {'base': 88, 'consistency': 0.90, 'strategy': 0.90},
            'Aston Martin': {'base': 78, 'consistency': 0.75, 'strategy': 0.80},
            'RB': {'base': 72, 'consistency': 0.78, 'strategy': 0.75},
            'Alpine': {'base': 70, 'consistency': 0.70, 'strategy': 0.72},
            'Williams': {'base': 67, 'consistency': 0.85, 'strategy': 0.70},
            'Haas F1 Team': {'base': 65, 'consistency': 0.75, 'strategy': 0.65},
            'Sauber': {'base': 60, 'consistency': 0.70, 'strategy': 0.60}
        }
        self.default_team = {'base': 70, 'consistency': 0.75, 'strategy': 0.70}
        self.wet_specialists = {'Aston Martin', 'Alpine', 'Mercedes'}
        # Common aliases to ensure upstream data maps to internal keys
        self.team_aliases = {
            'Red Bull': 'Red Bull Racing',
            'Racing Bulls': 'RB',
            'Visa Cash App RB': 'RB',
            'Haas': 'Haas F1 Team',
            'Kick Sauber': 'Sauber',
            'Stake F1 Team': 'Sauber',
            'Stake F1 Team Kick Sauber': 'Sauber',
        }

    def _normalize_team(self, team_name):
        try:
            if pd.isna(team_name):
                return team_name
        except Exception:
            pass
        # Map known aliases, else return original
        return self.team_aliases.get(team_name, team_name)

    def create_features(self, data, weather_data=None):
        features_df = data.copy()

        # Normalize team names for robust mapping
        if 'Team' in features_df.columns:
            features_df['Team'] = features_df['Team'].map(self._normalize_team)

        # Team performance features
        features_df['TeamPerformance'] = features_df['Team'].map(
            lambda t: self.team_mapping.get(t, self.default_team)['base']
        )
        features_df['TeamConsistency'] = features_df['Team'].map(
            lambda t: self.team_mapping.get(t, self.default_team)['consistency']
        )
        features_df['TeamStrategy'] = features_df['Team'].map(
            lambda t: self.team_mapping.get(t, self.default_team)['strategy']
        )

        # Qualifying position impact
        if 'QualifyingTime' in features_df.columns and len(features_df) > 1:
            min_qual_time = features_df['QualifyingTime'].min()
            features_df['QualifyingAdvantage'] = (min_qual_time / features_df['QualifyingTime']) ** 2
            features_df['QualifyingPosition'] = features_df['QualifyingTime'].rank(method='min')
            features_df['GridAdvantage'] = np.where(
                features_df['QualifyingPosition'] <= 3, 1.1,
                np.where(features_df['QualifyingPosition'] <= 10, 1.0, 0.95)
            )
            features_df['NormalizedQualTime'] = (
                features_df['QualifyingTime'] / features_df['QualifyingTime'].median()
            )
        else:
            features_df['QualifyingAdvantage'] = 1.0
            features_df['QualifyingPosition'] = 1.0
            features_df['GridAdvantage'] = 1.0
            features_df['NormalizedQualTime'] = 1.0

        # Weather features
        if weather_data:
            rain_prob = weather_data.get('rain_probability', 0.3)
            temp = weather_data.get('temperature', 25)
            humidity = weather_data.get('humidity', 60)

            features_df['WeatherComplexity'] = (
                rain_prob * 0.5 +
                abs(temp - 22) / 30 * 0.3 +
                humidity / 100 * 0.2
            )
            features_df['WeatherAdaptation'] = features_df['Team'].map(
                lambda t: 1.1 if t in self.wet_specialists and rain_prob > 0.3 else 1.0
            )
        else:
            features_df['WeatherComplexity'] = 0.3
            features_df['WeatherAdaptation'] = 1.0

        # Interaction features
        features_df['TeamQualifyingInteraction'] = (
            features_df['TeamPerformance'] * features_df['QualifyingAdvantage']
        )
        features_df['ConsistencyWeatherInteraction'] = (
            features_df['TeamConsistency'] * (1 + features_df['WeatherComplexity'])
        )

        return features_df

    def select_features(self, features_df, model_type='advanced'):
        if model_type == 'basic':
            selected_features = ['QualifyingTime', 'NormalizedQualTime']
        else:
            selected_features = [
                'QualifyingTime', 'NormalizedQualTime', 'QualifyingAdvantage', 'QualifyingPosition',
                'TeamPerformance', 'TeamConsistency', 'TeamStrategy', 'GridAdvantage',
                'WeatherComplexity', 'WeatherAdaptation',
                'TeamQualifyingInteraction', 'ConsistencyWeatherInteraction'
            ]

        # Ensure all selected features exist, fill with default if not
        for feature in selected_features:
            if feature not in features_df.columns:
                features_df[feature] = 1.0  # Default value

        return features_df[selected_features]


class EnsembleF1Predictor(BaseEstimator, RegressorMixin):
    """
    Enhanced ensemble predictor combining multiple algorithms
    with weighted voting based on historical performance
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.fitted = False

    def _create_models(self):
        """Create diverse ensemble of models with different strengths"""
        return {
            'gbr': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.8,
                random_state=self.random_state,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=self.random_state
            )
        }

    def fit(self, X, y):
        """Fit ensemble with automatic weight optimization"""
        self.models = self._create_models()

        cv_scores = {}
        for name, model in self.models.items():
            try:
                # Adaptive KFold for stability on small datasets
                n_splits = 5 if len(X) >= 5 else max(2, len(X))
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
                cv_scores[name] = np.mean(scores)
                model.fit(X, y)
            except Exception:
                cv_scores[name] = -1000
                model.fit(X, y)

        if cv_scores:
            scores_array = np.array(list(cv_scores.values()))
            shifted_scores = scores_array - np.min(scores_array) + 1
            exp_scores = np.exp(shifted_scores / 100)
            self.weights = dict(zip(cv_scores.keys(), exp_scores / np.sum(exp_scores)))
        else:
            self.weights = {name: 1/len(self.models) for name in self.models.keys()}

        self.fitted = True
        return self

    def predict(self, X):
        """Make ensemble predictions with weighted voting"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")

        predictions = {}
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception:
                predictions[name] = np.mean(X[:, 0]) * 78.0 # Fallback

        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred

        return ensemble_pred


class F1Predictor:
    """
    Main class for F1 race prediction, orchestrating data processing, feature engineering, and modeling.
    """
    def __init__(self, model_type='advanced', imputer_strategy='median', scaler_type='robust', random_state=42):
        self.model_type = model_type
        self.imputer_strategy = imputer_strategy
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.data_processor = DataProcessor(imputer_strategy=self.imputer_strategy, scaler_type=self.scaler_type)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.feature_names = None
        # Learned clipping multipliers (low/high) for mapping qualifying -> race time
        self.clip_multiplier_low = 70.0
        self.clip_multiplier_high = 85.0

    def train(self, data, weather_data=None):
        # 1. Feature Engineering
        X_features = self.feature_engineer.create_features(data, weather_data)
        X_selected = self.feature_engineer.select_features(X_features, self.model_type)
        y = data['RaceTime'] if 'RaceTime' in data.columns else data['QualifyingTime'] * 78.0

        # Learn reasonable clipping multipliers from provided ground truth when available
        try:
            if 'RaceTime' in data.columns and 'QualifyingTime' in data.columns:
                valid_mask = data['RaceTime'].notna() & data['QualifyingTime'].notna() & (data['QualifyingTime'] > 0)
                ratios = (data.loc[valid_mask, 'RaceTime'] / data.loc[valid_mask, 'QualifyingTime']).values
                if len(ratios) >= 3:
                    self.clip_multiplier_low = float(np.percentile(ratios, 10))
                    self.clip_multiplier_high = float(np.percentile(ratios, 90))
                elif len(ratios) > 0:
                    r = float(np.median(ratios))
                    self.clip_multiplier_low = max(60.0, 0.9 * r)
                    self.clip_multiplier_high = min(95.0, 1.1 * r)
        except Exception:
            # Keep defaults on any failure
            pass

        # Handle missing values and outliers before scaling
        # This part is now handled by DataProcessor's fit_transform
        X_processed = self.data_processor.fit_transform(X_selected)

        # Remove any infinite or NaN values after processing
        mask = np.isfinite(X_processed).all(axis=1) & np.isfinite(y)
        X_clean = X_processed[mask]
        y_clean = y[mask]

        if len(X_clean) < 3: # Need minimum samples for ensemble
            raise ValueError("Not enough clean data to train the model.")

        # 2. Model Training
        if self.model_type == 'basic':
            self.model = Pipeline([
                ('model', GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.8,
                    validation_fraction=0.15,
                    n_iter_no_change=8,
                    random_state=self.random_state
                ))
            ])
        else:
            self.model = EnsembleF1Predictor(random_state=self.random_state)

        self.model.fit(X_clean, y_clean)
        self.feature_names = self.feature_engineer.select_features(X_features, self.model_type).columns.tolist()

    def predict(self, data, weather_data=None):
        if self.model is None:
            raise ValueError("Model not trained yet. Call .train() first.")

        X_features = self.feature_engineer.create_features(data, weather_data)
        X_selected = self.feature_engineer.select_features(X_features, self.model_type)
        X_processed = self.data_processor.transform(X_selected)

        predictions = self.model.predict(X_processed)

        # Ensure predictions are reasonable
        qmin = float(data['QualifyingTime'].min()) if 'QualifyingTime' in data.columns else 60.0
        qmax = float(data['QualifyingTime'].max()) if 'QualifyingTime' in data.columns else 120.0
        low_bound = qmin * self.clip_multiplier_low
        high_bound = qmax * self.clip_multiplier_high
        predictions = np.clip(predictions, low_bound, high_bound)
        return predictions

    def evaluate(self, X, y):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return {'mse': mse, 'r2': r2}

    def get_feature_importance(self):
        if self.model is None:
            return None
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importances_'): # For Pipeline
            return dict(zip(self.feature_names, self.model.model.feature_importances_))
        return None

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)
        joblib.dump(self.data_processor, path.replace('.joblib', '_processor.joblib'))
        joblib.dump(self.feature_engineer, path.replace('.joblib', '_engineer.joblib'))

    @classmethod
    def load_model(cls, path):
        predictor = cls()
        predictor.model = joblib.load(path)
        predictor.data_processor = joblib.load(path.replace('.joblib', '_processor.joblib'))
        predictor.feature_engineer = joblib.load(path.replace('.joblib', '_engineer.joblib'))
        # Reconstruct feature_names if possible
        if hasattr(predictor.model, 'feature_names'):
            predictor.feature_names = predictor.model.feature_names
        elif hasattr(predictor.feature_engineer, 'selected_features'):
            # This might need adjustment based on how feature_engineer is used during prediction
            # For now, assume it can reconstruct feature names from its internal state
            pass # Feature names will be set during prediction if needed
        return predictor


def calculate_dynamic_confidence(predictions, qualifying_data, model_performance=None):
    """
    Calculate more sophisticated confidence scores based on multiple factors
    """
    n_drivers = len(predictions)

    # Base confidence decreases with position (winners more predictable)
    base_confidence = np.linspace(92, 65, n_drivers)

    # Adjust based on qualifying time gaps
    if 'QualifyingTime' in qualifying_data.columns and len(qualifying_data) > 1:
        qual_times = qualifying_data['QualifyingTime'].values
        time_gaps = np.diff(np.sort(qual_times))
        avg_gap = np.mean(time_gaps) if len(time_gaps) > 0 else 1.0

        # Closer qualifying = lower confidence (more competitive)
        gap_factor = np.clip(avg_gap / 0.5, 0.5, 1.5)  # Normalize around 0.5s gaps
        base_confidence *= gap_factor

    # Add team-based confidence adjustments
    team_reliability = {
        'Red Bull Racing': 1.15, 'Mercedes': 1.10, 'Ferrari': 1.05, 'McLaren': 1.08,
        'Aston Martin': 0.95, 'Alpine': 0.90, 'Williams': 0.85,
        'RB': 0.92, 'Haas F1 Team': 0.88, 'Sauber': 0.85
    }

    confidence_scores = []
    for i, (_, row) in enumerate(qualifying_data.iterrows()):
        team_factor = team_reliability.get(row.get('Team', ''), 0.90)

        # Position-based confidence with team adjustment
        pos_confidence = base_confidence[i] * team_factor

        # Deterministic tiny noise based on driver string for stability
        driver_id = str(row.get('Driver', f'idx{i}'))
        seed = abs(hash(driver_id)) % 10000
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1.5)  # Very small jitter

        final_confidence = np.clip(pos_confidence + noise, 15, 98)
        confidence_scores.append(final_confidence)

    return np.array(confidence_scores)


def format_race_time(seconds):
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)

        if hours > 0:
            return f"{hours:01d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
        else:
            return f"{minutes:02d}:{secs:02d}.{millisecs:03d}"
    except:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def create_basic_model(data, random_state=42):
    """
    Create and train a basic F1 prediction model
    
    Args:
        data (pd.DataFrame): Training data with qualifying times
        random_state (int): Random seed for reproducibility
        
    Returns:
        F1Predictor: Trained basic model
    """
    predictor = F1Predictor(model_type='basic', random_state=random_state)
    try:
        predictor.train(data)
        return predictor
    except Exception as e:
        print(f"Error in basic model creation: {e}")
        # Create a minimal fallback model
        predictor = F1Predictor(model_type='basic', random_state=random_state)
        return predictor


def create_advanced_model(data, weather_data=None, random_state=42):
    """
    Create and train an advanced F1 prediction model with weather data
    
    Args:
        data (pd.DataFrame): Training data with qualifying times
        weather_data (dict): Weather conditions for the race
        random_state (int): Random seed for reproducibility
        
    Returns:
        F1Predictor: Trained advanced model
    """
    predictor = F1Predictor(model_type='advanced', random_state=random_state)
    try:
        predictor.train(data, weather_data)
        return predictor
    except Exception as e:
        print(f"Error in advanced model creation: {e}")
        # Fallback to basic model
        return create_basic_model(data, random_state)


def predict_race_results(qualifying_data, model_type='advanced', model=None, weather_data=None):
    """
    Enhanced race prediction with improved ML models and feature engineering
    """
    try:
        # Input validation
        if qualifying_data is None or qualifying_data.empty:
            raise ValueError("Qualifying data is empty")

        required_columns = ['Driver', 'Team', 'QualifyingTime']
        missing_columns = [col for col in required_columns if col not in qualifying_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if qualifying_data['Driver'].isna().any():
            raise ValueError("Driver column contains empty values")

        # Create working copy
        qualifying_data = qualifying_data.copy()

        # Model creation or loading
        f1_predictor = F1Predictor(model_type=model_type)
        if model is None:
            try:
                f1_predictor.train(qualifying_data, weather_data)
            except Exception as e:
                print(f"Error creating model: {e}")
                # Fallback to a simpler model if training fails
                f1_predictor = F1Predictor(model_type='basic')
                f1_predictor.train(qualifying_data, weather_data) # Try training basic model
        else:
            f1_predictor = model # Use provided pre-trained model

        # Generate predictions
        predictions = f1_predictor.predict(qualifying_data, weather_data)

        # Create results dataframe
        results = pd.DataFrame({
            'Driver': qualifying_data['Driver'],
            'Team': qualifying_data['Team'],
            'QualifyingTime': qualifying_data['QualifyingTime'],
            'PredictedRaceTime': predictions
        })

        results['ReadableRaceTime'] = results['PredictedRaceTime'].apply(format_race_time)

        # Sort by predicted race time and add positions
        results = results.sort_values('PredictedRaceTime').reset_index(drop=True)
        results['Position'] = range(1, len(results) + 1)

        # Enhanced confidence calculation
        confidence_scores = calculate_dynamic_confidence(
            results['PredictedRaceTime'].values,
            qualifying_data,
            model_performance=getattr(f1_predictor.model, 'weights', None)
        )
        results['Confidence'] = confidence_scores

        # Final column ordering
        column_order = ['Position', 'Driver', 'Team', 'QualifyingTime',
                       'PredictedRaceTime', 'ReadableRaceTime', 'Confidence']
        results = results[column_order]

        return results

    except Exception as e:
        print(f"Error in enhanced prediction: {e}")

        # Robust fallback with better error handling
        try:
            results = qualifying_data.copy()

            # Ensure basic columns exist
            if 'QualifyingTime' not in results.columns:
                results['QualifyingTime'] = [90 + i * 0.5 for i in range(len(results))]

            # Simple time multiplication with some variation
            base_multiplier = 78.0
            np.random.seed(42)  # Consistent fallback
            multipliers = np.random.normal(base_multiplier, 2, len(results))
            results['PredictedRaceTime'] = results['QualifyingTime'] * multipliers

            # Format times
            results['ReadableRaceTime'] = results['PredictedRaceTime'].apply(
                lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
            )

            # Sort and add positions
            results = results.sort_values('PredictedRaceTime').reset_index(drop=True)
            results['Position'] = range(1, len(results) + 1)
            results['Confidence'] = np.linspace(85, 50, len(results))

            # Ensure required columns
            for col in ['Driver', 'Team']:
                if col not in results.columns:
                    results[col] = [f"{col} {i+1}" for i in range(len(results))]

            return results[['Position', 'Driver', 'Team', 'QualifyingTime',
                          'PredictedRaceTime', 'ReadableRaceTime', 'Confidence']]

        except Exception:
            # Absolute fallback
            return pd.DataFrame(columns=['Position', 'Driver', 'Team', 'QualifyingTime',
                                       'PredictedRaceTime', 'ReadableRaceTime', 'Confidence'])


