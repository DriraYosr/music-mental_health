"""
Unit Tests for Music Wellbeing Recommendation System

This module contains comprehensive tests for the recommendation engine,
models, data preprocessing, and utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from recommend import MusicRecommendationEngine, create_recommendation_engine
from modeling import MoodPredictionModel, GenreClassifier, ModelEnsemble
from data_preprocessing import DataPreprocessor, preprocess_pipeline
from utils import (
    create_user_profile, validate_user_input, calculate_mood_improvement,
    get_primary_mental_health_concern, calculate_recommendation_confidence,
    convert_numpy_types
)

class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = DataPreprocessor()
        self.sample_data = pd.DataFrame({
            'user_id': ['U001', 'U002', 'U003'],
            'age': [25, 30, 35],
            'gender': ['Female', 'Male', 'Female'],
            'occupation': ['Student', 'Engineer', 'Designer'],
            'anxiety_level': [7, 4, 6],
            'depression_level': [5, 3, 5],
            'insomnia_level': [6, 2, 7],
            'ocd_level': [4, 1, 5],
            'genre': ['Electronic', 'Classical', 'Jazz'],
            'tempo': [128, 72, 95],
            'energy': [0.8, 0.3, 0.5],
            'valence': [0.6, 0.4, 0.7],
            'acousticness': [0.2, 0.9, 0.6],
            'danceability': [0.7, 0.1, 0.4],
            'time_of_day': ['Evening', 'Night', 'Afternoon'],
            'mood_before': [3, 2, 4],
            'mood_after': [6, 5, 6],
            'rating': [4, 5, 4],
            'mood_improvement': [3, 3, 2]
        })
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        
        cleaned_data = self.processor.clean_data(data_with_missing)
        
        # Should remove rows with missing values
        assert len(cleaned_data) == 2
        assert cleaned_data.isnull().sum().sum() == 0
    
    def test_create_features(self):
        """Test feature engineering."""
        features_data = self.processor.create_features(self.sample_data)
        
        # Check if new features are created
        assert 'mental_health_score' in features_data.columns
        assert 'energy_valence' in features_data.columns
        assert 'tempo_category' in features_data.columns
        assert 'time_numeric' in features_data.columns
        assert 'mood_change' in features_data.columns
        
        # Verify calculations
        expected_mental_health = (7 + 5 + 6 + 4) / 4  # First row
        assert abs(features_data.loc[0, 'mental_health_score'] - expected_mental_health) < 0.01
        
        expected_energy_valence = 0.8 * 0.6  # First row
        assert abs(features_data.loc[0, 'energy_valence'] - expected_energy_valence) < 0.01
    
    def test_encode_categorical_features(self):
        """Test categorical encoding."""
        features_data = self.processor.create_features(self.sample_data)
        encoded_data = self.processor.encode_categorical_features(features_data)
        
        # Check if encoded columns are created
        assert 'gender_encoded' in encoded_data.columns
        assert 'occupation_encoded' in encoded_data.columns
        assert 'genre_encoded' in encoded_data.columns
        
        # Verify encoding is numeric
        assert encoded_data['gender_encoded'].dtype in [np.int64, np.int32]
    
    def test_prepare_features_for_modeling(self):
        """Test feature preparation for modeling."""
        features_data = self.processor.create_features(self.sample_data)
        encoded_data = self.processor.encode_categorical_features(features_data)
        
        X, y_mood, y_rating, y_genre = self.processor.prepare_features_for_modeling(encoded_data)
        
        # Check dimensions
        assert X.shape[0] == len(self.sample_data)
        assert len(y_mood) == len(self.sample_data)
        assert len(y_rating) == len(self.sample_data)
        assert len(y_genre) == len(self.sample_data)
        
        # Check feature columns are stored
        assert len(self.processor.feature_columns) > 0

class TestMoodPredictionModel:
    """Test mood prediction model functionality."""
    
    def setup_method(self):
        """Set up test data and model."""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(50, 10))
        self.y_train = pd.Series(np.random.rand(50) * 5)  # Mood improvement 0-5
        self.X_test = pd.DataFrame(np.random.rand(20, 10))
        self.y_test = pd.Series(np.random.rand(20) * 5)
        
        self.model = MoodPredictionModel('random_forest')
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Test different model types
        rf_model = MoodPredictionModel('random_forest')
        xgb_model = MoodPredictionModel('xgboost')
        linear_model = MoodPredictionModel('linear')
        
        assert rf_model.model_type == 'random_forest'
        assert xgb_model.model_type == 'xgboost'
        assert linear_model.model_type == 'linear'
        
        # Test invalid model type
        with pytest.raises(ValueError):
            MoodPredictionModel('invalid_model')
    
    def test_model_training(self):
        """Test model training."""
        metrics = self.model.train(self.X_train, self.y_train)
        
        assert self.model.is_fitted
        assert 'mse_train' in metrics
        assert 'rmse_train' in metrics
        assert 'r2_train' in metrics
        assert metrics['mse_train'] >= 0
        assert metrics['rmse_train'] >= 0
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Should raise error if not trained
        with pytest.raises(ValueError):
            self.model.predict(self.X_test)
        
        # Train model first
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert all(pred >= 0 for pred in predictions)  # Non-negative predictions
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        assert 'mse_test' in metrics
        assert 'rmse_test' in metrics
        assert 'r2_test' in metrics
        assert 'mae_test' in metrics
        assert all(metric >= 0 for key, metric in metrics.items() if 'r2' not in key)

class TestGenreClassifier:
    """Test genre classification functionality."""
    
    def setup_method(self):
        """Set up test data and model."""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(50, 10))
        self.y_train = pd.Series(np.random.choice(['Classical', 'Jazz', 'Electronic'], 50))
        self.X_test = pd.DataFrame(np.random.rand(20, 10))
        self.y_test = pd.Series(np.random.choice(['Classical', 'Jazz', 'Electronic'], 20))
        
        self.classifier = GenreClassifier('random_forest')
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        rf_classifier = GenreClassifier('random_forest')
        svm_classifier = GenreClassifier('svm')
        lr_classifier = GenreClassifier('logistic')
        
        assert rf_classifier.model_type == 'random_forest'
        assert svm_classifier.model_type == 'svm'
        assert lr_classifier.model_type == 'logistic'
        
        # Test invalid model type
        with pytest.raises(ValueError):
            GenreClassifier('invalid_model')
    
    def test_classifier_training(self):
        """Test classifier training."""
        metrics = self.classifier.train(self.X_train, self.y_train)
        
        assert self.classifier.is_fitted
        assert 'accuracy_train' in metrics
        assert 'n_classes' in metrics
        assert 0 <= metrics['accuracy_train'] <= 1
        assert metrics['n_classes'] == 3  # Classical, Jazz, Electronic
    
    def test_genre_recommendations(self):
        """Test genre recommendation functionality."""
        self.classifier.train(self.X_train, self.y_train)
        recommendations = self.classifier.get_genre_recommendations(self.X_test, top_k=3)
        
        assert len(recommendations) == len(self.X_test) * 3  # top_k per user
        assert 'genre' in recommendations.columns
        assert 'confidence' in recommendations.columns
        assert 'rank' in recommendations.columns
        assert all(1 <= rank <= 3 for rank in recommendations['rank'])
        assert all(0 <= conf <= 1 for conf in recommendations['confidence'])

class TestUtils:
    """Test utility functions."""
    
    def test_create_user_profile(self):
        """Test user profile creation."""
        profile = create_user_profile(
            age=28, gender='Female', occupation='Designer',
            anxiety_level=7, depression_level=5, insomnia_level=6, ocd_level=4
        )
        
        assert profile['age'] == 28
        assert profile['gender'] == 'Female'
        assert profile['occupation'] == 'Designer'
        assert profile['anxiety_level'] == 7
        assert 'mental_health_score' in profile
        assert 'primary_concern' in profile
        assert 'created_at' in profile
        
        # Test calculated fields
        expected_mh_score = (7 + 5 + 6 + 4) / 4
        assert abs(profile['mental_health_score'] - expected_mh_score) < 0.01
    
    def test_validate_user_input(self):
        """Test input validation."""
        # Valid input
        valid_input = {
            'age': 25,
            'anxiety_level': 5,
            'depression_level': 4,
            'insomnia_level': 6,
            'ocd_level': 3
        }
        
        is_valid, errors = validate_user_input(valid_input)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid input - missing fields
        invalid_input = {'age': 25}
        is_valid, errors = validate_user_input(invalid_input)
        assert not is_valid
        assert len(errors) > 0
        
        # Invalid input - out of range
        invalid_input = {
            'age': 150,  # Too high
            'anxiety_level': 15,  # Too high
            'depression_level': 4,
            'insomnia_level': 6,
            'ocd_level': 3
        }
        is_valid, errors = validate_user_input(invalid_input)
        assert not is_valid
        assert any('Age must be between' in error for error in errors)
    
    def test_get_primary_mental_health_concern(self):
        """Test primary concern identification."""
        # High anxiety
        concern = get_primary_mental_health_concern(8, 3, 4, 2)
        assert concern == 'anxiety'
        
        # Mild depression
        concern = get_primary_mental_health_concern(3, 5, 3, 2)
        assert concern == 'mild_depression'
        
        # General wellbeing (all low)
        concern = get_primary_mental_health_concern(2, 3, 2, 1)
        assert concern == 'general_wellbeing'
    
    def test_calculate_mood_improvement(self):
        """Test mood improvement calculation."""
        # Positive improvement
        improvement = calculate_mood_improvement(3, 6)
        assert improvement > 2.5  # Above neutral
        
        # No change
        improvement = calculate_mood_improvement(4, 4)
        assert improvement == 2.5  # Neutral
        
        # Negative change (shouldn't go below 0)
        improvement = calculate_mood_improvement(6, 2)
        assert improvement >= 0
    
    def test_convert_numpy_types(self):
        """Test numpy type conversion."""
        test_data = {
            'int_val': np.int64(42),
            'float_val': np.float64(3.14),
            'array_val': np.array([1, 2, 3]),
            'nested': {
                'np_val': np.int32(10)
            }
        }
        
        converted = convert_numpy_types(test_data)
        
        assert isinstance(converted['int_val'], int)
        assert isinstance(converted['float_val'], float)
        assert isinstance(converted['array_val'], list)
        assert isinstance(converted['nested']['np_val'], int)
    
    def test_calculate_recommendation_confidence(self):
        """Test recommendation confidence calculation."""
        user_profile = {
            'primary_concern': 'anxiety',
            'music_preferences': {
                'preferred_tempo': 70,
                'preferred_energy': 0.3
            }
        }
        
        # Should have high confidence for ambient music (good for anxiety)
        confidence_ambient = calculate_recommendation_confidence(user_profile, 'Ambient')
        confidence_jazz = calculate_recommendation_confidence(user_profile, 'Jazz')
        
        assert 0 <= confidence_ambient <= 1
        assert 0 <= confidence_jazz <= 1
        # Ambient should be more suitable for anxiety than Jazz
        assert confidence_ambient >= confidence_jazz

class TestMusicRecommendationEngine:
    """Test the main recommendation engine."""
    
    def setup_method(self):
        """Set up test engine and data."""
        self.engine = MusicRecommendationEngine()
        self.user_profile = create_user_profile(
            age=28, gender='Female', occupation='Designer',
            anxiety_level=7, depression_level=5, insomnia_level=6, ocd_level=4,
            current_mood=3, time_of_day='Evening'
        )
    
    def test_profile_to_features(self):
        """Test user profile to features conversion."""
        features = self.engine._profile_to_features(self.user_profile)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1  # Single user
        assert 'age' in features.columns
        assert 'anxiety_level' in features.columns
        assert 'mental_health_score' in features.columns
        
        # Check feature values
        assert features.loc[0, 'age'] == 28
        assert features.loc[0, 'anxiety_level'] == 7
    
    def test_encoding_functions(self):
        """Test encoding helper functions."""
        # Time encoding
        assert self.engine._encode_time('Morning') == 0
        assert self.engine._encode_time('Evening') == 2
        assert self.engine._encode_time('Invalid') == 2  # Default
        
        # Gender encoding
        assert self.engine._encode_gender('Male') == 0
        assert self.engine._encode_gender('Female') == 1
        assert self.engine._encode_gender('Other') == 2
        
        # Occupation encoding
        assert self.engine._encode_occupation('Student') == 0
        assert self.engine._encode_occupation('Engineer') == 1
        assert self.engine._encode_occupation('Unknown') == 9  # Default

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to recommendations."""
        # Create sample data
        sample_data = pd.DataFrame({
            'user_id': ['U001', 'U002'],
            'age': [25, 30],
            'gender': ['Female', 'Male'],
            'occupation': ['Student', 'Engineer'],
            'anxiety_level': [7, 4],
            'depression_level': [5, 3],
            'insomnia_level': [6, 2],
            'ocd_level': [4, 1],
            'genre': ['Electronic', 'Classical'],
            'tempo': [128, 72],
            'energy': [0.8, 0.3],
            'valence': [0.6, 0.4],
            'acousticness': [0.2, 0.9],
            'danceability': [0.7, 0.1],
            'time_of_day': ['Evening', 'Night'],
            'mood_before': [3, 2],
            'mood_after': [6, 5],
            'rating': [4, 5],
            'mood_improvement': [3, 3]
        })
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test preprocessing pipeline
            processor = DataPreprocessor()
            df = processor.load_data(temp_file)
            df_clean = processor.clean_data(df)
            df_features = processor.create_features(df_clean)
            df_encoded = processor.encode_categorical_features(df_features)
            
            X, y_mood, y_rating, y_genre = processor.prepare_features_for_modeling(df_encoded)
            
            # Verify dimensions
            assert X.shape[0] == 2
            assert len(y_mood) == 2
            assert len(y_genre) == 2
            
            # Test model training (with small dataset, just verify it runs)
            mood_model = MoodPredictionModel('linear')  # Simpler model for small data
            mood_model.train(X, y_mood)
            
            assert mood_model.is_fitted
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_demo_recommendation_system(self):
        """Test the demo recommendation system."""
        # Create demo engine (since we don't have trained models)
        from app.streamlit_app import create_demo_recommendation_engine
        
        demo_engine = create_demo_recommendation_engine()
        
        user_profile = create_user_profile(
            age=28, gender='Female', occupation='Designer',
            anxiety_level=8, depression_level=4, insomnia_level=6, ocd_level=3
        )
        
        recommendations = demo_engine.get_recommendations(user_profile)
        
        # Verify recommendation structure
        assert 'genre_recommendations' in recommendations
        assert 'track_recommendations' in recommendations
        assert 'expected_mood_improvement' in recommendations
        assert 'explanations' in recommendations
        
        # Verify genre recommendations
        genre_recs = recommendations['genre_recommendations']
        assert len(genre_recs) > 0
        assert 'genre' in genre_recs.columns
        assert 'confidence' in genre_recs.columns
        
        # Verify track recommendations
        track_recs = recommendations['track_recommendations']
        assert len(track_recs) > 0
        assert all('genre' in track for track in track_recs)
        
        # For high anxiety, should recommend calming genres
        recommended_genres = [row['genre'] for _, row in genre_recs.iterrows()]
        assert any(genre in ['Ambient', 'Classical', 'New Age'] for genre in recommended_genres)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])