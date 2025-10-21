"""
Music Wellbeing Recommender Package

A machine learning-based system for recommending music to improve mental wellbeing.
Provides personalized recommendations for anxiety, depression, insomnia, and OCD management.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .recommend import MusicRecommendationEngine
from .modeling import MoodPredictionModel, GenreClassifier
from .data_preprocessing import DataPreprocessor
from .utils import load_models, create_user_profile, calculate_mood_improvement

__all__ = [
    'MusicRecommendationEngine',
    'MoodPredictionModel', 
    'GenreClassifier',
    'DataPreprocessor',
    'load_models',
    'create_user_profile',
    'calculate_mood_improvement'
]