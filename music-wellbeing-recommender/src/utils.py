"""
Utility Functions for Music Wellbeing Recommender

This module contains helper functions and utilities used across
the music recommendation system.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Tuple
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_models(models_dir: str = "../models/") -> Dict[str, Any]:
    """
    Load all trained models from directory.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Dict with loaded models
    """
    models = {}
    
    if not os.path.exists(models_dir):
        logging.warning(f"Models directory {models_dir} not found")
        return models
    
    try:
        # Load mood prediction models
        mood_models = ['mood_random_forest.pkl', 'mood_xgboost.pkl', 'mood_linear.pkl']
        for model_file in mood_models:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                models[model_name] = joblib.load(model_path)
                logging.info(f"Loaded {model_name}")
        
        # Load genre classification models
        genre_models = ['genre_random_forest.pkl', 'genre_svm.pkl', 'genre_logistic.pkl']
        for model_file in genre_models:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                models[model_name] = joblib.load(model_path)
                logging.info(f"Loaded {model_name}")
        
        logging.info(f"Successfully loaded {len(models)} models")
        
    except Exception as e:
        logging.error(f"Error loading models: {e}")
    
    return models

def create_user_profile(age: int, gender: str, occupation: str,
                       anxiety_level: int, depression_level: int,
                       insomnia_level: int, ocd_level: int,
                       current_mood: int = 3, time_of_day: str = "Evening",
                       music_preferences: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Create a standardized user profile dictionary.
    
    Args:
        age: User's age
        gender: User's gender
        occupation: User's occupation
        anxiety_level: Anxiety level (1-10)
        depression_level: Depression level (1-10)
        insomnia_level: Insomnia level (1-10)
        ocd_level: OCD level (1-10)
        current_mood: Current mood (1-7)
        time_of_day: Time of listening
        music_preferences: Dict with music feature preferences
        
    Returns:
        Standardized user profile dictionary
    """
    if music_preferences is None:
        music_preferences = {
            'preferred_tempo': 100,
            'preferred_energy': 0.5,
            'preferred_valence': 0.5,
            'preferred_acousticness': 0.5,
            'preferred_danceability': 0.5
        }
    
    profile = {
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'anxiety_level': anxiety_level,
        'depression_level': depression_level,
        'insomnia_level': insomnia_level,
        'ocd_level': ocd_level,
        'current_mood': current_mood,
        'time_of_day': time_of_day,
        'music_preferences': music_preferences,
        'created_at': datetime.now().isoformat()
    }
    
    # Add computed fields
    profile['mental_health_score'] = np.mean([
        anxiety_level, depression_level, insomnia_level, ocd_level
    ])
    
    profile['primary_concern'] = get_primary_mental_health_concern(
        anxiety_level, depression_level, insomnia_level, ocd_level
    )
    
    return profile

def get_primary_mental_health_concern(anxiety_level: int, depression_level: int,
                                    insomnia_level: int, ocd_level: int) -> str:
    """
    Identify the primary mental health concern based on levels.
    
    Args:
        anxiety_level: Anxiety level (1-10)
        depression_level: Depression level (1-10)
        insomnia_level: Insomnia level (1-10)
        ocd_level: OCD level (1-10)
        
    Returns:
        Primary concern as string
    """
    levels = {
        'anxiety': anxiety_level,
        'depression': depression_level,
        'insomnia': insomnia_level,
        'ocd': ocd_level
    }
    
    max_concern = max(levels, key=levels.get)
    max_level = levels[max_concern]
    
    if max_level >= 7:
        return max_concern
    elif max_level >= 4:
        return f"mild_{max_concern}"
    else:
        return "general_wellbeing"

def calculate_mood_improvement(mood_before: int, mood_after: int) -> float:
    """
    Calculate mood improvement score.
    
    Args:
        mood_before: Mood before listening (1-7)
        mood_after: Mood after listening (1-7)
        
    Returns:
        Mood improvement score
    """
    improvement = mood_after - mood_before
    
    # Normalize to 0-5 scale
    normalized_improvement = max(0, min(5, improvement + 2.5))
    
    return normalized_improvement

def validate_user_input(user_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate user input for the recommendation system.
    
    Args:
        user_input: User input dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ['age', 'anxiety_level', 'depression_level', 
                      'insomnia_level', 'ocd_level']
    
    for field in required_fields:
        if field not in user_input:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(user_input[field], (int, float)):
            errors.append(f"Field {field} must be numeric")
    
    # Validate ranges
    if 'age' in user_input:
        if not (13 <= user_input['age'] <= 100):
            errors.append("Age must be between 13 and 100")
    
    mental_health_fields = ['anxiety_level', 'depression_level', 
                           'insomnia_level', 'ocd_level']
    
    for field in mental_health_fields:
        if field in user_input:
            if not (1 <= user_input[field] <= 10):
                errors.append(f"{field} must be between 1 and 10")
    
    if 'current_mood' in user_input:
        if not (1 <= user_input['current_mood'] <= 7):
            errors.append("Current mood must be between 1 and 7")
    
    # Validate time_of_day
    valid_times = ['Morning', 'Afternoon', 'Evening', 'Night']
    if 'time_of_day' in user_input:
        if user_input['time_of_day'] not in valid_times:
            errors.append(f"time_of_day must be one of: {valid_times}")
    
    return len(errors) == 0, errors

def format_recommendations_for_display(recommendations: Dict[str, Any]) -> str:
    """
    Format recommendations for user-friendly display.
    
    Args:
        recommendations: Recommendations dictionary from engine
        
    Returns:
        Formatted string for display
    """
    output = []
    
    output.append("🎵 PERSONALIZED MUSIC RECOMMENDATIONS")
    output.append("=" * 50)
    
    # Genre recommendations
    if 'genre_recommendations' in recommendations:
        output.append("\n🎼 Recommended Genres:")
        for _, row in recommendations['genre_recommendations'].iterrows():
            genre = row['genre']
            confidence = row['confidence']
            output.append(f"  {row['rank']}. {genre} (Confidence: {confidence:.1%})")
    
    # Track recommendations
    if 'track_recommendations' in recommendations:
        output.append("\n🎵 Suggested Tracks:")
        for i, track in enumerate(recommendations['track_recommendations'], 1):
            output.append(f"  {i}. {track['track_name']} - {track['artist']}")
            output.append(f"     Genre: {track['genre']} | Expected improvement: +{track['expected_mood_improvement']:.1f}")
    
    # Expected mood improvement
    if 'expected_mood_improvement' in recommendations:
        overall = recommendations['expected_mood_improvement'].get('overall', 0)
        output.append(f"\n📈 Expected Overall Mood Improvement: +{overall:.1f} points")
    
    # Explanations
    if 'explanations' in recommendations:
        output.append("\n🧠 Why These Recommendations:")
        for genre, explanation in recommendations['explanations'].items():
            output.append(f"\n• {genre}:")
            output.append(f"  {explanation}")
    
    return "\n".join(output)

def save_user_session(user_profile: Dict[str, Any], recommendations: Dict[str, Any],
                     filepath: str = None) -> str:
    """
    Save user session data for analysis and improvement.
    
    Args:
        user_profile: User profile data
        recommendations: Generated recommendations
        filepath: Optional custom filepath
        
    Returns:
        Path where session was saved
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"../data/sessions/session_{timestamp}.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'user_profile': user_profile,
        'recommendations': recommendations
    }
    
    # Convert numpy types to native Python types for JSON serialization
    session_data = convert_numpy_types(session_data)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logging.info(f"Session saved to {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Error saving session: {e}")
        raise

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with converted types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_genre_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Load metadata about music genres including therapeutic properties.
    
    Returns:
        Dict with genre metadata
    """
    genre_metadata = {
        'Classical': {
            'typical_tempo': (60, 90),
            'typical_energy': (0.1, 0.4),
            'therapeutic_properties': ['anxiety_relief', 'focus_enhancement'],
            'best_for_conditions': ['anxiety', 'ocd'],
            'description': 'Structured harmonies promote relaxation and mental clarity'
        },
        'Ambient': {
            'typical_tempo': (40, 80),
            'typical_energy': (0.1, 0.3),
            'therapeutic_properties': ['deep_relaxation', 'meditation'],
            'best_for_conditions': ['anxiety', 'insomnia'],
            'description': 'Minimal rhythmic elements create meditative atmosphere'
        },
        'Lo-fi Hip Hop': {
            'typical_tempo': (70, 90),
            'typical_energy': (0.3, 0.5),
            'therapeutic_properties': ['focus', 'comfort'],
            'best_for_conditions': ['mild_anxiety', 'depression'],
            'description': 'Repetitive gentle beats provide comforting background'
        },
        'Jazz': {
            'typical_tempo': (80, 120),
            'typical_energy': (0.4, 0.7),
            'therapeutic_properties': ['emotional_regulation', 'creativity'],
            'best_for_conditions': ['depression', 'general_wellbeing'],
            'description': 'Complex harmonies provide emotional depth and stimulation'
        },
        'New Age': {
            'typical_tempo': (50, 80),
            'typical_energy': (0.1, 0.4),
            'therapeutic_properties': ['healing', 'spiritual_connection'],
            'best_for_conditions': ['anxiety', 'depression', 'insomnia'],
            'description': 'Designed specifically for therapeutic and wellness purposes'
        }
    }
    
    return genre_metadata

def calculate_recommendation_confidence(user_profile: Dict[str, Any],
                                     genre: str) -> float:
    """
    Calculate confidence score for a genre recommendation.
    
    Args:
        user_profile: User characteristics
        genre: Music genre
        
    Returns:
        Confidence score (0-1)
    """
    genre_metadata = load_genre_metadata()
    
    if genre not in genre_metadata:
        return 0.5  # Default confidence for unknown genres
    
    genre_info = genre_metadata[genre]
    primary_concern = user_profile.get('primary_concern', 'general_wellbeing')
    
    # Base confidence from genre suitability
    if primary_concern in genre_info['best_for_conditions']:
        base_confidence = 0.8
    elif any(concern in primary_concern for concern in genre_info['best_for_conditions']):
        base_confidence = 0.6
    else:
        base_confidence = 0.4
    
    # Adjust based on user preferences
    music_prefs = user_profile.get('music_preferences', {})
    
    # Tempo matching
    preferred_tempo = music_prefs.get('preferred_tempo', 100)
    tempo_range = genre_info['typical_tempo']
    
    if tempo_range[0] <= preferred_tempo <= tempo_range[1]:
        tempo_bonus = 0.1
    else:
        tempo_bonus = -0.1
    
    # Energy matching
    preferred_energy = music_prefs.get('preferred_energy', 0.5)
    energy_range = genre_info['typical_energy']
    
    if energy_range[0] <= preferred_energy <= energy_range[1]:
        energy_bonus = 0.1
    else:
        energy_bonus = -0.1
    
    # Calculate final confidence
    final_confidence = base_confidence + tempo_bonus + energy_bonus
    
    # Clamp to valid range
    return max(0.0, min(1.0, final_confidence))

def get_success_stories() -> List[Dict[str, Any]]:
    """
    Get example success stories for demonstration purposes.
    
    Returns:
        List of success story dictionaries
    """
    stories = [
        {
            'user_type': 'Student with High Anxiety',
            'profile': {
                'age': 22,
                'anxiety_level': 8,
                'depression_level': 4,
                'primary_concern': 'anxiety'
            },
            'recommendation': 'Ambient Electronic',
            'outcome': 'Reduced anxiety from 8 to 4 within 20 minutes',
            'feedback': 'The soft, repetitive sounds helped quiet my racing thoughts before exams.'
        },
        {
            'user_type': 'Professional with Depression',
            'profile': {
                'age': 35,
                'depression_level': 7,
                'anxiety_level': 3,
                'primary_concern': 'depression'
            },
            'recommendation': 'Uplifting Jazz',
            'outcome': 'Mood improvement from 2 to 5 over 30-minute session',
            'feedback': 'The complex melodies gave me something positive to focus on.'
        },
        {
            'user_type': 'Shift Worker with Insomnia',
            'profile': {
                'age': 28,
                'insomnia_level': 9,
                'time_of_day': 'Night',
                'primary_concern': 'insomnia'
            },
            'recommendation': 'Sleep-Inducing Classical',
            'outcome': 'Fell asleep 40% faster than usual',
            'feedback': 'The slow tempo helped my body relax and prepare for sleep.'
        }
    ]
    
    return stories

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing utility functions...")
    
    # Test user profile creation
    profile = create_user_profile(
        age=28, gender='Female', occupation='Designer',
        anxiety_level=7, depression_level=5, insomnia_level=6, ocd_level=4
    )
    
    print(f"Created user profile: {profile['primary_concern']}")
    
    # Test validation
    is_valid, errors = validate_user_input(profile)
    print(f"Profile validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print("✅ Utility functions test completed")