"""
Music Recommendation Engine

This module provides the main recommendation engine that combines
mood prediction and genre classification to provide personalized
music recommendations for mental wellbeing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import joblib
import logging
from .modeling import MoodPredictionModel, GenreClassifier
from .utils import create_user_profile, calculate_mood_improvement

class MusicRecommendationEngine:
    """
    Main recommendation engine that provides personalized music suggestions
    based on user's mental health indicators and preferences.
    """
    
    def __init__(self):
        self.mood_model = None
        self.genre_model = None
        self.music_database = None
        self.is_initialized = False
        
    def load_models(self, mood_model_path: str, genre_model_path: str):
        """
        Load pre-trained models.
        
        Args:
            mood_model_path: Path to mood prediction model
            genre_model_path: Path to genre classification model
        """
        try:
            self.mood_model = MoodPredictionModel()
            self.mood_model.load_model(mood_model_path)
            
            self.genre_model = GenreClassifier()
            self.genre_model.load_model(genre_model_path)
            
            self.is_initialized = True
            logging.info("Models loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def load_music_database(self, database_path: str):
        """
        Load music database with genre characteristics.
        
        Args:
            database_path: Path to music database CSV
        """
        try:
            self.music_database = pd.read_csv(database_path)
            logging.info(f"Music database loaded: {len(self.music_database)} entries")
        except Exception as e:
            logging.error(f"Error loading music database: {e}")
            raise
    
    def get_recommendations(self, user_profile: Dict[str, Any], 
                          num_recommendations: int = 5) -> Dict[str, Any]:
        """
        Generate personalized music recommendations.
        
        Args:
            user_profile: Dict containing user characteristics and preferences
            num_recommendations: Number of recommendations to return
            
        Returns:
            Dict with recommendations and explanations
        """
        if not self.is_initialized:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Convert user profile to feature vector
        user_features = self._profile_to_features(user_profile)
        
        # Predict mood improvement for different genres
        genre_recommendations = self._get_genre_recommendations(user_features, num_recommendations)
        
        # Get specific track recommendations
        track_recommendations = self._get_track_recommendations(genre_recommendations, user_profile)
        
        # Predict expected mood improvement
        expected_improvement = self._predict_mood_improvement(user_features, genre_recommendations)
        
        # Generate explanations
        explanations = self._generate_explanations(user_profile, genre_recommendations)
        
        return {
            'genre_recommendations': genre_recommendations,
            'track_recommendations': track_recommendations,
            'expected_mood_improvement': expected_improvement,
            'explanations': explanations,
            'user_profile': user_profile
        }
    
    def _profile_to_features(self, user_profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert user profile to model features.
        
        Args:
            user_profile: User characteristics
            
        Returns:
            DataFrame with features for model prediction
        """
        # Create feature vector based on user profile
        features = {
            'age': user_profile.get('age', 30),
            'anxiety_level': user_profile.get('anxiety_level', 5),
            'depression_level': user_profile.get('depression_level', 5),
            'insomnia_level': user_profile.get('insomnia_level', 5),
            'ocd_level': user_profile.get('ocd_level', 5),
            'mood_before': user_profile.get('current_mood', 3),
            'time_numeric': self._encode_time(user_profile.get('time_of_day', 'Evening')),
            'gender_encoded': self._encode_gender(user_profile.get('gender', 'Other')),
            'occupation_encoded': self._encode_occupation(user_profile.get('occupation', 'Other'))
        }
        
        # Add music preference features if available
        music_prefs = user_profile.get('music_preferences', {})
        features.update({
            'tempo': music_prefs.get('preferred_tempo', 100),
            'energy': music_prefs.get('preferred_energy', 0.5),
            'valence': music_prefs.get('preferred_valence', 0.5),
            'acousticness': music_prefs.get('preferred_acousticness', 0.5),
            'danceability': music_prefs.get('preferred_danceability', 0.5)
        })
        
        # Calculate composite features
        features['mental_health_score'] = np.mean([
            features['anxiety_level'],
            features['depression_level'], 
            features['insomnia_level'],
            features['ocd_level']
        ])
        
        features['energy_valence'] = features['energy'] * features['valence']
        
        return pd.DataFrame([features])
    
    def _get_genre_recommendations(self, user_features: pd.DataFrame, 
                                 num_recommendations: int) -> pd.DataFrame:
        """
        Get genre recommendations using the trained classifier.
        
        Args:
            user_features: User feature vector
            num_recommendations: Number of genres to recommend
            
        Returns:
            DataFrame with recommended genres and confidence scores
        """
        recommendations = self.genre_model.get_genre_recommendations(
            user_features, top_k=num_recommendations
        )
        
        return recommendations
    
    def _get_track_recommendations(self, genre_recommendations: pd.DataFrame,
                                 user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get specific track recommendations based on recommended genres.
        
        Args:
            genre_recommendations: Recommended genres with confidence scores
            user_profile: User preferences
            
        Returns:
            List of recommended tracks
        """
        if self.music_database is None:
            # Return placeholder recommendations
            tracks = []
            for _, row in genre_recommendations.iterrows():
                genre = row['genre']
                confidence = row['confidence']
                
                tracks.append({
                    'genre': genre,
                    'track_name': f"Sample {genre} Track",
                    'artist': f"{genre} Artist",
                    'confidence': confidence,
                    'expected_mood_improvement': confidence * 3.0  # Scale to 0-3 range
                })
            
            return tracks
        
        # Use actual music database if available
        tracks = []
        for _, row in genre_recommendations.iterrows():
            genre = row['genre']
            confidence = row['confidence']
            
            # Filter tracks by genre
            genre_tracks = self.music_database[self.music_database['genre'] == genre]
            
            if not genre_tracks.empty:
                # Select best matching track based on user preferences
                best_track = self._select_best_track(genre_tracks, user_profile)
                
                tracks.append({
                    'genre': genre,
                    'track_name': best_track.get('track_name', f"Sample {genre} Track"),
                    'artist': best_track.get('artist', f"{genre} Artist"),
                    'confidence': confidence,
                    'expected_mood_improvement': confidence * 3.0
                })
        
        return tracks
    
    def _predict_mood_improvement(self, user_features: pd.DataFrame,
                                genre_recommendations: pd.DataFrame) -> Dict[str, float]:
        """
        Predict expected mood improvement for recommendations.
        
        Args:
            user_features: User feature vector
            genre_recommendations: Recommended genres
            
        Returns:
            Dict with mood improvement predictions
        """
        predictions = {}
        
        for _, row in genre_recommendations.iterrows():
            genre = row['genre']
            
            # Create features for this genre recommendation
            genre_features = user_features.copy()
            genre_features['genre_encoded'] = self._encode_genre(genre)
            
            # Predict mood improvement
            mood_improvement = self.mood_model.predict(genre_features)[0]
            predictions[genre] = max(0, min(5, mood_improvement))  # Clamp to 0-5 range
        
        # Overall expected improvement (weighted by confidence)
        total_confidence = genre_recommendations['confidence'].sum()
        if total_confidence > 0:
            weighted_improvement = sum(
                predictions[row['genre']] * row['confidence']
                for _, row in genre_recommendations.iterrows()
            ) / total_confidence
        else:
            weighted_improvement = 0
        
        predictions['overall'] = weighted_improvement
        
        return predictions
    
    def _generate_explanations(self, user_profile: Dict[str, Any],
                             genre_recommendations: pd.DataFrame) -> Dict[str, str]:
        """
        Generate explanations for recommendations.
        
        Args:
            user_profile: User characteristics
            genre_recommendations: Recommended genres
            
        Returns:
            Dict with explanations for each recommendation
        """
        explanations = {}
        
        # Get primary mental health concerns
        concerns = []
        if user_profile.get('anxiety_level', 0) > 6:
            concerns.append('anxiety')
        if user_profile.get('depression_level', 0) > 6:
            concerns.append('depression')
        if user_profile.get('insomnia_level', 0) > 6:
            concerns.append('insomnia')
        if user_profile.get('ocd_level', 0) > 6:
            concerns.append('OCD')
        
        # Genre-specific explanations
        genre_explanations = {
            'Classical': "Classical music's structured harmonies and moderate tempo (60-90 BPM) have been shown to reduce cortisol levels and promote relaxation.",
            'Ambient': "Ambient music with minimal rhythmic elements helps quiet mental chatter and promotes a meditative state, ideal for anxiety relief.",
            'Lo-fi Hip Hop': "The repetitive, gentle beats of lo-fi hip hop create a comfortable background that can improve focus and reduce stress.",
            'Jazz': "Jazz music's complex but soothing harmonies can provide emotional regulation while maintaining gentle stimulation.",
            'New Age': "New Age music specifically designed for wellness combines natural sounds with calming melodies for therapeutic effect.",
            'Folk': "Acoustic folk music's organic, unplugged nature promotes groundedness and emotional connection.",
            'Meditation Music': "Specifically designed to synchronize with brainwave patterns associated with relaxation and reduced anxiety."
        }
        
        for _, row in genre_recommendations.iterrows():
            genre = row['genre']
            confidence = row['confidence']
            
            base_explanation = genre_explanations.get(genre, 
                f"{genre} music has shown positive effects for mood regulation.")
            
            # Add personalized elements
            if concerns:
                concern_text = ', '.join(concerns)
                explanation = f"For {concern_text} relief: {base_explanation} "
            else:
                explanation = base_explanation + " "
            
            # Add confidence and expected outcome
            explanation += f"Based on your profile, this has a {confidence*100:.0f}% "
            explanation += f"confidence rating for mood improvement."
            
            explanations[genre] = explanation
        
        return explanations
    
    def _select_best_track(self, genre_tracks: pd.DataFrame, 
                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best track from a genre based on user preferences.
        
        Args:
            genre_tracks: Available tracks in the genre
            user_profile: User preferences
            
        Returns:
            Best matching track
        """
        # Simple selection - in practice, this would use more sophisticated matching
        if len(genre_tracks) > 0:
            return genre_tracks.iloc[0].to_dict()
        else:
            return {}
    
    def _encode_time(self, time_of_day: str) -> int:
        """Encode time of day to numeric value."""
        time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
        return time_mapping.get(time_of_day, 2)
    
    def _encode_gender(self, gender: str) -> int:
        """Encode gender to numeric value."""
        gender_mapping = {'Male': 0, 'Female': 1, 'Non-binary': 2, 'Other': 2}
        return gender_mapping.get(gender, 2)
    
    def _encode_occupation(self, occupation: str) -> int:
        """Encode occupation to numeric value."""
        occupation_mapping = {
            'Student': 0, 'Engineer': 1, 'Designer': 2, 'Manager': 3,
            'Teacher': 4, 'Artist': 5, 'Developer': 6, 'Nurse': 7,
            'Doctor': 8, 'Other': 9
        }
        return occupation_mapping.get(occupation, 9)
    
    def _encode_genre(self, genre: str) -> int:
        """Encode genre to numeric value."""
        # This would typically use the same encoder as during training
        # For now, return a placeholder
        return hash(genre) % 100
    
    def batch_recommendations(self, user_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_profiles: List of user profiles
            
        Returns:
            List of recommendation results
        """
        results = []
        for profile in user_profiles:
            try:
                recommendations = self.get_recommendations(profile)
                results.append(recommendations)
            except Exception as e:
                logging.error(f"Error generating recommendations for user: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def update_user_feedback(self, user_id: str, recommendations: Dict[str, Any],
                           feedback: Dict[str, Any]):
        """
        Update models based on user feedback (for future implementation).
        
        Args:
            user_id: User identifier
            recommendations: Previously given recommendations
            feedback: User feedback on recommendations
        """
        # Placeholder for feedback learning mechanism
        # In production, this would update model weights or retrain
        logging.info(f"Received feedback from user {user_id}")
        pass

def create_recommendation_engine(mood_model_path: str = None, 
                               genre_model_path: str = None) -> MusicRecommendationEngine:
    """
    Factory function to create and configure a recommendation engine.
    
    Args:
        mood_model_path: Path to mood prediction model
        genre_model_path: Path to genre classification model
        
    Returns:
        Configured recommendation engine
    """
    engine = MusicRecommendationEngine()
    
    if mood_model_path and genre_model_path:
        engine.load_models(mood_model_path, genre_model_path)
    
    return engine

if __name__ == "__main__":
    # Example usage
    engine = MusicRecommendationEngine()
    
    # Example user profile
    user_profile = {
        'age': 28,
        'gender': 'Female',
        'occupation': 'Designer',
        'anxiety_level': 7,
        'depression_level': 5,
        'insomnia_level': 6,
        'ocd_level': 4,
        'current_mood': 3,
        'time_of_day': 'Evening',
        'music_preferences': {
            'preferred_tempo': 80,
            'preferred_energy': 0.3,
            'preferred_valence': 0.4
        }
    }
    
    print("Example user profile created")
    print("Note: Load trained models before generating recommendations")