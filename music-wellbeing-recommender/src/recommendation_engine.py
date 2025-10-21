
from typing import Dict, Union, List, Tuple
import pandas as pd
import numpy as np
import joblib

def recommend_music_genres(
    user_profile: Dict[str, Union[int, float, bool]], 
    target_to_improve: str,
    model_path: str = '../models/music_effect_model.pkl',
    encoder_path: str = '../models/label_encoder.pkl',
    features_path: str = '../models/feature_columns.pkl',
    top_n: int = 5
) -> pd.DataFrame:
    """
    Recommend music genres most likely to improve a user's mental wellbeing.

    Args:
        user_profile: User characteristics and mental health scores
        target_to_improve: Mental health aspect to focus on
        model_path: Path to trained model file
        encoder_path: Path to label encoder file  
        features_path: Path to feature columns file
        top_n: Number of recommendations to return

    Returns:
        DataFrame with recommended genres and probabilities
    """

    # Load model artifacts
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_columns = joblib.load(features_path)

    # Extract available genres
    genre_features = [col for col in feature_columns if col.startswith('Fav genre_')]
    available_genres = [genre.replace('Fav genre_', '') for genre in genre_features]

    if not available_genres:
        return pd.DataFrame()

    # Create base profile
    base_profile = pd.Series(user_profile)
    recommendations = []

    # Test each genre
    for genre in available_genres:
        test_profile = base_profile.copy()

        # Reset all genre flags
        for genre_feature in genre_features:
            test_profile[genre_feature] = False

        # Set current genre
        test_profile[f'Fav genre_{genre}'] = True

        # Reindex and prepare for prediction
        test_profile = test_profile.reindex(feature_columns, fill_value=False)

        # Fixed type conversion - handle boolean and NaN values properly
        for col in test_profile.index:
            if isinstance(test_profile[col], (bool, np.bool_)):
                test_profile[col] = int(test_profile[col])
            elif pd.isna(test_profile[col]):
                test_profile[col] = 0

        # Get prediction
        test_array = test_profile.values.reshape(1, -1)
        probabilities = model.predict_proba(test_array)[0]

        # Find "Improve" probability
        improve_idx = np.where(label_encoder.classes_ == 'Improve')[0]
        if len(improve_idx) > 0:
            improve_prob = probabilities[improve_idx[0]]
            recommendations.append({
                'Genre': genre,
                'Improve_Probability': improve_prob,
                'Predicted_Effect': label_encoder.classes_[np.argmax(probabilities)]
            })

    # Convert to DataFrame and sort
    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        return pd.DataFrame()

    recommendations_df = recommendations_df.sort_values('Improve_Probability', ascending=False)
    return recommendations_df.head(top_n)
