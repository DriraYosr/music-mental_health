"""
Data Preprocessing Module for Music Wellbeing Recommender

This module handles data cleaning, feature engineering, and preprocessing
for the music recommendation system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

class DataPreprocessor:
    """
    Handles data preprocessing for the music wellbeing recommendation system.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_columns = []
        
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna()
        
        # Remove outliers using IQR method for numerical columns
        numerical_cols = ['age', 'anxiety_level', 'depression_level', 'insomnia_level', 'ocd_level',
                         'tempo', 'energy', 'valence', 'acousticness', 'danceability']
        
        for col in numerical_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        logging.info(f"Data cleaned: {df_clean.shape}")
        return df_clean
    
    def create_features(self, df):
        """
        Create engineered features for better model performance.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        df_features = df.copy()
        
        # Mental health composite score
        mental_health_cols = ['anxiety_level', 'depression_level', 'insomnia_level', 'ocd_level']
        df_features['mental_health_score'] = df_features[mental_health_cols].mean(axis=1)
        
        # Music feature combinations
        df_features['energy_valence'] = df_features['energy'] * df_features['valence']
        df_features['tempo_category'] = pd.cut(df_features['tempo'], 
                                             bins=[0, 60, 100, 140, 200], 
                                             labels=['Very Slow', 'Slow', 'Medium', 'Fast'])
        
        # Time-based features
        time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
        df_features['time_numeric'] = df_features['time_of_day'].map(time_mapping)
        
        # User characteristics
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 25, 35, 50, 100], 
                                        labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        # Mood change calculation
        df_features['mood_change'] = df_features['mood_after'] - df_features['mood_before']
        
        logging.info("Feature engineering completed")
        return df_features
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features for machine learning models.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = ['gender', 'occupation', 'genre', 'time_of_day', 'tempo_category', 'age_group']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features_for_modeling(self, df):
        """
        Prepare final feature set for machine learning models.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (X_features, y_targets)
        """
        # Select features for modeling
        feature_cols = [
            'age', 'anxiety_level', 'depression_level', 'insomnia_level', 'ocd_level',
            'tempo', 'energy', 'valence', 'acousticness', 'danceability',
            'mental_health_score', 'energy_valence', 'time_numeric', 'mood_before',
            'gender_encoded', 'occupation_encoded', 'genre_encoded', 'time_of_day_encoded'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols]
        
        # Target variables
        y_mood_improvement = df['mood_improvement']
        y_rating = df['rating']
        y_genre = df['genre']
        
        self.feature_columns = available_cols
        
        return X, y_mood_improvement, y_rating, y_genre
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Scaled training and test features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_pipeline(data_path, target_type='mood_improvement'):
    """
    Complete preprocessing pipeline.
    
    Args:
        data_path (str): Path to the data file
        target_type (str): Type of target variable ('mood_improvement', 'rating', 'genre')
        
    Returns:
        tuple: Preprocessed data ready for modeling
    """
    processor = DataPreprocessor()
    
    # Load and clean data
    df = processor.load_data(data_path)
    df_clean = processor.clean_data(df)
    
    # Feature engineering
    df_features = processor.create_features(df_clean)
    df_encoded = processor.encode_categorical_features(df_features)
    
    # Prepare for modeling
    X, y_mood, y_rating, y_genre = processor.prepare_features_for_modeling(df_encoded)
    
    # Select target based on target_type
    target_map = {
        'mood_improvement': y_mood,
        'rating': y_rating,
        'genre': y_genre
    }
    y = target_map[target_type]
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Scale features (only for numerical targets)
    if target_type in ['mood_improvement', 'rating']:
        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, processor
    
    return X_train, X_test, y_train, y_test, processor

if __name__ == "__main__":
    # Example usage
    data_path = "../data/raw/music_listening_data.csv"
    X_train, X_test, y_train, y_test, processor = preprocess_pipeline(data_path)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")