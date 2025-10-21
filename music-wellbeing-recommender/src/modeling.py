"""
Machine Learning Models for Music Wellbeing Recommendation

This module contains model classes for predicting mood improvement and 
classifying optimal music genres for mental health conditions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any

class MoodPredictionModel:
    """
    Predicts mood improvement based on user characteristics and music features.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the mood prediction model.
        
        Args:
            model_type (str): Type of model ('random_forest', 'xgboost', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the mood prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets (mood improvement scores)
            
        Returns:
            Dict with training metrics
        """
        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Calculate training metrics
            y_pred_train = self.model.predict(X_train)
            mse_train = mean_squared_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)
            
            metrics = {
                'mse_train': mse_train,
                'rmse_train': np.sqrt(mse_train),
                'r2_train': r2_train
            }
            
            logging.info(f"Model {self.model_type} trained successfully")
            return metrics
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted mood improvement scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse_test': mean_squared_error(y_test, y_pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_test': r2_score(y_test, y_pred),
            'mae_test': np.mean(np.abs(y_test - y_pred))
        }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        return {
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available for this model type")
        
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logging.info(f"Model loaded from {filepath}")

class GenreClassifier:
    """
    Classifies optimal music genres for different mental health conditions.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the genre classifier.
        
        Args:
            model_type (str): Type of model ('random_forest', 'svm', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.classes_ = None
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the genre classification model.
        
        Args:
            X_train: Training features
            y_train: Training targets (genre labels)
            
        Returns:
            Dict with training metrics
        """
        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            
            # Calculate training accuracy
            y_pred_train = self.model.predict(X_train)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            
            metrics = {
                'accuracy_train': accuracy_train,
                'n_classes': len(self.classes_)
            }
            
            logging.info(f"Genre classifier {self.model_type} trained successfully")
            return metrics
            
        except Exception as e:
            logging.error(f"Error training genre classifier: {e}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal genres for given conditions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted genre labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict genre probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            Prediction probabilities for each genre
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X_test)
    
    def get_genre_recommendations(self, X_test: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Get top-k genre recommendations with confidence scores.
        
        Args:
            X_test: Test features
            top_k: Number of top recommendations to return
            
        Returns:
            DataFrame with recommended genres and confidence scores
        """
        probabilities = self.predict_proba(X_test)
        
        recommendations = []
        for i, probs in enumerate(probabilities):
            # Get top-k genres
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_genres = self.classes_[top_indices]
            top_probs = probs[top_indices]
            
            for j, (genre, prob) in enumerate(zip(top_genres, top_probs)):
                recommendations.append({
                    'user_index': i,
                    'rank': j + 1,
                    'genre': genre,
                    'confidence': prob
                })
        
        return pd.DataFrame(recommendations)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy_test': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'classes_': self.classes_,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"Genre classifier saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.classes_ = model_data['classes_']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        logging.info(f"Genre classifier loaded from {filepath}")

class ModelEnsemble:
    """
    Ensemble of multiple models for improved predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
            weight: Model weight in ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            weight = self.weights[name] / total_weight
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)

def train_all_models(X_train: pd.DataFrame, y_mood_train: pd.Series, 
                    y_genre_train: pd.Series) -> Dict[str, Any]:
    """
    Train all models and return trained instances.
    
    Args:
        X_train: Training features
        y_mood_train: Mood improvement targets
        y_genre_train: Genre targets
        
    Returns:
        Dict with trained models
    """
    models = {}
    
    # Train mood prediction models
    for model_type in ['random_forest', 'xgboost', 'linear']:
        mood_model = MoodPredictionModel(model_type)
        mood_model.train(X_train, y_mood_train)
        models[f'mood_{model_type}'] = mood_model
    
    # Train genre classification models
    for model_type in ['random_forest', 'svm', 'logistic']:
        genre_model = GenreClassifier(model_type)
        genre_model.train(X_train, y_genre_train)
        models[f'genre_{model_type}'] = genre_model
    
    return models

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    data_path = "../data/raw/music_listening_data.csv"
    X_train, X_test, y_train, y_test, processor = preprocess_pipeline(data_path, 'mood_improvement')
    
    # Train mood prediction model
    mood_model = MoodPredictionModel('random_forest')
    train_metrics = mood_model.train(X_train, y_train)
    print("Training metrics:", train_metrics)
    
    # Evaluate model
    test_metrics = mood_model.evaluate(X_test, y_test)
    print("Test metrics:", test_metrics)