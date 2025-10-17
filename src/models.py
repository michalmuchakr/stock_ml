"""
Machine learning models module for stock prediction.
Handles model training, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any, List


class ModelTrainer:
    """Handles training of machine learning models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create model instances with optimized hyperparameters.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.random_state
            )
        }
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all models on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        models = self.create_models()
        
        print("Training models...")
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        print("Model training completed.")
        return self.models
    
    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
            threshold: Classification threshold
            
        Returns:
            Dictionary of predictions for each model
        """
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                predictions[name] = (proba >= threshold).astype(int)
            else:
                predictions[name] = model.predict(X_test)
        
        return predictions
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for models that support it.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance DataFrames
        """
        importance_dfs = {}
        
        for name, importance in self.feature_importance.items():
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            importance_dfs[name] = df
        
        return importance_dfs


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate hit rate (fraction of correct predictions)."""
        return (y_pred == y_true).mean()
    
    @staticmethod
    def evaluate_models(y_test: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models and return metrics.
        
        Args:
            y_test: True labels
            predictions: Dictionary of predictions for each model
            
        Returns:
            Dictionary of metrics for each model
        """
        results = {}
        
        for name, y_pred in predictions.items():
            accuracy = ModelEvaluator.calculate_accuracy(y_test, y_pred)
            hit_rate = ModelEvaluator.calculate_hit_rate(y_test, y_pred)
            
            results[name] = {
                "accuracy": accuracy,
                "hit_rate": hit_rate
            }
        
        return results
    
    @staticmethod
    def print_evaluation_results(results: Dict[str, Dict[str, float]]) -> None:
        """Print evaluation results in a formatted way."""
        print("\n=== Model Evaluation Results ===")
        for name, metrics in results.items():
            print(f"{name.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            print()


class ModelManager:
    """Main model management class."""
    
    def __init__(self, random_state: int = 42):
        self.trainer = ModelTrainer(random_state)
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.predictions = {}
        self.evaluation_results = {}
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            Dictionary containing models, predictions, and results
        """
        # Train models
        self.models = self.trainer.train_models(X_train, y_train)
        
        # Make predictions
        self.predictions = self.trainer.predict(X_test)
        
        # Evaluate models
        self.evaluation_results = self.evaluator.evaluate_models(y_test, self.predictions)
        
        # Print results
        self.evaluator.print_evaluation_results(self.evaluation_results)
        
        # Get feature importance
        feature_importance = self.trainer.get_feature_importance(feature_names)
        
        return {
            "models": self.models,
            "predictions": self.predictions,
            "evaluation_results": self.evaluation_results,
            "feature_importance": feature_importance
        }
    
    def get_model(self, name: str):
        """Get a specific trained model."""
        return self.models.get(name)
    
    def get_predictions(self, name: str) -> np.ndarray:
        """Get predictions for a specific model."""
        return self.predictions.get(name)
    
    def get_evaluation_results(self, name: str) -> Dict[str, float]:
        """Get evaluation results for a specific model."""
        return self.evaluation_results.get(name, {})
