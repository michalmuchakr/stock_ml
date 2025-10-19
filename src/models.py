"""
Machine learning models module for stock prediction.
Handles model training, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any, List


class ModelTrainer:
    """Handles training of machine learning models."""
    
    def __init__(self, random_state: int = 42, enable_hyperparameter_tuning: bool = False):
        self.random_state = random_state
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.models = {}
        self.feature_importance = {}
        self.best_params = {}
    
    def _get_parameter_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Define parameter grids for hyperparameter tuning.
        
        Returns:
            Dictionary of parameter grids for each model
        """
        return {
            "random_forest": {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6, 8, 10, None],
                "min_samples_leaf": [1, 2, 3, 5, 10],
                "min_samples_split": [2, 5, 10, 15],
                "max_features": ["sqrt", "log2", None]
            },
            "xgboost": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1, 0.2]
            }
        }
    
    def _get_default_models(self) -> Dict[str, Any]:
        """
        Create model instances with default hyperparameters.
        
        Returns:
            Dictionary of model instances
        """
        return {
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
                random_state=self.random_state,
                verbosity=0
            )
        }
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV for Random Forest and early stopping for XGBoost.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of best models for each algorithm
        """
        print("Performing hyperparameter tuning...")
        parameter_grids = self._get_parameter_grids()
        tuned_models = {}
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Tune Random Forest with RandomizedSearchCV (faster than GridSearchCV)
        print("  Tuning random_forest...")
        rf_base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_random_search = RandomizedSearchCV(
            estimator=rf_base_model,
            param_distributions=parameter_grids["random_forest"],
            cv=tscv,
            scoring='accuracy',
            n_iter=20,  # Number of parameter settings sampled
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state
        )
        
        rf_random_search.fit(X_train, y_train)
        tuned_models["random_forest"] = rf_random_search.best_estimator_
        self.best_params["random_forest"] = rf_random_search.best_params_
        
        print(f"    Best parameters: {rf_random_search.best_params_}")
        print(f"    Best CV score: {rf_random_search.best_score_:.4f}")
        
        # Tune XGBoost with early stopping approach
        print("  Tuning xgboost...")
        best_xgb_score = 0
        best_xgb_params = None
        best_xgb_model = None
        
        # Split data for early stopping
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
        
        # Early stopping-based tuning for XGBoost
        for n_est in parameter_grids["xgboost"]["n_estimators"]:
            for max_d in parameter_grids["xgboost"]["max_depth"]:
                for lr in parameter_grids["xgboost"]["learning_rate"]:
                    xgb_model = XGBClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        learning_rate=lr,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=self.random_state,
                        verbosity=0,
                        early_stopping_rounds=10
                    )
                    
                    # Use early stopping for faster training
                    xgb_model.fit(
                        X_train_split, y_train_split,
                        eval_set=[(X_val_split, y_val_split)],
                        verbose=False
                    )
                    
                    # Evaluate on validation set
                    val_score = xgb_model.score(X_val_split, y_val_split)
                    
                    if val_score > best_xgb_score:
                        best_xgb_score = val_score
                        best_xgb_params = {
                            'n_estimators': xgb_model.get_booster().num_boosted_rounds(),
                            'max_depth': max_d,
                            'learning_rate': lr
                        }
                        best_xgb_model = xgb_model
        
        tuned_models["xgboost"] = best_xgb_model
        self.best_params["xgboost"] = best_xgb_params
        
        print(f"    Best parameters: {best_xgb_params}")
        print(f"    Best validation score: {best_xgb_score:.4f}")
        
        return tuned_models
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create model instances with optimized hyperparameters.
        
        Returns:
            Dictionary of model instances
        """
        if self.enable_hyperparameter_tuning:
            # Return default models - hyperparameter tuning will be done during training
            return self._get_default_models()
        else:
            return self._get_default_models()
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all models on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        if self.enable_hyperparameter_tuning:
            # Perform hyperparameter tuning first
            models = self._tune_hyperparameters(X_train, y_train)
            print("Hyperparameter tuning completed.")
        else:
            models = self.create_models()
        
        print("Training models...")
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # For XGBoost, create a fresh model without early stopping for final training
            if name == "xgboost" and self.enable_hyperparameter_tuning:
                # Get the best parameters from tuning
                best_params = self.best_params.get(name, {})
                final_model = XGBClassifier(
                    n_estimators=best_params.get('n_estimators', 300),
                    max_depth=best_params.get('max_depth', 4),
                    learning_rate=best_params.get('learning_rate', 0.1),
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbosity=0
                )
                final_model.fit(X_train, y_train)
                self.models[name] = final_model
            else:
                model.fit(X_train, y_train)
                self.models[name] = model
            
            # Store feature importance
            if hasattr(self.models[name], 'feature_importances_'):
                self.feature_importance[name] = self.models[name].feature_importances_
        
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
    
    def __init__(self, random_state: int = 42, enable_hyperparameter_tuning: bool = False):
        self.trainer = ModelTrainer(random_state, enable_hyperparameter_tuning)
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.predictions = {}
        self.evaluation_results = {}
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
    
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
        
        # Print hyperparameter tuning results if enabled
        if self.enable_hyperparameter_tuning and self.trainer.best_params:
            self._print_hyperparameter_results()
        
        # Get feature importance
        feature_importance = self.trainer.get_feature_importance(feature_names)
        
        return {
            "models": self.models,
            "predictions": self.predictions,
            "evaluation_results": self.evaluation_results,
            "feature_importance": feature_importance,
            "best_params": self.trainer.best_params if self.enable_hyperparameter_tuning else {}
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
    
    def _print_hyperparameter_results(self):
        """Print hyperparameter tuning results."""
        print("\n=== Hyperparameter Tuning Results ===")
        for name, params in self.trainer.best_params.items():
            print(f"{name.replace('_', ' ').title()}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            print()
