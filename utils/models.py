"""
Machine Learning Models for Forest Fire Prediction

This module contains model training, evaluation, and utility functions
for forest fire prediction and classification tasks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import streamlit as st

def train_regression_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models for forest fire area prediction.
    
    Args:
        X_train (array-like): Training features
        X_test (array-like): Test features  
        y_train (array-like): Training target
        y_test (array-like): Test target
        
    Returns:
        tuple: (trained_models, metrics)
    """
    
    models = {}
    metrics = {}
    
    # Define models with their configurations
    model_configs = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        ),
        'Support Vector Regression': SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        ),
        'K-Nearest Neighbors': KNeighborsRegressor(
            n_neighbors=5,
            weights='distance'
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    for model_name, model in model_configs.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store model and metrics
            models[model_name] = model
            metrics[model_name] = {
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'R² Score': test_r2,
                'RMSE': np.sqrt(test_mse),
                'MAE': test_mae
            }
            
        except Exception as e:
            st.warning(f"Error training {model_name}: {str(e)}")
            continue
    
    return models, metrics

def train_classification_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models for forest fire risk prediction.
    
    Args:
        X_train (array-like): Training features
        X_test (array-like): Test features
        y_train (array-like): Training target
        y_test (array-like): Test target
        
    Returns:
        tuple: (trained_models, metrics)
    """
    
    models = {}
    metrics = {}
    
    # Define models with their configurations
    model_configs = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        ),
        'Support Vector Classifier': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    for model_name, model in model_configs.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # For multiclass classification, use weighted average
            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            # Store model and metrics
            models[model_name] = model
            metrics[model_name] = {
                'Train Accuracy': train_accuracy,
                'Accuracy': test_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
            
        except Exception as e:
            st.warning(f"Error training {model_name}: {str(e)}")
            continue
    
    return models, metrics

def optimize_model_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring=None):
    """
    Optimize model hyperparameters using grid search.
    
    Args:
        model: Scikit-learn model instance
        param_grid (dict): Parameter grid for grid search
        X_train (array-like): Training features
        y_train (array-like): Training target
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        sklearn.model_selection.GridSearchCV: Fitted grid search object
    """
    
    if scoring is None:
        # Auto-determine scoring based on model type
        if hasattr(model, 'predict_proba'):
            scoring = 'accuracy'
        else:
            scoring = 'r2'
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained models.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    
    importance_df = pd.DataFrame()
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) > 1:
            # Multi-class classification
            avg_coef = np.mean(np.abs(model.coef_), axis=0)
        else:
            avg_coef = np.abs(model.coef_)
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_coef
        }).sort_values('importance', ascending=False)
    
    return importance_df

def evaluate_model_performance(model, X_test, y_test, problem_type='regression'):
    """
    Comprehensive model performance evaluation.
    
    Args:
        model: Trained model
        X_test (array-like): Test features
        y_test (array-like): Test target
        problem_type (str): 'regression' or 'classification'
        
    Returns:
        dict: Performance metrics
    """
    
    predictions = model.predict(X_test)
    performance = {}
    
    if problem_type == 'regression':
        performance = {
            'MSE': mean_squared_error(y_test, predictions),
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R²': r2_score(y_test, predictions)
        }
        
        # Additional regression metrics
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        performance['Explained Variance'] = 1 - (ss_res / ss_tot)
        
        # Mean Absolute Percentage Error (MAPE)
        non_zero_mask = y_test != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
            performance['MAPE'] = mape
    
    else:
        performance = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        # Classification report
        performance['Classification Report'] = classification_report(y_test, predictions, output_dict=True)
        performance['Confusion Matrix'] = confusion_matrix(y_test, predictions)
        
        # Per-class metrics
        if hasattr(model, 'classes_'):
            for i, class_label in enumerate(model.classes_):
                class_precision = precision_score(y_test, predictions, labels=[class_label], average='micro', zero_division=0)
                class_recall = recall_score(y_test, predictions, labels=[class_label], average='micro', zero_division=0)
                performance[f'Class {class_label} Precision'] = class_precision
                performance[f'Class {class_label} Recall'] = class_recall
    
    return performance

def create_ensemble_model(models, method='voting', weights=None):
    """
    Create ensemble model from multiple trained models.
    
    Args:
        models (dict): Dictionary of trained models
        method (str): Ensemble method ('voting', 'stacking')
        weights (list): Weights for voting ensemble
        
    Returns:
        ensemble model
    """
    
    from sklearn.ensemble import VotingRegressor, VotingClassifier
    
    # Determine if classification or regression based on first model
    first_model = list(models.values())[0]
    is_classifier = hasattr(first_model, 'predict_proba')
    
    model_list = [(name, model) for name, model in models.items()]
    
    if method == 'voting':
        if is_classifier:
            ensemble = VotingClassifier(
                estimators=model_list,
                voting='soft' if all(hasattr(m, 'predict_proba') for _, m in model_list) else 'hard',
                weights=weights
            )
        else:
            ensemble = VotingRegressor(
                estimators=model_list,
                weights=weights
            )
    
    elif method == 'stacking':
        from sklearn.ensemble import StackingRegressor, StackingClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        
        if is_classifier:
            ensemble = StackingClassifier(
                estimators=model_list,
                final_estimator=LogisticRegression(),
                cv=5
            )
        else:
            ensemble = StackingRegressor(
                estimators=model_list,
                final_estimator=LinearRegression(),
                cv=5
            )
    
    return ensemble

def validate_model_robustness(model, X, y, cv_folds=5, random_seeds=None):
    """
    Validate model robustness across different data splits and random seeds.
    
    Args:
        model: Model to validate
        X (array-like): Features
        y (array-like): Target
        cv_folds (int): Number of cross-validation folds
        random_seeds (list): List of random seeds to test
        
    Returns:
        dict: Robustness metrics
    """
    
    if random_seeds is None:
        random_seeds = [42, 123, 456, 789, 999]
    
    results = {
        'cv_scores': [],
        'seed_scores': []
    }
    
    # Cross-validation robustness
    if hasattr(model, 'predict_proba'):
        scoring = 'accuracy'
    else:
        scoring = 'r2'
    
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
    results['cv_scores'] = cv_scores
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()
    
    # Random seed robustness
    seed_scores = []
    for seed in random_seeds:
        # Create new model instance with different random seed
        model_copy = type(model)(**model.get_params())
        if hasattr(model_copy, 'random_state'):
            model_copy.set_params(random_state=seed)
        
        # Evaluate with cross-validation
        scores = cross_val_score(model_copy, X, y, cv=3, scoring=scoring)
        seed_scores.append(scores.mean())
    
    results['seed_scores'] = seed_scores
    results['seed_mean'] = np.mean(seed_scores)
    results['seed_std'] = np.std(seed_scores)
    
    return results

def analyze_prediction_intervals(model, X_test, confidence_level=0.95):
    """
    Analyze prediction intervals for regression models.
    
    Args:
        model: Trained regression model
        X_test (array-like): Test features
        confidence_level (float): Confidence level for intervals
        
    Returns:
        dict: Prediction intervals and statistics
    """
    
    predictions = model.predict(X_test)
    
    # For ensemble models, try to get prediction intervals
    if hasattr(model, 'estimators_'):
        # For ensemble models, use predictions from individual estimators
        individual_predictions = np.array([
            estimator.predict(X_test) for estimator in model.estimators_
        ])
        
        pred_mean = np.mean(individual_predictions, axis=0)
        pred_std = np.std(individual_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_bound = pred_mean - 1.96 * pred_std  # Assuming normal distribution
        upper_bound = pred_mean + 1.96 * pred_std
        
        return {
            'predictions': predictions,
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    else:
        # For single models, return basic statistics
        return {
            'predictions': predictions,
            'prediction_mean': predictions,
            'prediction_std': np.zeros_like(predictions),
            'lower_bound': predictions,
            'upper_bound': predictions,
            'interval_width': np.zeros_like(predictions)
        }

def compare_models_statistically(model1_scores, model2_scores, alpha=0.05):
    """
    Perform statistical comparison between two models using paired t-test.
    
    Args:
        model1_scores (array-like): Cross-validation scores for model 1
        model2_scores (array-like): Cross-validation scores for model 2
        alpha (float): Significance level
        
    Returns:
        dict: Statistical test results
    """
    
    from scipy import stats
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Effect size (Cohen's d)
    diff_scores = model1_scores - model2_scores
    pooled_std = np.sqrt((np.var(model1_scores) + np.var(model2_scores)) / 2)
    cohens_d = np.mean(diff_scores) / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_difference': np.mean(diff_scores),
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'cohens_d': cohens_d,
        'confidence_interval': stats.t.interval(
            1 - alpha, 
            len(diff_scores) - 1,
            loc=np.mean(diff_scores),
            scale=stats.sem(diff_scores)
        )
    }

def create_model_explanation(model, feature_names, X_sample):
    """
    Create model explanations using SHAP-like attribution (simplified).
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        X_sample (array-like): Sample instances for explanation
        
    Returns:
        dict: Model explanation results
    """
    
    explanations = {}
    
    # Feature importance (global explanation)
    if hasattr(model, 'feature_importances_'):
        explanations['global_importance'] = dict(zip(feature_names, model.feature_importances_))
    
    # For tree-based models, try to get path information
    if hasattr(model, 'decision_path'):
        try:
            decision_paths = model.decision_path(X_sample)
            explanations['decision_paths'] = decision_paths
        except:
            pass
    
    # Linear model coefficients
    if hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            explanations['coefficients'] = dict(zip(feature_names, model.coef_[0]))
        else:
            explanations['coefficients'] = dict(zip(feature_names, model.coef_))
    
    return explanations

def save_model_artifacts(models, preprocessors, filepath):
    """
    Save trained models and preprocessing artifacts.
    
    Args:
        models (dict): Trained models
        preprocessors (dict): Preprocessing objects
        filepath (str): Path to save artifacts
    """
    
    import joblib
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    artifacts = {
        'models': models,
        'preprocessors': preprocessors,
        'metadata': {
            'model_count': len(models),
            'model_names': list(models.keys()),
            'preprocessing_steps': list(preprocessors.keys())
        }
    }
    
    joblib.dump(artifacts, filepath)
    st.success(f"Model artifacts saved to {filepath}")

def load_model_artifacts(filepath):
    """
    Load trained models and preprocessing artifacts.
    
    Args:
        filepath (str): Path to load artifacts from
        
    Returns:
        dict: Loaded artifacts
    """
    
    import joblib
    
    try:
        artifacts = joblib.load(filepath)
        st.success(f"Model artifacts loaded from {filepath}")
        return artifacts
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None

