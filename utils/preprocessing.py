"""
Data Preprocessing Utilities for Forest Fire Prediction

This module contains functions for feature engineering, data cleaning,
and preprocessing for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import streamlit as st

def create_features(data):
    """
    Create additional features from the base forest fire dataset.
    
    Args:
        data (pd.DataFrame): Base forest fire dataset
        
    Returns:
        pd.DataFrame: Dataset with additional features
    """
    
    enhanced_data = data.copy()
    
    # Temperature-based features
    if 'temp' in enhanced_data.columns:
        # Temperature categories
        enhanced_data['temp_category'] = pd.cut(
            enhanced_data['temp'], 
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=['cold', 'cool', 'warm', 'hot']
        )
        
        # Squared temperature (non-linear relationship)
        enhanced_data['temp_squared'] = enhanced_data['temp'] ** 2
        
        # Temperature deviation from mean
        enhanced_data['temp_deviation'] = enhanced_data['temp'] - enhanced_data['temp'].mean()
    
    # Humidity-based features
    if 'RH' in enhanced_data.columns:
        # Humidity categories
        enhanced_data['humidity_category'] = pd.cut(
            enhanced_data['RH'],
            bins=[0, 30, 60, 80, 100],
            labels=['very_dry', 'dry', 'moderate', 'humid']
        )
        
        # Dryness index (inverse of humidity)
        enhanced_data['dryness_index'] = 100 - enhanced_data['RH']
        
        # Critical humidity threshold
        enhanced_data['critical_humidity'] = (enhanced_data['RH'] < 30).astype(int)
    
    # Wind-based features
    if 'wind' in enhanced_data.columns:
        # Wind categories
        enhanced_data['wind_category'] = pd.cut(
            enhanced_data['wind'],
            bins=[0, 5, 15, 25, np.inf],
            labels=['calm', 'light', 'moderate', 'strong']
        )
        
        # Wind speed squared (for spread rate calculations)
        enhanced_data['wind_squared'] = enhanced_data['wind'] ** 2
        
        # High wind flag
        enhanced_data['high_wind'] = (enhanced_data['wind'] > 20).astype(int)
    
    # Rain-based features
    if 'rain' in enhanced_data.columns:
        # Rain presence
        enhanced_data['has_rain'] = (enhanced_data['rain'] > 0).astype(int)
        
        # Drought conditions
        enhanced_data['drought_condition'] = (enhanced_data['rain'] < 0.1).astype(int)
        
        # Rain categories
        enhanced_data['rain_category'] = pd.cut(
            enhanced_data['rain'],
            bins=[0, 0.1, 1, 5, np.inf],
            labels=['none', 'light', 'moderate', 'heavy']
        )
    
    # Fire Weather Index combinations
    if all(col in enhanced_data.columns for col in ['FFMC', 'DMC', 'DC', 'ISI']):
        # Fire danger rating (simplified Canadian Fire Weather Index)
        enhanced_data['fire_danger_rating'] = calculate_fire_danger_rating(
            enhanced_data['FFMC'], enhanced_data['DMC'], 
            enhanced_data['DC'], enhanced_data['ISI']
        )
        
        # Buildup Index (combination of DMC and DC)
        enhanced_data['buildup_index'] = (enhanced_data['DMC'] + enhanced_data['DC']) / 2
        
        # Spread component (FFMC and wind interaction)
        if 'wind' in enhanced_data.columns:
            enhanced_data['spread_component'] = enhanced_data['FFMC'] * enhanced_data['wind'] / 100
    
    # Temporal features
    if 'month' in enhanced_data.columns:
        # Season
        season_mapping = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        enhanced_data['season'] = enhanced_data['month'].map(season_mapping)
        
        # Fire season flag (typically May to October)
        enhanced_data['fire_season'] = enhanced_data['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
        
        # Peak fire months (July, August, September)
        enhanced_data['peak_fire_month'] = enhanced_data['month'].isin([7, 8, 9]).astype(int)
        
        # Cyclical encoding for month
        enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
        enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
    
    # Day of week features
    if 'day' in enhanced_data.columns:
        # Weekend flag
        weekend_days = ['sat', 'sun']
        enhanced_data['is_weekend'] = enhanced_data['day'].isin(weekend_days).astype(int)
    
    # Spatial features
    if 'X' in enhanced_data.columns and 'Y' in enhanced_data.columns:
        # Distance from center
        center_x, center_y = enhanced_data['X'].mean(), enhanced_data['Y'].mean()
        enhanced_data['distance_from_center'] = np.sqrt(
            (enhanced_data['X'] - center_x)**2 + (enhanced_data['Y'] - center_y)**2
        )
        
        # Spatial clusters (simplified)
        enhanced_data['spatial_cluster'] = (
            (enhanced_data['X'] <= 5).astype(int) * 2 + 
            (enhanced_data['Y'] <= 5).astype(int)
        )
        
        # Border proximity (assuming grid is 1-9)
        enhanced_data['near_border'] = (
            (enhanced_data['X'] <= 2) | (enhanced_data['X'] >= 8) |
            (enhanced_data['Y'] <= 2) | (enhanced_data['Y'] >= 8)
        ).astype(int)
    
    # Interaction features
    if 'temp' in enhanced_data.columns and 'RH' in enhanced_data.columns:
        # Vapor pressure deficit (measure of atmospheric dryness)
        enhanced_data['vapor_pressure_deficit'] = calculate_vpd(
            enhanced_data['temp'], enhanced_data['RH']
        )
        
        # Heat index
        enhanced_data['heat_index'] = enhanced_data['temp'] + (enhanced_data['RH'] - 50) * 0.1
    
    if 'temp' in enhanced_data.columns and 'wind' in enhanced_data.columns:
        # Wind chill equivalent for fire (cooling effect)
        enhanced_data['fire_weather_index'] = enhanced_data['temp'] + enhanced_data['wind'] * 0.5
    
    # Drought stress indicators
    if all(col in enhanced_data.columns for col in ['temp', 'RH', 'rain']):
        # Simple drought stress index
        enhanced_data['drought_stress'] = (
            enhanced_data['temp'] / 30 + 
            (100 - enhanced_data['RH']) / 100 + 
            np.log1p(1 / (enhanced_data['rain'] + 0.1))
        ) / 3
    
    # Fuel moisture content estimation
    if all(col in enhanced_data.columns for col in ['temp', 'RH', 'rain']):
        enhanced_data['estimated_fuel_moisture'] = estimate_fuel_moisture(
            enhanced_data['temp'], enhanced_data['RH'], enhanced_data['rain']
        )
    
    return enhanced_data

def calculate_fire_danger_rating(ffmc, dmc, dc, isi):
    """
    Calculate a simplified fire danger rating based on Canadian Fire Weather Index components.
    
    Args:
        ffmc (pd.Series): Fine Fuel Moisture Code
        dmc (pd.Series): Duff Moisture Code  
        dc (pd.Series): Drought Code
        isi (pd.Series): Initial Spread Index
        
    Returns:
        pd.Series: Fire danger rating (0-10 scale)
    """
    
    # Normalize components to 0-1 scale
    ffmc_norm = ffmc / 100
    dmc_norm = np.clip(dmc / 100, 0, 1)
    dc_norm = np.clip(dc / 400, 0, 1)
    isi_norm = np.clip(isi / 20, 0, 1)
    
    # Weighted combination
    danger_rating = (
        ffmc_norm * 0.3 +
        dmc_norm * 0.25 +
        dc_norm * 0.25 +
        isi_norm * 0.2
    ) * 10
    
    return danger_rating.clip(0, 10)

def calculate_vpd(temperature, humidity):
    """
    Calculate Vapor Pressure Deficit (VPD) as a measure of atmospheric dryness.
    
    Args:
        temperature (pd.Series): Temperature in Celsius
        humidity (pd.Series): Relative humidity in percentage
        
    Returns:
        pd.Series: Vapor pressure deficit in kPa
    """
    
    # Saturation vapor pressure using Magnus formula
    svp = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
    
    # Actual vapor pressure
    avp = svp * humidity / 100
    
    # Vapor pressure deficit
    vpd = svp - avp
    
    return vpd

def estimate_fuel_moisture(temperature, humidity, rain):
    """
    Estimate fuel moisture content based on weather conditions.
    
    Args:
        temperature (pd.Series): Temperature in Celsius
        humidity (pd.Series): Relative humidity in percentage
        rain (pd.Series): Rain in mm
        
    Returns:
        pd.Series: Estimated fuel moisture percentage
    """
    
    # Base fuel moisture from humidity
    base_moisture = humidity * 0.3
    
    # Temperature adjustment (higher temp reduces moisture)
    temp_adjustment = (temperature - 20) * -0.5
    
    # Rain adjustment (recent rain increases moisture)
    rain_adjustment = np.minimum(rain * 2, 10)
    
    # Combine adjustments
    fuel_moisture = base_moisture + temp_adjustment + rain_adjustment
    
    # Ensure reasonable bounds
    fuel_moisture = fuel_moisture.clip(5, 40)
    
    return fuel_moisture

def preprocess_data(data, target_column='area', problem_type='regression', 
                   feature_selection=True, n_features=None):
    """
    Complete preprocessing pipeline for forest fire data.
    
    Args:
        data (pd.DataFrame): Raw forest fire dataset
        target_column (str): Name of target variable column
        problem_type (str): 'regression' or 'classification'
        feature_selection (bool): Whether to perform feature selection
        n_features (int): Number of features to select (None for automatic)
        
    Returns:
        tuple: (X_processed, y, feature_names, preprocessors)
    """
    
    # Create enhanced features
    processed_data = create_features(data.copy())
    
    # Prepare target variable
    if target_column in processed_data.columns:
        if problem_type == 'regression':
            y = processed_data[target_column].copy()
            # Log transform for area (common for fire data)
            y = np.log1p(y)
        else:
            # Create categorical target for classification
            y = create_fire_risk_categories(processed_data[target_column])
    else:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Remove target from features
    feature_data = processed_data.drop(columns=[target_column])
    
    # Handle categorical variables
    categorical_columns = feature_data.select_dtypes(include=['object', 'category']).columns
    categorical_encoders = {}
    
    for col in categorical_columns:
        if feature_data[col].nunique() < 50:  # Only encode if reasonable number of categories
            le = LabelEncoder()
            feature_data[col] = le.fit_transform(feature_data[col].astype(str))
            categorical_encoders[col] = le
        else:
            # Drop high cardinality categorical columns
            feature_data = feature_data.drop(columns=[col])
    
    # Select numeric features only
    numeric_features = feature_data.select_dtypes(include=[np.number])
    
    # Handle missing values
    numeric_features = numeric_features.fillna(numeric_features.median())
    
    # Feature selection
    if feature_selection and len(numeric_features.columns) > 1:
        if n_features is None:
            n_features = min(15, len(numeric_features.columns))  # Select top 15 features
        
        if problem_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        else:
            selector = SelectKBest(score_func=f_classif, k=n_features)
        
        # Remove any infinite or NaN values before feature selection
        mask = np.isfinite(numeric_features.values).all(axis=1) & np.isfinite(y)
        
        if mask.sum() > 0:
            X_selected = selector.fit_transform(numeric_features[mask], y[mask])
            selected_features = numeric_features.columns[selector.get_support()]
            
            # Apply selection to full dataset
            X_processed = numeric_features[selected_features]
        else:
            st.warning("No valid samples found for feature selection. Using all features.")
            X_processed = numeric_features
            selected_features = numeric_features.columns
            selector = None
    else:
        X_processed = numeric_features
        selected_features = numeric_features.columns
        selector = None
    
    # Store preprocessing information
    preprocessors = {
        'categorical_encoders': categorical_encoders,
        'feature_selector': selector,
        'selected_features': list(selected_features)
    }
    
    return X_processed, y, list(selected_features), preprocessors

def create_fire_risk_categories(area_data):
    """
    Create fire risk categories from burned area data.
    
    Args:
        area_data (pd.Series): Burned area data
        
    Returns:
        pd.Series: Fire risk categories (0: No fire, 1: Low, 2: Medium, 3: High)
    """
    
    def categorize_risk(area):
        if area == 0:
            return 0  # No fire
        elif area <= 1:
            return 1  # Low risk (small fire)
        elif area <= 10:
            return 2  # Medium risk
        else:
            return 3  # High risk (large fire)
    
    return area_data.apply(categorize_risk)

def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        method (str): Method for outlier detection ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Boolean mask indicating outliers
    """
    
    outlier_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_mask[col] = z_scores > threshold
    
    return outlier_mask

def handle_outliers(data, outlier_mask, method='winsorize', percentiles=(5, 95)):
    """
    Handle outliers in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        outlier_mask (pd.DataFrame): Boolean mask indicating outliers
        method (str): Method for handling outliers ('remove', 'winsorize', 'transform')
        percentiles (tuple): Percentiles for winsorizing
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    
    processed_data = data.copy()
    
    if method == 'remove':
        # Remove rows with any outliers
        rows_with_outliers = outlier_mask.any(axis=1)
        processed_data = processed_data[~rows_with_outliers]
        
    elif method == 'winsorize':
        # Cap outliers at specified percentiles
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if outlier_mask[col].any():
                lower_percentile = np.percentile(data[col], percentiles[0])
                upper_percentile = np.percentile(data[col], percentiles[1])
                
                processed_data[col] = processed_data[col].clip(
                    lower=lower_percentile, 
                    upper=upper_percentile
                )
    
    elif method == 'transform':
        # Apply log transformation to reduce outlier impact
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if outlier_mask[col].any() and (data[col] > 0).all():
                processed_data[col] = np.log1p(processed_data[col])
    
    return processed_data

def scale_features(X_train, X_test=None, method='standard'):
    """
    Scale features using specified method.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features (optional)
        method (str): Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, scaler

def create_polynomial_features(X, degree=2, interaction_only=False):
    """
    Create polynomial features.
    
    Args:
        X (pd.DataFrame): Input features
        degree (int): Polynomial degree
        interaction_only (bool): Only create interaction terms
        
    Returns:
        tuple: (X_poly, feature_names)
    """
    
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(
        degree=degree, 
        interaction_only=interaction_only,
        include_bias=False
    )
    
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    
    return X_poly, feature_names, poly

def validate_preprocessing(X, y):
    """
    Validate preprocessing results.
    
    Args:
        X (pd.DataFrame): Processed features
        y (pd.Series): Target variable
        
    Returns:
        dict: Validation results
    """
    
    validation_results = {
        'feature_shape': X.shape,
        'target_shape': y.shape,
        'missing_features': X.isnull().sum().sum(),
        'missing_target': y.isnull().sum(),
        'infinite_features': np.isinf(X.values).sum(),
        'infinite_target': np.isinf(y.values).sum() if hasattr(y, 'values') else 0,
        'feature_ranges': {},
        'target_range': (y.min(), y.max())
    }
    
    # Check feature ranges
    for col in X.columns:
        validation_results['feature_ranges'][col] = (X[col].min(), X[col].max())
    
    # Check for constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
    
    validation_results['constant_features'] = constant_features
    
    return validation_results

