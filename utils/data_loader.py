"""
Forest Fire Data Loading Utilities

This module handles loading and initial processing of forest fire datasets.
"""

import pandas as pd
import numpy as np
import os
import requests
from io import StringIO
import streamlit as st

def load_forest_fire_data():
    """
    Load forest fire dataset. Tries multiple sources including online datasets.
    
    Returns:
        pd.DataFrame: Forest fire dataset with standard columns
    """
    
    # Try to load from local file first
    local_paths = [
        "data/forestfires.csv",
        "forestfires.csv",
        "data/forest_fire_data.csv"
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                st.info(f"Loaded data from local file: {path}")
                return validate_and_clean_data(data)
            except Exception as e:
                st.warning(f"Could not load from {path}: {str(e)}")
                continue
    
    # Try to download from UCI ML Repository (forest fires dataset)
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = pd.read_csv(StringIO(response.text))
        st.success("Successfully downloaded forest fire data from UCI ML Repository")
        return validate_and_clean_data(data)
        
    except Exception as e:
        st.warning(f"Could not download from UCI: {str(e)}")
    
    # Generate synthetic data as last resort with clear indication
    st.warning("Using generated sample data for demonstration. For production use, please provide real forest fire data.")
    return generate_sample_data()

def validate_and_clean_data(data):
    """
    Validate and clean the loaded forest fire data.
    
    Args:
        data (pd.DataFrame): Raw forest fire data
        
    Returns:
        pd.DataFrame: Cleaned and validated data
    """
    
    # Check for required columns and rename if necessary
    column_mapping = {
        'FFMC': 'FFMC',
        'DMC': 'DMC', 
        'DC': 'DC',
        'ISI': 'ISI',
        'temp': 'temp',
        'RH': 'RH',
        'wind': 'wind',
        'rain': 'rain',
        'area': 'area',
        'X': 'X',
        'Y': 'Y',
        'month': 'month',
        'day': 'day'
    }
    
    # Handle month name to number conversion if needed
    if 'month' in data.columns:
        month_mapping = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        if data['month'].dtype == 'object':
            data['month'] = data['month'].str.lower().map(month_mapping)
    
    # Handle day name conversion if needed
    if 'day' in data.columns and data['day'].dtype == 'object':
        # Keep as string for categorical encoding later
        pass
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area', 'X', 'Y']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle missing values
    # For weather variables, forward fill or use median
    weather_cols = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI']
    for col in weather_cols:
        if col in data.columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
    
    # Remove rows with missing critical information
    critical_cols = ['area']
    data = data.dropna(subset=[col for col in critical_cols if col in data.columns])
    
    # Ensure area is non-negative
    if 'area' in data.columns:
        data['area'] = data['area'].clip(lower=0)
    
    # Validate coordinate ranges (if present)
    if 'X' in data.columns and 'Y' in data.columns:
        # Remove obviously invalid coordinates
        data = data[(data['X'] >= 0) & (data['Y'] >= 0)]
        data = data[(data['X'] <= 10) & (data['Y'] <= 10)]  # Typical range for this dataset
    
    # Validate weather ranges
    if 'temp' in data.columns:
        data = data[(data['temp'] >= -10) & (data['temp'] <= 50)]  # Reasonable temperature range
    
    if 'RH' in data.columns:
        data = data[(data['RH'] >= 0) & (data['RH'] <= 100)]  # Humidity percentage
    
    if 'wind' in data.columns:
        data = data[(data['wind'] >= 0) & (data['wind'] <= 100)]  # Wind speed km/h
    
    if 'rain' in data.columns:
        data = data[data['rain'] >= 0]  # Non-negative rain
    
    # Reset index
    data = data.reset_index(drop=True)
    
    st.info(f"Data validation complete. Final dataset shape: {data.shape}")
    
    return data

def generate_sample_data():
    """
    Generate realistic sample forest fire data for demonstration.
    
    Returns:
        pd.DataFrame: Generated sample data
    """
    
    np.random.seed(42)  # For reproducibility
    n_samples = 517  # Similar to original dataset size
    
    # Generate weather conditions
    temp = np.random.normal(20, 8, n_samples).clip(0, 40)
    RH = np.random.normal(50, 20, n_samples).clip(10, 90)
    wind = np.random.exponential(8, n_samples).clip(0, 40)
    rain = np.random.exponential(0.5, n_samples).clip(0, 20)
    
    # Generate fire weather indices with correlations
    FFMC = 80 + (temp - 20) * 0.5 + np.random.normal(0, 5, n_samples)
    FFMC = FFMC.clip(50, 100)
    
    DMC = 30 + (100 - RH) * 0.3 + np.random.normal(0, 10, n_samples)
    DMC = DMC.clip(0, 200)
    
    DC = 100 + temp * 2 - rain * 5 + np.random.normal(0, 50, n_samples)
    DC = DC.clip(0, 800)
    
    ISI = wind * 0.5 + (FFMC - 80) * 0.1 + np.random.normal(0, 2, n_samples)
    ISI = ISI.clip(0, 20)
    
    # Generate spatial coordinates
    X = np.random.randint(1, 10, n_samples)
    Y = np.random.randint(1, 10, n_samples)
    
    # Generate temporal data
    month = np.random.randint(1, 13, n_samples)
    day = np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], n_samples)
    
    # Generate burned area with realistic distribution
    # Most fires are small, few are large
    fire_probability = 1 / (1 + np.exp(-(FFMC - 85) * 0.2 + (RH - 50) * 0.05 + wind * 0.1 - rain * 0.3))
    
    area = np.zeros(n_samples)
    
    for i in range(n_samples):
        if np.random.random() < fire_probability[i] * 0.3:  # 30% base fire occurrence
            # Log-normal distribution for fire size
            log_area = np.random.normal(0, 1.5)
            area[i] = np.exp(log_area) - 1
            area[i] = max(0, area[i])
    
    # Create DataFrame
    data = pd.DataFrame({
        'X': X,
        'Y': Y,
        'month': month,
        'day': day,
        'FFMC': FFMC.round(2),
        'DMC': DMC.round(2),
        'DC': DC.round(2),
        'ISI': ISI.round(2),
        'temp': temp.round(1),
        'RH': RH.round(1),
        'wind': wind.round(1),
        'rain': rain.round(2),
        'area': area.round(2)
    })
    
    return data

def load_additional_datasets():
    """
    Load additional forest fire related datasets if available.
    
    Returns:
        dict: Dictionary of additional datasets
    """
    
    additional_data = {}
    
    # Try to load vegetation data
    vegetation_files = [
        "data/vegetation.csv",
        "vegetation.csv"
    ]
    
    for file_path in vegetation_files:
        if os.path.exists(file_path):
            try:
                vegetation_data = pd.read_csv(file_path)
                additional_data['vegetation'] = vegetation_data
                break
            except Exception as e:
                st.warning(f"Could not load vegetation data from {file_path}: {str(e)}")
    
    # Try to load weather station data
    weather_files = [
        "data/weather_stations.csv",
        "weather_stations.csv"
    ]
    
    for file_path in weather_files:
        if os.path.exists(file_path):
            try:
                weather_data = pd.read_csv(file_path)
                additional_data['weather_stations'] = weather_data
                break
            except Exception as e:
                st.warning(f"Could not load weather station data from {file_path}: {str(e)}")
    
    return additional_data

def get_data_info(data):
    """
    Get comprehensive information about the dataset.
    
    Args:
        data (pd.DataFrame): Forest fire dataset
        
    Returns:
        dict: Dataset information
    """
    
    info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'data_types': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    # Categorical column analysis
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        info['categorical_analysis'] = {}
        for col in categorical_cols:
            info['categorical_analysis'][col] = {
                'unique_values': data[col].nunique(),
                'value_counts': data[col].value_counts().to_dict()
            }
    
    # Outlier detection for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info['outliers'] = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            info['outliers'][col] = len(outliers)
    
    return info

def save_processed_data(data, filename="processed_forest_fire_data.csv"):
    """
    Save processed data to file.
    
    Args:
        data (pd.DataFrame): Processed forest fire data
        filename (str): Output filename
    """
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        file_path = os.path.join("data", filename)
        data.to_csv(file_path, index=False)
        
        st.success(f"Data saved to {file_path}")
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

def create_data_splits(data, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits for the data.
    
    Args:
        data (pd.DataFrame): Forest fire dataset
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=create_stratification_column(data)
    )
    
    # Second split: separate validation from training
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=create_stratification_column(train_val_data)
        )
    else:
        train_data = train_val_data
        val_data = None
    
    return train_data, val_data, test_data

def create_stratification_column(data):
    """
    Create a column for stratified sampling based on fire risk levels.
    
    Args:
        data (pd.DataFrame): Forest fire dataset
        
    Returns:
        pd.Series: Stratification column
    """
    
    if 'area' not in data.columns:
        return None
    
    def categorize_fire_risk(area):
        if area == 0:
            return 'no_fire'
        elif area <= 1:
            return 'small'
        elif area <= 10:
            return 'medium'
        else:
            return 'large'
    
    return data['area'].apply(categorize_fire_risk)

