# Forest Fire Prediction & Simulation System

## Overview

This is a comprehensive forest fire prediction and simulation system built with Streamlit. The application provides data analysis, machine learning model training, fire risk prediction, fire spread simulation, and model performance evaluation capabilities. It's designed to help forest management professionals and researchers understand and predict forest fire behavior using historical data and real-time environmental conditions.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - A Python web framework for rapid development of data applications
- **Multi-page Structure**: Uses Streamlit's native page system with separate modules for different functionalities
- **Interactive Visualizations**: Plotly for charts and graphs, Folium for maps
- **Real-time Updates**: Session state management for data persistence across pages

### Backend Architecture
- **Language**: Python 3.x
- **Machine Learning**: scikit-learn with multiple algorithm support (Random Forest, XGBoost, SVM, Neural Networks)
- **Data Processing**: pandas and numpy for data manipulation
- **Simulation Engine**: Custom cellular automata and physics-based fire spread models

### Data Storage Solutions
- **Local Files**: CSV file support for datasets
- **Remote Data**: UCI ML Repository integration for forest fire datasets
- **Model Persistence**: joblib for saving and loading trained models
- **Session Storage**: Streamlit session state for temporary data storage

## Key Components

### 1. Main Application (`app.py`)
- Entry point with navigation and system overview
- Custom CSS styling for improved UI
- Sidebar navigation with system information

### 2. Data Overview (`pages/1_Data_Overview.py`)
- Interactive data exploration and filtering
- Statistical summaries and visualizations
- Correlation analysis and seasonal patterns

### 3. Model Training (`pages/2_Model_Training.py`)
- Support for both regression (burned area prediction) and classification (risk level)
- Multiple ML algorithms with automatic hyperparameter tuning
- Feature selection and preprocessing options
- Model comparison and evaluation metrics

### 4. Fire Prediction (`pages/3_Fire_Prediction.py`)
- Single location and batch prediction capabilities
- Weather API integration for real-time conditions
- Risk mapping with interactive visualizations
- Model ensemble predictions

### 5. Fire Simulation (`pages/4_Fire_Simulation.py`)
- Cellular automata-based fire spread simulation
- Physics-based fire behavior modeling
- Interactive parameter adjustment
- Animation and statistical analysis of spread patterns

### 6. Model Performance (`pages/5_Model_Performance.py`)
- Comprehensive model evaluation metrics
- Cross-validation analysis
- Learning curves and diagnostic plots
- Performance comparison tools

### 7. Utility Modules
- **Data Loader**: Handles data ingestion from multiple sources
- **Models**: ML model training and evaluation functions
- **Preprocessing**: Feature engineering and data cleaning
- **Simulation**: Fire spread simulation engines
- **Visualization**: Chart and map creation utilities
- **Weather API**: Real-time weather data integration

## Data Flow

1. **Data Ingestion**: Data is loaded from local files or downloaded from UCI ML Repository
2. **Preprocessing**: Features are engineered, data is cleaned and prepared for modeling
3. **Model Training**: Multiple ML models are trained with cross-validation
4. **Prediction**: Trained models make predictions on new data with weather integration
5. **Simulation**: Fire spread is simulated using environmental parameters
6. **Visualization**: Results are displayed through interactive charts and maps

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **plotly**: Interactive visualizations
- **folium**: Map visualizations

### Machine Learning
- **xgboost**: Gradient boosting framework
- **joblib**: Model serialization

### Weather Integration
- **requests**: HTTP client for API calls
- OpenWeatherMap API (optional)
- WeatherAPI (optional)

### Simulation
- **scipy**: Scientific computing for fire spread models
- **matplotlib**: Additional plotting capabilities

## Deployment Strategy

### Local Development
- Direct execution with `streamlit run app.py`
- Environment variables for API keys
- Local data file support

### Production Deployment
- Containerized deployment with Docker
- Environment variable configuration for API keys
- Cloud storage integration for datasets
- Scalable compute resources for model training

### Configuration
- API keys stored as environment variables
- Model artifacts saved in `models/` directory
- Data files in `data/` directory structure

## Changelog

- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.