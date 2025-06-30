"""
Database Management for Forest Fire Prediction System

This module handles all database operations including schema creation,
data storage, and retrieval for the forest fire prediction system.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import sessionmaker, declarative_base
import streamlit as st

# Database configuration
DATABASE_URL = "sqlite:///forestfire.db"
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

Base = declarative_base()

def clean_json(data):
    """Recursively convert any object to JSON-serializable types."""
    import numpy as np

    if isinstance(data, dict):
        return {str(k): clean_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json(v) for v in data]
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.ndarray,)):
        return clean_json(data.tolist())
    elif isinstance(data, (np.bool_)):
        return bool(data)
    else:
        try:
            # Final fallback
            import json
            json.dumps(data)
            return data
        except:
            return str(data)

class FireData(Base):
    """Table for storing historical fire data."""
    __tablename__ = 'fire_data'
    
    id = Column(Integer, primary_key=True)
    x_coord = Column(Float, nullable=False)
    y_coord = Column(Float, nullable=False)
    month = Column(Integer, nullable=False)
    day = Column(String(10))
    ffmc = Column(Float)
    dmc = Column(Float)
    dc = Column(Float)
    isi = Column(Float)
    temp = Column(Float)
    rh = Column(Float)
    wind = Column(Float)
    rain = Column(Float)
    area = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    data_source = Column(String(50))

class ModelResults(Base):
    """Table for storing trained model results and metrics."""
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    problem_type = Column(String(20), nullable=False)  # 'regression' or 'classification'
    training_date = Column(DateTime, default=datetime.now(timezone.utc))
    feature_names = Column(JSON)
    metrics = Column(JSON)
    hyperparameters = Column(JSON)
    model_path = Column(String(255))
    dataset_size = Column(Integer)
    test_score = Column(Float)
    cross_val_score = Column(Float)
    notes = Column(Text)

class Predictions(Base):
    """Table for storing fire risk predictions."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    prediction_date = Column(DateTime, default=datetime.now(timezone.utc))
    model_name = Column(String(100), nullable=False)
    x_coord = Column(Float, nullable=False)
    y_coord = Column(Float, nullable=False)
    weather_conditions = Column(JSON)
    predicted_value = Column(Float, nullable=False)
    prediction_type = Column(String(20))  # 'area' or 'risk_class'
    confidence_score = Column(Float)
    risk_level = Column(String(20))
    user_id = Column(String(50))
    batch_id = Column(String(50))

class Simulations(Base):
    """Table for storing fire spread simulation results."""
    __tablename__ = 'simulations'
    
    id = Column(Integer, primary_key=True)
    simulation_date = Column(DateTime, default=datetime.now(timezone.utc))
    simulation_type = Column(String(50), nullable=False)
    grid_size = Column(Integer, nullable=False)
    time_steps = Column(Integer, nullable=False)
    initial_conditions = Column(JSON)
    environmental_params = Column(JSON)
    final_statistics = Column(JSON)
    total_burned_area = Column(Float)
    fire_duration = Column(Integer)
    max_spread_rate = Column(Float)
    simulation_data_path = Column(String(255))
    user_id = Column(String(50))

class WeatherData(Base):
    """Table for storing weather observations."""
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True)
    observation_date = Column(DateTime, nullable=False)
    x_coord = Column(Float, nullable=False)
    y_coord = Column(Float, nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    precipitation = Column(Float)
    pressure = Column(Float)
    data_source = Column(String(50))
    is_forecast = Column(Boolean, default=False)

def init_database():
    """Initialize database tables."""
    if not engine:
        st.error("Database connection not available")
        return False
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        st.success("Database tables initialized successfully")
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return False

def get_session():
    """Get database session."""
    if not engine:
        return None
    
    Session = sessionmaker(bind=engine)
    return Session()

def store_fire_data(data_df, source="manual"):
    """
    Store fire data in the database.
    
    Args:
        data_df (pd.DataFrame): Fire data to store
        source (str): Data source identifier
        
    Returns:
        bool: Success status
    """
    if not engine:
        st.warning("Database not available")
        return False
    
    session = get_session()
    if not session:
        return False
    
    try:
        # Convert DataFrame to database records
        records = []
        for _, row in data_df.iterrows():
            record = FireData(
                x_coord=float(row.get('X', 0)),
                y_coord=float(row.get('Y', 0)),
                month=int(row.get('month', 1)),
                day=str(row.get('day', 'unknown')),
                ffmc=float(row.get('FFMC', 0)) if pd.notna(row.get('FFMC')) else None,
                dmc=float(row.get('DMC', 0)) if pd.notna(row.get('DMC')) else None,
                dc=float(row.get('DC', 0)) if pd.notna(row.get('DC')) else None,
                isi=float(row.get('ISI', 0)) if pd.notna(row.get('ISI')) else None,
                temp=float(row.get('temp', 0)) if pd.notna(row.get('temp')) else None,
                rh=float(row.get('RH', 0)) if pd.notna(row.get('RH')) else None,
                wind=float(row.get('wind', 0)) if pd.notna(row.get('wind')) else None,
                rain=float(row.get('rain', 0)) if pd.notna(row.get('rain')) else None,
                area=float(row.get('area', 0)),
                data_source=source
            )
            records.append(record)
        
        # Bulk insert
        session.add_all(records)
        session.commit()
        
        st.success(f"Stored {len(records)} fire data records in database")
        return True
        
    except Exception as e:
        session.rollback()
        st.error(f"Error storing fire data: {str(e)}")
        return False
    finally:
        session.close()

def store_model_results(model_name, problem_type, metrics, feature_names, 
                       hyperparameters=None, dataset_size=None, notes=None):
    """
    Store model training results in the database.
    
    Args:
        model_name (str): Name of the model
        problem_type (str): 'regression' or 'classification'
        metrics (dict): Model performance metrics
        feature_names (list): List of feature names
        hyperparameters (dict): Model hyperparameters
        dataset_size (int): Size of training dataset
        notes (str): Additional notes
        
    Returns:
        bool: Success status
    """
    if not engine:
        return False
    
    session = get_session()
    if not session:
        return False
    
    try:
        # Determine primary metric
        if problem_type == 'regression':
            test_score = metrics.get('R² Score', 0)
        else:
            test_score = metrics.get('Accuracy', 0)
        metrics = clean_json(metrics)
        feature_names = clean_json(feature_names)
        hyperparameters = clean_json(hyperparameters or {})

        record = ModelResults(
            model_name=model_name,
            problem_type=problem_type,
            feature_names=feature_names,
            metrics=metrics,
            hyperparameters=hyperparameters or {},
            dataset_size=dataset_size,
            test_score=test_score,
            notes=notes
        )
        
        session.add(record)
        session.commit()
        
        return True
        
    except Exception as e:
        session.rollback()
        st.error(f"Error storing model results: {str(e)}")
        return False
    finally:
        session.close()

def store_prediction(model_name, x_coord, y_coord, weather_conditions, 
                    predicted_value, prediction_type, confidence_score=None,
                    risk_level=None, user_id=None, batch_id=None):
    """
    Store prediction results in the database.
    
    Args:
        model_name (str): Name of the model used
        x_coord (float): X coordinate
        y_coord (float): Y coordinate
        weather_conditions (dict): Weather input data
        predicted_value (float): Predicted value
        prediction_type (str): Type of prediction
        confidence_score (float): Confidence score
        risk_level (str): Risk level category
        user_id (str): User identifier
        batch_id (str): Batch identifier for bulk predictions
        
    Returns:
        bool: Success status
    """
    if not engine:
        return False
    
    session = get_session()
    if not session:
        return False
    
    try:
        weather_conditions = clean_json(weather_conditions)

        record = Predictions(
            model_name=model_name,
            x_coord=x_coord,
            y_coord=y_coord,
            weather_conditions=weather_conditions,
            predicted_value=predicted_value,
            prediction_type=prediction_type,
            confidence_score=confidence_score,
            risk_level=risk_level,
            user_id=user_id,
            batch_id=batch_id
        )
        
        session.add(record)
        session.commit()
        
        return True
        
    except Exception as e:
        session.rollback()
        st.error(f"Error storing prediction: {str(e)}")
        return False
    finally:
        session.close()

def store_simulation_results(simulation_type, grid_size, time_steps, 
                           initial_conditions, environmental_params, 
                           final_statistics, user_id=None):
    """
    Store simulation results in the database.
    
    Args:
        simulation_type (str): Type of simulation
        grid_size (int): Grid size
        time_steps (int): Number of time steps
        initial_conditions (dict): Initial simulation conditions
        environmental_params (dict): Environmental parameters
        final_statistics (dict): Final simulation statistics
        user_id (str): User identifier
        
    Returns:
        bool: Success status
    """
    if not engine:
        return False
    
    session = get_session()
    if not session:
        return False
    
    try:
        record = Simulations(
            simulation_type=simulation_type,
            grid_size=grid_size,
            time_steps=time_steps,
            initial_conditions=initial_conditions,
            environmental_params=environmental_params,
            final_statistics=final_statistics,
            total_burned_area=final_statistics.get('total_burned', 0),
            fire_duration=final_statistics.get('fire_duration', 0),
            max_spread_rate=final_statistics.get('max_spread_rate', 0),
            user_id=user_id
        )
        
        session.add(record)
        session.commit()
        
        return True
        
    except Exception as e:
        session.rollback()
        st.error(f"Error storing simulation: {str(e)}")
        return False
    finally:
        session.close()

def get_fire_data(limit=None, location_filter=None, date_filter=None):
    """
    Retrieve fire data from the database.
    
    Args:
        limit (int): Maximum number of records
        location_filter (dict): Location filtering parameters
        date_filter (dict): Date filtering parameters
        
    Returns:
        pd.DataFrame: Fire data
    """
    if not engine:
        return pd.DataFrame()
    
    session = get_session()
    if not session:
        return pd.DataFrame()
    
    try:
        query = session.query(FireData)
        
        # Apply filters
        if location_filter:
            if 'x_min' in location_filter:
                query = query.filter(FireData.x_coord >= location_filter['x_min'])
            if 'x_max' in location_filter:
                query = query.filter(FireData.x_coord <= location_filter['x_max'])
            if 'y_min' in location_filter:
                query = query.filter(FireData.y_coord >= location_filter['y_min'])
            if 'y_max' in location_filter:
                query = query.filter(FireData.y_coord <= location_filter['y_max'])
        
        if date_filter:
            if 'start_date' in date_filter:
                query = query.filter(FireData.created_at >= date_filter['start_date'])
            if 'end_date' in date_filter:
                query = query.filter(FireData.created_at <= date_filter['end_date'])
        
        if limit:
            query = query.limit(limit)
        
        # Execute query and convert to DataFrame
        results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for record in results:
            data.append({
                'X': record.x_coord,
                'Y': record.y_coord,
                'month': record.month,
                'day': record.day,
                'FFMC': record.ffmc,
                'DMC': record.dmc,
                'DC': record.dc,
                'ISI': record.isi,
                'temp': record.temp,
                'RH': record.rh,
                'wind': record.wind,
                'rain': record.rain,
                'area': record.area,
                'data_source': record.data_source,
                'created_at': record.created_at
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error retrieving fire data: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def get_model_history():
    """
    Get model training history.
    
    Returns:
        pd.DataFrame: Model history
    """
    if not engine:
        return pd.DataFrame()
    
    session = get_session()
    if not session:
        return pd.DataFrame()
    
    try:
        query = session.query(ModelResults).order_by(ModelResults.training_date.desc())
        results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for record in results:
            data.append({
                'Model': record.model_name,
                'Type': record.problem_type,
                'Training Date': record.training_date,
                'Test Score': record.test_score,
                'Dataset Size': record.dataset_size,
                'Features': len(record.feature_names) if record.feature_names else 0,
                'Notes': record.notes
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error retrieving model history: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def get_prediction_history(limit=100):
    """
    Get prediction history.
    
    Args:
        limit (int): Maximum number of records
        
    Returns:
        pd.DataFrame: Prediction history
    """
    if not engine:
        return pd.DataFrame()
    
    session = get_session()
    if not session:
        return pd.DataFrame()
    
    try:
        query = session.query(Predictions).order_by(
            Predictions.prediction_date.desc()
        ).limit(limit)
        
        results = query.all()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for record in results:
            data.append({
                'Date': record.prediction_date,
                'Model': record.model_name,
                'Location': f"({record.x_coord}, {record.y_coord})",
                'Predicted Value': record.predicted_value,
                'Risk Level': record.risk_level,
                'Confidence': record.confidence_score,
                'Type': record.prediction_type
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error retrieving prediction history: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def get_database_stats():
    """
    Get database statistics.
    
    Returns:
        dict: Database statistics
    """
    if not engine:
        return {}
    
    session = get_session()
    if not session:
        return {}
    
    try:
        stats = {}
        
        # Count records in each table
        stats['fire_data_records'] = session.query(FireData).count()
        stats['model_results'] = session.query(ModelResults).count()
        stats['predictions'] = session.query(Predictions).count()
        stats['simulations'] = session.query(Simulations).count()
        stats['weather_records'] = session.query(WeatherData).count()
        
        # Get date ranges
        if stats['fire_data_records'] > 0:
            earliest = session.query(FireData.created_at).order_by(FireData.created_at.asc()).first()[0]
            latest = session.query(FireData.created_at).order_by(FireData.created_at.desc()).first()[0]
            stats['data_date_range'] = f"{earliest.date()} to {latest.date()}"
        
        return stats
        
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
        return {}
    finally:
        session.close()

def backup_database_to_csv(output_dir="data/backup"):
    """
    Backup database tables to CSV files.
    
    Args:
        output_dir (str): Output directory for backup files
        
    Returns:
        bool: Success status
    """
    if not engine:
        return False
    
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Backup each table
        tables = {
            'fire_data': get_fire_data(),
            'model_history': get_model_history(),
            'prediction_history': get_prediction_history(limit=None)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for table_name, df in tables.items():
            if not df.empty:
                filename = f"{output_dir}/{table_name}_backup_{timestamp}.csv"
                df.to_csv(filename, index=False)
                st.info(f"Backed up {table_name} to {filename}")
        
        return True
        
    except Exception as e:
        st.error(f"Error backing up database: {str(e)}")
        return False

def clear_old_data(days_to_keep=30):
    """
    Clear old data from the database.
    
    Args:
        days_to_keep (int): Number of days of data to keep
        
    Returns:
        bool: Success status
    """
    if not engine:
        return False
    
    session = get_session()
    if not session:
        return False
    
    try:
        from datetime import timedelta
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Delete old predictions
        deleted_predictions = session.query(Predictions).filter(
            Predictions.prediction_date < cutoff_date
        ).delete()
        
        # Delete old simulations
        deleted_simulations = session.query(Simulations).filter(
            Simulations.simulation_date < cutoff_date
        ).delete()
        
        session.commit()
        
        st.info(f"Cleaned up {deleted_predictions} old predictions and {deleted_simulations} old simulations")
        return True
        
    except Exception as e:
        session.rollback()
        st.error(f"Error cleaning old data: {str(e)}")
        return False
    finally:
        session.close()
