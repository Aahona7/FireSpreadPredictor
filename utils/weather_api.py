"""
Weather API Integration for Forest Fire Prediction

This module handles integration with various weather APIs to fetch
real-time and historical weather data for fire prediction.
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import streamlit as st

def get_weather_data(lat, lon, api_key=None, source="openweathermap"):
    """
    Fetch current weather data from various APIs.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        api_key (str): API key for weather service
        source (str): Weather data source ('openweathermap', 'weatherapi')
        
    Returns:
        dict: Weather data or None if failed
    """
    
    # Use environment variable if no API key provided
    if api_key is None:
        api_key = os.getenv('WEATHER_API_KEY')
    
    if not api_key:
        st.warning("No weather API key provided. Using mock data for demonstration.")
        return generate_mock_weather_data(lat, lon)
    
    try:
        if source == "openweathermap":
            return fetch_openweathermap_data(lat, lon, api_key)
        elif source == "weatherapi":
            return fetch_weatherapi_data(lat, lon, api_key)
        else:
            st.error(f"Unsupported weather data source: {source}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return generate_mock_weather_data(lat, lon)

def fetch_openweathermap_data(lat, lon, api_key):
    """
    Fetch data from OpenWeatherMap API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        api_key (str): OpenWeatherMap API key
        
    Returns:
        dict: Formatted weather data
    """
    
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract relevant weather information
    weather_data = {
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'pressure': data['main']['pressure'],
        'wind_speed': data['wind'].get('speed', 0) * 3.6,  # Convert m/s to km/h
        'wind_direction': data['wind'].get('deg', 0),
        'precipitation': data.get('rain', {}).get('1h', 0),  # mm in last hour
        'cloudiness': data['clouds']['all'],
        'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
        'weather_description': data['weather'][0]['description'],
        'location': f"{data['name']}, {data['sys']['country']}",
        'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
    }
    
    return weather_data

def fetch_weatherapi_data(lat, lon, api_key):
    """
    Fetch data from WeatherAPI.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        api_key (str): WeatherAPI key
        
    Returns:
        dict: Formatted weather data
    """
    
    url = f"http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': f"{lat},{lon}",
        'aqi': 'no'
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract relevant weather information
    current = data['current']
    location = data['location']
    
    weather_data = {
        'temperature': current['temp_c'],
        'humidity': current['humidity'],
        'pressure': current['pressure_mb'],
        'wind_speed': current['wind_kph'],
        'wind_direction': current['wind_degree'],
        'precipitation': current['precip_mm'],
        'cloudiness': current['cloud'],
        'visibility': current['vis_km'],
        'weather_description': current['condition']['text'],
        'location': f"{location['name']}, {location['country']}",
        'timestamp': current['last_updated']
    }
    
    return weather_data

def generate_mock_weather_data(lat, lon):
    """
    Generate realistic mock weather data for demonstration.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Mock weather data
    """
    
    # Generate realistic weather based on location and season
    now = datetime.now()
    season_factor = np.sin(2 * np.pi * now.timetuple().tm_yday / 365.25)
    
    # Base temperature varies with latitude and season
    base_temp = 20 - abs(lat) * 0.5 + season_factor * 15
    
    # Add some randomness
    np.random.seed(int(lat * 1000 + lon * 1000) % 1000)
    
    mock_data = {
        'temperature': base_temp + np.random.normal(0, 5),
        'humidity': max(10, min(90, 50 + np.random.normal(0, 20))),
        'pressure': 1013 + np.random.normal(0, 20),
        'wind_speed': max(0, np.random.exponential(8)),
        'wind_direction': np.random.uniform(0, 360),
        'precipitation': max(0, np.random.exponential(0.5)),
        'cloudiness': np.random.uniform(0, 100),
        'visibility': max(1, np.random.normal(10, 3)),
        'weather_description': 'Partly cloudy',
        'location': f"Location ({lat:.2f}, {lon:.2f})",
        'timestamp': now.isoformat(),
        'is_mock': True
    }
    
    return mock_data

def get_historical_weather_data(lat, lon, start_date, end_date, api_key=None):
    """
    Fetch historical weather data for a location and date range.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (datetime): Start date
        end_date (datetime): End date
        api_key (str): API key
        
    Returns:
        pd.DataFrame: Historical weather data
    """
    
    if api_key is None:
        api_key = os.getenv('WEATHER_API_KEY')
    
    if not api_key:
        st.warning("No API key provided. Generating mock historical data.")
        return generate_mock_historical_data(lat, lon, start_date, end_date)
    
    try:
        # For demonstration, using WeatherAPI historical endpoint
        historical_data = []
        current_date = start_date
        
        while current_date <= end_date:
            url = "http://api.weatherapi.com/v1/history.json"
            params = {
                'key': api_key,
                'q': f"{lat},{lon}",
                'dt': current_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract daily data
            for hour_data in data['forecast']['forecastday'][0]['hour']:
                historical_data.append({
                    'datetime': hour_data['time'],
                    'temperature': hour_data['temp_c'],
                    'humidity': hour_data['humidity'],
                    'pressure': hour_data['pressure_mb'],
                    'wind_speed': hour_data['wind_kph'],
                    'wind_direction': hour_data['wind_degree'],
                    'precipitation': hour_data['precip_mm'],
                    'cloudiness': hour_data['cloud'],
                    'visibility': hour_data['vis_km']
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(historical_data)
    
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return generate_mock_historical_data(lat, lon, start_date, end_date)

def generate_mock_historical_data(lat, lon, start_date, end_date):
    """
    Generate mock historical weather data.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        pd.DataFrame: Mock historical weather data
    """
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducibility
    
    historical_data = []
    
    for date in date_range:
        # Seasonal temperature variation
        day_of_year = date.timetuple().tm_yday
        season_factor = np.sin(2 * np.pi * day_of_year / 365.25)
        base_temp = 20 - abs(lat) * 0.5 + season_factor * 15
        
        # Generate daily weather
        daily_temp = base_temp + np.random.normal(0, 5)
        daily_humidity = max(10, min(90, 50 + np.random.normal(0, 20)))
        daily_wind = max(0, np.random.exponential(8))
        daily_rain = max(0, np.random.exponential(0.5))
        
        historical_data.append({
            'date': date,
            'temperature': daily_temp,
            'humidity': daily_humidity,
            'wind_speed': daily_wind,
            'precipitation': daily_rain,
            'pressure': 1013 + np.random.normal(0, 10),
            'is_mock': True
        })
    
    return pd.DataFrame(historical_data)

def calculate_fire_weather_indices(weather_data):
    """
    Calculate fire weather indices from basic weather data.
    
    Args:
        weather_data (dict): Weather data dictionary
        
    Returns:
        dict: Fire weather indices
    """
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    wind = weather_data.get('wind_speed', 10)
    rain = weather_data.get('precipitation', 0)
    
    # Simplified calculations (real FWI system is more complex)
    
    # Fine Fuel Moisture Code (FFMC) - simplified
    ffmc = 85 + (temp - 20) * 0.5 - (humidity - 50) * 0.3
    ffmc = max(0, min(100, ffmc))
    
    # Duff Moisture Code (DMC) - simplified
    dmc = 30 + (100 - humidity) * 0.3 + temp * 0.2 - rain * 2
    dmc = max(0, dmc)
    
    # Drought Code (DC) - simplified
    dc = 200 + temp * 2 - rain * 10
    dc = max(0, dc)
    
    # Initial Spread Index (ISI) - simplified
    isi = wind * 0.5 + (ffmc - 80) * 0.1
    isi = max(0, isi)
    
    # Build Up Index (BUI)
    bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    
    # Fire Weather Index (FWI)
    fwi = 2.0 * np.log(isi * bui + 1)
    
    return {
        'FFMC': round(ffmc, 2),
        'DMC': round(dmc, 2),
        'DC': round(dc, 2),
        'ISI': round(isi, 2),
        'BUI': round(bui, 2),
        'FWI': round(fwi, 2)
    }

def get_weather_forecast(lat, lon, days=7, api_key=None):
    """
    Get weather forecast for fire prediction.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        days (int): Number of forecast days
        api_key (str): API key
        
    Returns:
        pd.DataFrame: Weather forecast data
    """
    
    if api_key is None:
        api_key = os.getenv('WEATHER_API_KEY')
    
    if not api_key:
        return generate_mock_forecast(lat, lon, days)
    
    try:
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': api_key,
            'q': f"{lat},{lon}",
            'days': min(days, 10),  # API limit
            'aqi': 'no'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        forecast_data = []
        
        for day_data in data['forecast']['forecastday']:
            day_info = day_data['day']
            
            forecast_data.append({
                'date': day_data['date'],
                'max_temp': day_info['maxtemp_c'],
                'min_temp': day_info['mintemp_c'],
                'avg_temp': day_info['avgtemp_c'],
                'max_wind': day_info['maxwind_kph'],
                'avg_humidity': day_info['avghumidity'],
                'precipitation': day_info['totalprecip_mm'],
                'condition': day_info['condition']['text']
            })
        
        return pd.DataFrame(forecast_data)
    
    except Exception as e:
        st.error(f"Error fetching forecast: {str(e)}")
        return generate_mock_forecast(lat, lon, days)

def generate_mock_forecast(lat, lon, days):
    """
    Generate mock weather forecast.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        days (int): Number of forecast days
        
    Returns:
        pd.DataFrame: Mock forecast data
    """
    
    np.random.seed(42)
    
    forecast_data = []
    current_date = datetime.now().date()
    
    for i in range(days):
        forecast_date = current_date + timedelta(days=i)
        
        # Seasonal base temperature
        day_of_year = forecast_date.timetuple().tm_yday
        season_factor = np.sin(2 * np.pi * day_of_year / 365.25)
        base_temp = 20 - abs(lat) * 0.5 + season_factor * 15
        
        # Add random variation
        daily_variation = np.random.normal(0, 3)
        avg_temp = base_temp + daily_variation
        
        forecast_data.append({
            'date': forecast_date,
            'max_temp': avg_temp + np.random.uniform(2, 8),
            'min_temp': avg_temp - np.random.uniform(2, 8),
            'avg_temp': avg_temp,
            'max_wind': max(0, np.random.exponential(12)),
            'avg_humidity': max(10, min(90, 50 + np.random.normal(0, 15))),
            'precipitation': max(0, np.random.exponential(1)),
            'condition': np.random.choice(['Sunny', 'Partly cloudy', 'Cloudy', 'Light rain']),
            'is_mock': True
        })
    
    return pd.DataFrame(forecast_data)

def validate_weather_data(weather_data):
    """
    Validate and clean weather data.
    
    Args:
        weather_data (dict): Weather data dictionary
        
    Returns:
        dict: Validated weather data
    """
    
    validated_data = weather_data.copy()
    
    # Temperature validation
    if 'temperature' in validated_data:
        temp = validated_data['temperature']
        if temp < -50 or temp > 60:
            st.warning(f"Unusual temperature value: {temp}°C")
            validated_data['temperature'] = max(-50, min(60, temp))
    
    # Humidity validation
    if 'humidity' in validated_data:
        humidity = validated_data['humidity']
        if humidity < 0 or humidity > 100:
            st.warning(f"Invalid humidity value: {humidity}%")
            validated_data['humidity'] = max(0, min(100, humidity))
    
    # Wind speed validation
    if 'wind_speed' in validated_data:
        wind = validated_data['wind_speed']
        if wind < 0 or wind > 200:
            st.warning(f"Unusual wind speed: {wind} km/h")
            validated_data['wind_speed'] = max(0, min(200, wind))
    
    # Precipitation validation
    if 'precipitation' in validated_data:
        precip = validated_data['precipitation']
        if precip < 0:
            validated_data['precipitation'] = 0
        elif precip > 200:
            st.warning(f"Unusual precipitation: {precip} mm")
            validated_data['precipitation'] = min(200, precip)
    
    return validated_data

def format_weather_for_prediction(weather_data):
    """
    Format weather data for use in fire prediction models.
    
    Args:
        weather_data (dict): Raw weather data
        
    Returns:
        dict: Formatted data for model input
    """
    
    # Validate data first
    validated_data = validate_weather_data(weather_data)
    
    # Calculate fire weather indices
    fire_indices = calculate_fire_weather_indices(validated_data)
    
    # Format for model input
    model_input = {
        'temp': validated_data.get('temperature', 20),
        'RH': validated_data.get('humidity', 50),
        'wind': validated_data.get('wind_speed', 10),
        'rain': validated_data.get('precipitation', 0),
        'FFMC': fire_indices['FFMC'],
        'DMC': fire_indices['DMC'],
        'DC': fire_indices['DC'],
        'ISI': fire_indices['ISI']
    }
    
    return model_input

def get_weather_alerts(weather_data, fire_danger_threshold=6):
    """
    Generate weather-based fire danger alerts.
    
    Args:
        weather_data (dict): Current weather data
        fire_danger_threshold (float): Threshold for high fire danger
        
    Returns:
        list: List of alert messages
    """
    
    alerts = []
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    wind = weather_data.get('wind_speed', 10)
    rain = weather_data.get('precipitation', 0)
    
    # Calculate fire weather indices
    fire_indices = calculate_fire_weather_indices(weather_data)
    fire_danger = fire_indices['FWI']
    
    # Temperature alerts
    if temp > 35:
        alerts.append("🔥 High temperature alert: Extreme fire weather conditions")
    elif temp > 30:
        alerts.append("⚠️ Elevated temperature: Increased fire risk")
    
    # Humidity alerts
    if humidity < 20:
        alerts.append("🔥 Critical humidity alert: Very dry conditions")
    elif humidity < 30:
        alerts.append("⚠️ Low humidity: Dry conditions increase fire risk")
    
    # Wind alerts
    if wind > 30:
        alerts.append("🔥 High wind alert: Rapid fire spread possible")
    elif wind > 20:
        alerts.append("⚠️ Strong winds: Enhanced fire spread conditions")
    
    # Precipitation alerts
    if rain < 0.1:
        alerts.append("⚠️ No recent precipitation: Dry fuel conditions")
    
    # Fire danger alerts
    if fire_danger > fire_danger_threshold:
        alerts.append("🚨 HIGH FIRE DANGER: Extreme caution advised")
    elif fire_danger > 4:
        alerts.append("⚠️ Moderate fire danger: Exercise caution")
    
    # Combined conditions
    if temp > 30 and humidity < 30 and wind > 15:
        alerts.append("🚨 CRITICAL FIRE WEATHER: All risk factors elevated")
    
    return alerts

