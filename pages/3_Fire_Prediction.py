import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import joblib
import os
from utils.weather_api import get_weather_data
from utils.visualization import create_risk_map, create_prediction_dashboard
from utils.database import store_prediction

st.set_page_config(page_title="Fire Prediction", page_icon="🔥", layout="wide")

st.title("🔥 Forest Fire Risk Prediction")

def clean_json(data):
    import numpy as np
    if isinstance(data, dict):
        return {str(k): clean_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json(i) for i in data]
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.ndarray,)):
        return clean_json(data.tolist())
    else:
        return data

# Load trained models
@st.cache_data
def load_trained_models():
    try:
        if os.path.exists("models/forest_fire_models.pkl"):
            return joblib.load("models/forest_fire_models.pkl")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

model_data = load_trained_models()

if model_data is None:
    st.error("No trained models found. Please train models first in the Model Training page.")
    st.stop()

models = model_data['models']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
categorical_encoders = model_data['categorical_encoders']
problem_type = model_data['problem_type']

st.success("✅ Trained models loaded successfully!")

# Prediction tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "🗺️ Risk Mapping"])

with tab1:
    st.header("Single Location Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather Conditions")
        
        # Weather inputs
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 20.0, 0.1)
        humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 50.0, 0.1)
        wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.1)
        rain = st.slider("Rain (mm/m²)", 0.0, 50.0, 0.0, 0.1)
        
        st.subheader("Fire Weather Indices")
        ffmc = st.slider("FFMC (Fine Fuel Moisture Code)", 0.0, 100.0, 85.0, 0.1)
        dmc = st.slider("DMC (Duff Moisture Code)", 0.0, 300.0, 25.0, 0.1)
        dc = st.slider("DC (Drought Code)", 0.0, 1000.0, 150.0, 0.1)
        isi = st.slider("ISI (Initial Spread Index)", 0.0, 50.0, 5.0, 0.1)
    
    with col2:
        st.subheader("Location & Time")
        
        # Location inputs
        x_coord = st.number_input("X Coordinate", value=5.0, format="%.2f")
        y_coord = st.number_input("Y Coordinate", value=5.0, format="%.2f")
        
        # Time inputs
        month = st.selectbox("Month", list(range(1, 13)), index=6)
        day = st.selectbox("Day of Week", 
                          ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
                          index=0)
        
        # Auto-fill with current weather (if API available)
        if st.button("Get Current Weather"):
            with st.spinner("Fetching weather data..."):
                try:
                    weather_data = get_weather_data(x_coord, y_coord)
                    if weather_data:
                        st.success("Weather data updated!")
                        # Update the session state with weather data
                        st.session_state.temp = weather_data.get('temperature', temp)
                        st.session_state.humidity = weather_data.get('humidity', humidity)
                        st.session_state.wind = weather_data.get('wind_speed', wind)
                        st.session_state.rain = weather_data.get('precipitation', rain)
                        st.rerun()
                    else:
                        st.warning("Could not fetch weather data. Using manual inputs.")
                except Exception as e:
                    st.warning(f"Weather API error: {str(e)}")
    
    # Prepare input data
    input_data = {
        'temp': temp,
        'RH': humidity,
        'wind': wind,
        'rain': rain,
        'FFMC': ffmc,
        'DMC': dmc,
        'DC': dc,
        'ISI': isi,
        'X': x_coord,
        'Y': y_coord,
        'month': month
    }
    
    # Add day encoding if it's a categorical feature
    if 'day' in feature_names:
        if 'day' in categorical_encoders:
            try:
                day_encoded = categorical_encoders['day'].transform([day])[0]
                input_data['day'] = day_encoded
            except:
                input_data['day'] = 0
        else:
            input_data['day'] = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'].index(day)
    
    # Create prediction dataframe
    input_df = pd.DataFrame([input_data])
    
    # Apply feature engineering (same as training)
    from utils.preprocessing import create_features
    input_df_enhanced = create_features(input_df)
    
    # Select only the features used in training
    input_features = input_df_enhanced[feature_names].fillna(0)
    
    # Scale features if scaler was used
    if scaler is not None:
        input_scaled = scaler.transform(input_features)
    else:
        input_scaled = input_features.values
    
    # Make predictions with all models
    st.subheader("Prediction Results")
    
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(input_scaled)[0]
        
        if problem_type == "regression":
            # Convert back from log scale
            actual_area = np.expm1(pred)  # exp(pred) - 1
            predictions[model_name] = actual_area
        else:
            predictions[model_name] = pred
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    model_names = list(predictions.keys())
    
    if problem_type == "regression":
        # Show burned area predictions
        for i, (model_name, pred_area) in enumerate(predictions.items()):
            with [col1, col2, col3][i % 3]:
                risk_level = "Low" if pred_area < 1 else "Medium" if pred_area < 10 else "High"
                risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                
                st.metric(
                    label=f"{model_name}",
                    value=f"{pred_area:.2f} ha",
                    help=f"Predicted burned area"
                )
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
    else:
        # Show risk category predictions
        risk_labels = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
        for i, (model_name, pred_class) in enumerate(predictions.items()):
            with [col1, col2, col3][i % 3]:
                risk_label = risk_labels[int(pred_class)]
                risk_color = "green" if pred_class <= 1 else "orange" if pred_class == 2 else "red"
                
                st.metric(
                    label=f"{model_name}",
                    value=risk_label,
                    help=f"Predicted risk category"
                )
    
    # Prediction confidence and model agreement
    st.subheader("Prediction Analysis")
    
    if problem_type == "regression":
        pred_values = list(predictions.values())
        avg_prediction = np.mean(pred_values)
        std_prediction = np.std(pred_values)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Prediction", f"{avg_prediction:.2f} ha")
            st.metric("Prediction Std Dev", f"{std_prediction:.2f} ha")
        
        with col2:
            # Confidence based on model agreement
            confidence = max(0, 100 - (std_prediction / max(avg_prediction, 0.1)) * 50)
            st.metric("Prediction Confidence", f"{confidence:.1f}%")
            
            # Risk assessment
            if avg_prediction < 1:
                st.success("🟢 Low fire risk")
            elif avg_prediction < 10:
                st.warning("🟡 Medium fire risk")
            else:
                st.error("🔴 High fire risk")
    
    # Feature contribution analysis
    if st.checkbox("Show Feature Contribution Analysis"):
        st.subheader("Feature Contribution")
        
        # For tree-based models, show feature importance
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                
                # Get current input values
                current_values = input_features.iloc[0].to_dict()
                
                # Create contribution dataframe
                contrib_data = []
                for feature, importance in importance_dict.items():
                    value = current_values.get(feature, 0)
                    contrib_data.append({
                        'Feature': feature,
                        'Value': value,
                        'Importance': importance,
                        'Contribution': value * importance
                    })
                
                contrib_df = pd.DataFrame(contrib_data).sort_values('Contribution', ascending=False)
                
                st.write(f"**{model_name} Feature Contributions:**")
                
                fig_contrib = px.bar(
                    contrib_df.head(10),
                    x='Contribution',
                    y='Feature',
                    orientation='h',
                    title=f"Top 10 Feature Contributions - {model_name}"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Store prediction in database
    if st.button("Save Prediction to Database"):
        try:
            # Get best model prediction for storage
            if problem_type == "regression":
                best_model_name = max(predictions.keys(), key=lambda k: predictions[k] if predictions[k] > 0 else 0)
                predicted_value = predictions[best_model_name]
                risk_level = "Low" if predicted_value < 1 else "Medium" if predicted_value < 10 else "High"
            else:
                # For classification, use the most common prediction
                pred_values = list(predictions.values())
                predicted_value = max(set(pred_values), key=pred_values.count)
                risk_labels = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
                risk_level = risk_labels[int(predicted_value)]
            
            # Weather conditions for storage
            weather_conditions = clean_json({
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind,
                'rain': rain,
                'ffmc': ffmc,
                'dmc': dmc,
                'dc': dc,
                'isi': isi
})

            
            # Store prediction
            success = store_prediction(
                model_name=best_model_name,
                x_coord=x_coord,
                y_coord=y_coord,
                weather_conditions=weather_conditions,
                predicted_value=float(predicted_value),
                prediction_type=problem_type,
                confidence_score=confidence if problem_type == "regression" else None,
                risk_level=risk_level,
                user_id="streamlit_user"
            )
            
            if success:
                st.success("Prediction saved to database successfully!")
            else:
                st.error("Failed to save prediction to database")
                
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")

with tab2:
    st.header("Batch Prediction")
    
    st.info("Upload a CSV file with weather and location data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should contain columns: temp, RH, wind, rain, FFMC, DMC, DC, ISI, X, Y, month"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            # Check for required columns
            required_cols = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'X', 'Y', 'month']
            missing_cols = [col for col in required_cols if col not in batch_data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Select model for batch prediction
                selected_model_name = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        try:
                            # Apply feature engineering
                            batch_enhanced = create_features(batch_data.copy())
                            
                            # Select features and handle missing values
                            batch_features = batch_enhanced[feature_names].fillna(0)
                            
                            # Scale features
                            if scaler is not None:
                                batch_scaled = scaler.transform(batch_features)
                            else:
                                batch_scaled = batch_features.values
                            
                            # Make predictions
                            selected_model = models[selected_model_name]
                            batch_predictions = selected_model.predict(batch_scaled)
                            
                            # Add predictions to original data
                            if problem_type == "regression":
                                batch_data['Predicted_Area_ha'] = np.expm1(batch_predictions)
                                batch_data['Risk_Level'] = pd.cut(
                                    batch_data['Predicted_Area_ha'],
                                    bins=[-np.inf, 1, 10, np.inf],
                                    labels=['Low', 'Medium', 'High']
                                )
                            else:
                                risk_labels = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
                                batch_data['Risk_Category'] = [risk_labels[int(pred)] for pred in batch_predictions]
                            
                            st.success("✅ Batch predictions completed!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(batch_data, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if problem_type == "regression":
                                    st.write("**Area Prediction Summary:**")
                                    st.write(f"- Mean predicted area: {batch_data['Predicted_Area_ha'].mean():.2f} ha")
                                    st.write(f"- Max predicted area: {batch_data['Predicted_Area_ha'].max():.2f} ha")
                                    st.write(f"- High risk locations: {(batch_data['Risk_Level'] == 'High').sum()}")
                                else:
                                    st.write("**Risk Category Summary:**")
                                    risk_counts = batch_data['Risk_Category'].value_counts()
                                    for category, count in risk_counts.items():
                                        st.write(f"- {category}: {count}")
                            
                            with col2:
                                # Visualization
                                if problem_type == "regression":
                                    fig_batch = px.histogram(
                                        batch_data,
                                        x='Predicted_Area_ha',
                                        title="Distribution of Predicted Burned Areas",
                                        nbins=30
                                    )
                                else:
                                    fig_batch = px.pie(
                                        values=batch_data['Risk_Category'].value_counts().values,
                                        names=batch_data['Risk_Category'].value_counts().index,
                                        title="Risk Category Distribution"
                                    )
                                
                                st.plotly_chart(fig_batch, use_container_width=True)
                            
                            # Download results
                            csv = batch_data.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name=f"fire_predictions_{selected_model_name}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error in batch prediction: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with tab3:
    st.header("Risk Mapping")
    
    # Grid-based risk mapping
    st.subheader("Generate Risk Map")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Spatial Parameters:**")
        x_min = st.number_input("X Min", value=1.0)
        x_max = st.number_input("X Max", value=9.0)
        y_min = st.number_input("Y Min", value=1.0)
        y_max = st.number_input("Y Max", value=9.0)
        grid_resolution = st.slider("Grid Resolution", 10, 50, 20)
    
    with col2:
        st.write("**Environmental Conditions:**")
        map_temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, key="map_temp")
        map_humidity = st.slider("Humidity (%)", 0.0, 100.0, 45.0, key="map_humidity")
        map_wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0, key="map_wind")
        map_rain = st.slider("Rain (mm/m²)", 0.0, 50.0, 0.0, key="map_rain")
        map_month = st.selectbox("Month", list(range(1, 13)), index=7, key="map_month")
    
    # Fire weather indices for map
    st.write("**Fire Weather Indices:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        map_ffmc = st.number_input("FFMC", value=90.0, key="map_ffmc")
    with col2:
        map_dmc = st.number_input("DMC", value=30.0, key="map_dmc")
    with col3:
        map_dc = st.number_input("DC", value=200.0, key="map_dc")
    with col4:
        map_isi = st.number_input("ISI", value=8.0, key="map_isi")
    
    # Model selection for mapping
    map_model_name = st.selectbox("Select Model for Risk Mapping", list(models.keys()), key="map_model")
    
    if st.button("Generate Risk Map"):
        with st.spinner("Generating risk map..."):
            try:
                # Create grid
                x_coords = np.linspace(x_min, x_max, grid_resolution)
                y_coords = np.linspace(y_min, y_max, grid_resolution)
                
                # Create meshgrid
                X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
                
                # Flatten for prediction
                grid_points = []
                for i in range(len(y_coords)):
                    for j in range(len(x_coords)):
                        grid_points.append({
                            'temp': map_temp,
                            'RH': map_humidity,
                            'wind': map_wind,
                            'rain': map_rain,
                            'FFMC': map_ffmc,
                            'DMC': map_dmc,
                            'DC': map_dc,
                            'ISI': map_isi,
                            'X': X_grid[i, j],
                            'Y': Y_grid[i, j],
                            'month': map_month
                        })
                
                grid_df = pd.DataFrame(grid_points)
                
                # Apply feature engineering
                grid_enhanced = create_features(grid_df)
                grid_features = grid_enhanced[feature_names].fillna(0)
                
                # Scale features
                if scaler is not None:
                    grid_scaled = scaler.transform(grid_features)
                else:
                    grid_scaled = grid_features.values
                
                # Make predictions
                map_model = models[map_model_name]
                grid_predictions = map_model.predict(grid_scaled)
                
                if problem_type == "regression":
                    # Convert from log scale
                    grid_risk = np.expm1(grid_predictions)
                else:
                    grid_risk = grid_predictions
                
                # Reshape for visualization
                risk_grid = grid_risk.reshape(grid_resolution, grid_resolution)
                
                # Create risk map visualization
                fig_risk_map = px.imshow(
                    risk_grid,
                    x=x_coords,
                    y=y_coords,
                    color_continuous_scale='Reds',
                    title=f"Fire Risk Map - {map_model_name}",
                    labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'color': 'Risk Level' if problem_type == 'classification' else 'Predicted Area (ha)'}
                )
                
                st.plotly_chart(fig_risk_map, use_container_width=True)
                
                # Risk statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    if problem_type == "regression":
                        st.metric("Average Risk (ha)", f"{grid_risk.mean():.2f}")
                        st.metric("Maximum Risk (ha)", f"{grid_risk.max():.2f}")
                        high_risk_pct = (grid_risk > 10).mean() * 100
                        st.metric("High Risk Areas (%)", f"{high_risk_pct:.1f}")
                    else:
                        unique_risks, counts = np.unique(grid_risk, return_counts=True)
                        risk_labels = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
                        for risk_val, count in zip(unique_risks, counts):
                            pct = (count / len(grid_risk)) * 100
                            st.metric(f"{risk_labels[int(risk_val)]} (%)", f"{pct:.1f}")
                
                with col2:
                    # Risk distribution histogram
                    fig_risk_dist = px.histogram(
                        x=grid_risk.flatten(),
                        nbins=20,
                        title="Risk Distribution Across Map"
                    )
                    st.plotly_chart(fig_risk_dist, use_container_width=True)
                
                # Interactive risk analysis
                if st.checkbox("Enable Interactive Risk Analysis"):
                    # Allow user to click on map for detailed prediction
                    selected_x = st.slider("Select X Coordinate", float(x_min), float(x_max), float((x_min + x_max) / 2))
                    selected_y = st.slider("Select Y Coordinate", float(y_min), float(y_max), float((y_min + y_max) / 2))
                    
                    # Find nearest grid point
                    x_idx = np.argmin(np.abs(x_coords - selected_x))
                    y_idx = np.argmin(np.abs(y_coords - selected_y))
                    point_risk = risk_grid[y_idx, x_idx]
                    
                    st.write(f"**Risk at ({selected_x:.2f}, {selected_y:.2f}):**")
                    if problem_type == "regression":
                        st.write(f"Predicted burned area: {point_risk:.2f} ha")
                        risk_level = "Low" if point_risk < 1 else "Medium" if point_risk < 10 else "High"
                        st.write(f"Risk level: {risk_level}")
                    else:
                        risk_labels = ["No Fire", "Low Risk", "Medium Risk", "High Risk"]
                        st.write(f"Risk category: {risk_labels[int(point_risk)]}")
                
            except Exception as e:
                st.error(f"Error generating risk map: {str(e)}")

# Weather integration section
st.header("Real-time Weather Integration")

with st.expander("Configure Weather Data Source"):
    st.info("Weather data integration allows automatic fetching of current conditions for predictions.")
    
    weather_api_key = st.text_input(
        "Weather API Key",
        type="password",
        help="Enter your weather API key (optional)"
    )
    
    weather_source = st.selectbox(
        "Weather Data Source",
        ["Manual Input", "OpenWeatherMap", "WeatherAPI"],
        help="Select weather data source"
    )
    
    if st.button("Test Weather API"):
        if weather_api_key:
            with st.spinner("Testing weather API..."):
                try:
                    test_weather = get_weather_data(40.7128, -74.0060, api_key=weather_api_key)
                    if test_weather:
                        st.success("✅ Weather API connection successful!")
                        st.json(test_weather)
                    else:
                        st.error("❌ Weather API test failed")
                except Exception as e:
                    st.error(f"Weather API error: {str(e)}")
        else:
            st.warning("Please enter a weather API key to test the connection.")
