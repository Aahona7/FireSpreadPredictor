"""
Visualization Utilities for Forest Fire Prediction

This module contains functions for creating interactive visualizations,
maps, charts, and plots for the forest fire prediction system.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import streamlit as st

def create_overview_map(data, lat_col='Y', lon_col='X', area_col='area'):
    """
    Create an overview map showing fire locations and severity.
    
    Args:
        data (pd.DataFrame): Forest fire dataset
        lat_col (str): Column name for latitude/Y coordinate
        lon_col (str): Column name for longitude/X coordinate  
        area_col (str): Column name for burned area
        
    Returns:
        folium.Map: Interactive map object or None if coordinates missing
    """
    
    if lat_col not in data.columns or lon_col not in data.columns:
        return None

    # Filter out coordinates likely over ocean or invalid range
    data = data[(data[lat_col] > 4) & (data[lat_col] < 45) & (data[lon_col] > -10) & (data[lon_col] < 40)]

    # Calculate map center
    center_lat = data[lat_col].mean()
    center_lon = data[lon_col].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add fire markers
    for idx, row in data.iterrows():
        # Approximate grid-to-latlon conversion for Portugal
        lat = 41.8 + (row[lat_col] - 5) * 0.01
        lon = -6.8 + (row[lon_col] - 5) * 0.01

        area = row.get(area_col, 0)

        # Determine color and size
        if area == 0:
            color = 'green'
            radius = 3
            popup_text = f"No fire detected<br>Location: ({lat}, {lon})"
        elif area <= 1:
            color = 'yellow'
            radius = 5
            popup_text = f"Small fire: {area:.2f} ha<br>Location: ({lat}, {lon})"
        elif area <= 10:
            color = 'orange'
            radius = 8
            popup_text = f"Medium fire: {area:.2f} ha<br>Location: ({lat}, {lon})"
        else:
            color = 'red'
            radius = 12
            popup_text = f"Large fire: {area:.2f} ha<br>Location: ({lat}, {lon})"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            tooltip=popup_text,
            color=color,          # Outline color = fill
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)


    # Improved legend
    legend_html = '''
    <div id="map-legend" style="
        position: fixed;
        bottom: 50px; left: 50px;
        width: 200px;
        background-color: white;
        border: 2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        cursor: move;
        border-radius: 8px;">
        
        <div id="legend-header" style="display: flex; justify-content: space-between; align-items: center;">
            <b>🔥 Fire Severity</b>
            <div>
                <button id="toggle-btn" style="border:none; background:none; cursor:pointer; font-size:16px;">🔽</button>
                <button id="close-btn" style="border:none; background:none; cursor:pointer; font-size:16px;">✖️</button>
            </div>
        </div>
        
        <div id="legend-body" style="margin-top:8px;">
            <span style="display:inline-block; width:12px; height:12px; background-color:green; border-radius:50%; margin-right:5px;"></span>No Fire<br>
            <span style="display:inline-block; width:12px; height:12px; background-color:yellow; border-radius:50%; margin-right:5px;"></span>Small (≤1 ha)<br>
            <span style="display:inline-block; width:12px; height:12px; background-color:orange; border-radius:50%; margin-right:5px;"></span>Medium (1–10 ha)<br>
            <span style="display:inline-block; width:12px; height:12px; background-color:red; border-radius:50%; margin-right:5px;"></span>Large (>10 ha)
        </div>
    </div>

    <!-- Restore Button -->
    <button id="restore-legend" style="
        position: fixed;
        bottom: 50px; left: 50px;
        z-index:9999;
        padding: 6px 12px;
        font-size: 14px;
        background-color: white;
        border: 1px solid #555;
        border-radius: 6px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        display: none;
        cursor: pointer;">
        🗺 Show Legend
    </button>

    <script>
        const legend = document.getElementById('map-legend');
        const header = document.getElementById('legend-header');
        const closeBtn = document.getElementById('close-btn');
        const toggleBtn = document.getElementById('toggle-btn');
        const body = document.getElementById('legend-body');
        const restoreBtn = document.getElementById('restore-legend');

        let offsetX, offsetY, isDragging = false;

        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            offsetX = e.clientX - legend.getBoundingClientRect().left;
            offsetY = e.clientY - legend.getBoundingClientRect().top;
        });

        document.addEventListener('mouseup', () => isDragging = false);

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                legend.style.left = (e.clientX - offsetX) + 'px';
                legend.style.top = (e.clientY - offsetY) + 'px';
                legend.style.bottom = 'auto';
            }
        });

        closeBtn.addEventListener('click', () => {
            legend.style.display = 'none';
            restoreBtn.style.display = 'block';
        });

        restoreBtn.addEventListener('click', () => {
            legend.style.display = 'block';
            restoreBtn.style.display = 'none';
        });

        toggleBtn.addEventListener('click', () => {
            if (body.style.display === 'none') {
                body.style.display = 'block';
                toggleBtn.textContent = '🔽';
            } else {
                body.style.display = 'none';
                toggleBtn.textContent = '🔼';
            }
        });
    </script>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def create_correlation_heatmap(data):
    """
    Create a correlation heatmap for numeric variables.
    
    Args:
        data (pd.DataFrame): Dataset with numeric variables
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        title_x=0.5,
        height=600,
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return fig

def create_seasonal_analysis(data, weather_vars):
    """
    Create seasonal analysis plots for weather variables.
    
    Args:
        data (pd.DataFrame): Dataset with month column and weather variables
        weather_vars (list): List of weather variable names
        
    Returns:
        plotly.graph_objects.Figure: Seasonal analysis plot
    """
    
    if 'month' not in data.columns or not weather_vars:
        return None
    
    # Calculate monthly averages
    monthly_stats = data.groupby('month')[weather_vars].mean()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[var.upper() for var in weather_vars[:4]],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, var in enumerate(weather_vars[:4]):
        if i < len(positions):
            row, col = positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_stats.index,
                    y=monthly_stats[var],
                    mode='lines+markers',
                    name=var.upper(),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="Seasonal Weather Patterns",
        height=500
    )
    
    # Update x-axes
    for i in range(1, 5):
        fig.update_xaxes(
            title_text="Month",
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )
    
    return fig

def create_risk_map(data, risk_column='risk_level'):
    """
    Create a risk assessment map using folium.
    
    Args:
        data (pd.DataFrame): Dataset with coordinates and risk levels
        risk_column (str): Column name for risk levels
        
    Returns:
        folium.Map: Risk map
    """
    
    if 'X' not in data.columns or 'Y' not in data.columns:
        return None
    
    # Calculate map center
    center_lat = data['Y'].mean()
    center_lon = data['X'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Risk color mapping
    risk_colors = {
        'Low': 'green',
        'Medium': 'yellow', 
        'High': 'red',
        'Very High': 'darkred'
    }
    
    # Add risk points
    for idx, row in data.iterrows():
        lat, lon = row['Y'], row['X']
        risk = row.get(risk_column, 'Unknown')
        
        color = risk_colors.get(risk, 'gray')
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"Risk Level: {risk}<br>Location: ({lat}, {lon})",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_prediction_dashboard(predictions, confidence_levels=None):
    """
    Create a comprehensive prediction dashboard.
    
    Args:
        predictions (dict): Dictionary of model predictions
        confidence_levels (dict): Dictionary of confidence levels (optional)
        
    Returns:
        plotly.graph_objects.Figure: Dashboard figure
    """
    
    # Create subplot dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Predictions', 'Prediction Confidence', 
                       'Risk Distribution', 'Model Agreement'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Model predictions bar chart
    model_names = list(predictions.keys())
    pred_values = list(predictions.values())
    
    fig.add_trace(
        go.Bar(x=model_names, y=pred_values, name='Predictions', showlegend=False),
        row=1, col=1
    )
    
    # Confidence levels (if provided)
    if confidence_levels:
        conf_values = [confidence_levels.get(model, 0) for model in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=conf_values, name='Confidence', showlegend=False),
            row=1, col=2
        )
    
    # Risk distribution pie chart
    if pred_values:
        risk_categories = []
        for pred in pred_values:
            if pred < 1:
                risk_categories.append('Low')
            elif pred < 10:
                risk_categories.append('Medium')
            else:
                risk_categories.append('High')
        
        risk_counts = pd.Series(risk_categories).value_counts()
        
        fig.add_trace(
            go.Pie(labels=risk_counts.index, values=risk_counts.values, showlegend=False),
            row=2, col=1
        )
    
    # Model agreement scatter
    if len(pred_values) > 1:
        mean_pred = np.mean(pred_values)
        deviations = [abs(pred - mean_pred) for pred in pred_values]
        
        fig.add_trace(
            go.Scatter(
                x=model_names, y=deviations, 
                mode='markers+lines', name='Deviation from Mean',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Prediction Dashboard",
        height=600
    )
    
    return fig

def create_fire_progression_animation(simulation_results):
    """
    Create an animated visualization of fire progression.
    
    Args:
        simulation_results (list): List of simulation grids over time
        
    Returns:
        plotly.graph_objects.Figure: Animated fire progression
    """
    
    if not simulation_results:
        return None
    
    # Create frames for animation
    frames = []
    
    for i, grid in enumerate(simulation_results):
        frame = go.Frame(
            data=[go.Heatmap(
                z=grid,
                colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
                showscale=True,
                colorbar=dict(
                    title="Fire State",
                    tickvals=[0, 1, 2],
                    ticktext=["Unburned", "Burning", "Burned"]
                )
            )],
            name=f"Time Step {i}"
        )
        frames.append(frame)
    
    # Create initial figure
    fig = go.Figure(
        data=[go.Heatmap(
            z=simulation_results[0],
            colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
            showscale=True
        )],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title="Fire Spread Simulation",
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time Step:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f"Time Step {i}"],
                             {"frame": {"duration": 300, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 300}}],
                    "label": f"{i}",
                    "method": "animate"
                }
                for i in range(len(simulation_results))
            ]
        }]
    )
    
    return fig

def create_simulation_animation(results, wind_speed, wind_direction, show_vectors=True):
    """
    Create animation for fire spread simulation with wind vectors.
    
    Args:
        results (list): List of simulation grids
        wind_speed (float): Wind speed
        wind_direction (float): Wind direction in degrees
        show_vectors (bool): Whether to show wind vectors
        
    Returns:
        plotly.graph_objects.Figure: Simulation animation
    """
    
    if not results:
        return None
    
    grid_size = len(results[0])
    
    # Create base figure with final state
    fig = px.imshow(
        results[-1],
        color_continuous_scale=[[0, 'darkgreen'], [0.33, 'green'], [0.66, 'orange'], [1, 'red']],
        title=f"Fire Spread Simulation (Wind: {wind_speed:.1f} km/h at {wind_direction:.0f}°)"
    )
    
    # Add wind vectors if requested
    if show_vectors and wind_speed > 0:
        # Calculate wind vector components
        wind_rad = np.radians(wind_direction)
        wind_u = wind_speed * np.sin(wind_rad) * 0.1  # Scale for visualization
        wind_v = wind_speed * np.cos(wind_rad) * 0.1
        
        # Add wind vectors at several points
        vector_spacing = max(1, grid_size // 10)
        for i in range(0, grid_size, vector_spacing):
            for j in range(0, grid_size, vector_spacing):
                fig.add_annotation(
                    x=j, y=i,
                    ax=j + wind_u, ay=i + wind_v,
                    arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor="blue", opacity=0.7
                )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate"
    )
    
    # Add colorbar labels
    fig.update_coloraxes(
        colorbar=dict(
            title="Fire State",
            tickvals=[0, 0.33, 0.66, 1],
            ticktext=["Unburned", "Fuel", "Burning", "Burned"]
        )
    )
    
    return fig

def plot_spread_statistics(simulation_results):
    """
    Plot statistics from fire spread simulation.
    
    Args:
        simulation_results (list): List of simulation grids
        
    Returns:
        plotly.graph_objects.Figure: Statistics plot
    """
    
    if not simulation_results:
        return None
    
    # Calculate statistics over time
    time_steps = list(range(len(simulation_results)))
    unburned_count = []
    burning_count = []
    burned_count = []
    
    for grid in simulation_results:
        grid_array = np.array(grid)
        unburned_count.append(np.sum(grid_array == 0))
        burning_count.append(np.sum(grid_array == 1))
        burned_count.append(np.sum(grid_array == 2))
    
    # Create stacked area chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=unburned_count,
        mode='lines', stackgroup='one',
        name='Unburned', fillcolor='green'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=burning_count,
        mode='lines', stackgroup='one',
        name='Burning', fillcolor='orange'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps, y=burned_count,
        mode='lines', stackgroup='one',
        name='Burned', fillcolor='red'
    ))
    
    fig.update_layout(
        title="Fire Progression Statistics",
        xaxis_title="Time Step",
        yaxis_title="Number of Cells",
        height=400
    )
    
    return fig

def create_weather_dashboard(weather_data):
    """
    Create a weather conditions dashboard.
    
    Args:
        weather_data (dict): Current weather data
        
    Returns:
        plotly.graph_objects.Figure: Weather dashboard
    """
    
    # Create gauge charts for key weather variables
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'Fire Danger')
    )
    
    # Temperature gauge
    temp = weather_data.get('temperature', 20)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=temp,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Temperature (°C)"},
            gauge={
                'axis': {'range': [-10, 50]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [-10, 10], 'color': "lightblue"},
                    {'range': [10, 25], 'color': "lightgreen"},
                    {'range': [25, 35], 'color': "yellow"},
                    {'range': [35, 50], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 35
                }
            }
        ),
        row=1, col=1
    )
    
    # Humidity gauge
    humidity = weather_data.get('humidity', 50)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=humidity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Humidity (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ),
        row=1, col=2
    )
    
    # Wind speed gauge
    wind = weather_data.get('wind_speed', 10)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=wind,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Wind Speed (km/h)"},
            gauge={
                'axis': {'range': [0, 60]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 15], 'color': "lightgreen"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 60], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ),
        row=2, col=1
    )
    
    # Fire danger index (calculated)
    fire_danger = calculate_simple_fire_danger(weather_data)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=fire_danger,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fire Danger Index"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 3], 'color': "green"},
                    {'range': [3, 6], 'color': "yellow"},
                    {'range': [6, 8], 'color': "orange"},
                    {'range': [8, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title="Current Weather Conditions")
    
    return fig

def calculate_simple_fire_danger(weather_data):
    """
    Calculate a simple fire danger index from weather data.
    
    Args:
        weather_data (dict): Weather data dictionary
        
    Returns:
        float: Fire danger index (0-10)
    """
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    wind = weather_data.get('wind_speed', 10)
    rain = weather_data.get('precipitation', 0)
    
    # Simple fire danger calculation
    temp_factor = max(0, (temp - 10) / 30)  # 0-1 scale
    humidity_factor = max(0, (100 - humidity) / 80)  # 0-1 scale, inverted
    wind_factor = min(1, wind / 40)  # 0-1 scale
    rain_factor = max(0, 1 - rain / 10)  # 0-1 scale, inverted
    
    # Weighted combination
    danger_index = (temp_factor * 2.5 + humidity_factor * 3 + wind_factor * 2 + rain_factor * 2.5) / 2.5
    
    return min(10, danger_index * 10)

def create_feature_importance_plot(importance_data, title="Feature Importance"):
    """
    Create a feature importance plot.
    
    Args:
        importance_data (dict or pd.DataFrame): Feature importance data
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Feature importance plot
    """
    
    if isinstance(importance_data, dict):
        features = list(importance_data.keys())
        importance = list(importance_data.values())
    else:
        features = importance_data.index.tolist()
        importance = importance_data.values.tolist()
    
    # Sort by importance
    sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_data)
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title=title,
        labels={'x': 'Importance', 'y': 'Features'}
    )
    
    fig.update_layout(
        height=max(400, len(features) * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_time_series_plot(data, x_col, y_col, title="Time Series"):
    """
    Create a time series plot.
    
    Args:
        data (pd.DataFrame): Time series data
        x_col (str): X-axis column (time)
        y_col (str): Y-axis column (values)
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Time series plot
    """
    
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title
    )
    
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        height=400
    )
    
    return fig

def create_distribution_comparison(data1, data2, labels=None, title="Distribution Comparison"):
    """
    Create overlapping distribution plots for comparison.
    
    Args:
        data1 (array-like): First dataset
        data2 (array-like): Second dataset
        labels (list): Labels for the datasets
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Distribution comparison plot
    """
    
    if labels is None:
        labels = ['Dataset 1', 'Dataset 2']
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data1, 
        opacity=0.7,
        name=labels[0],
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=data2,
        opacity=0.7, 
        name=labels[1],
        nbinsx=30
    ))
    
    fig.update_layout(
        title=title,
        barmode='overlay',
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig


