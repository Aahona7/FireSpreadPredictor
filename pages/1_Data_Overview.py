import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_forest_fire_data
from utils.visualization import create_correlation_heatmap, create_seasonal_analysis

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Forest Fire Data Overview")

# Load data
if 'fire_data' not in st.session_state:
    with st.spinner("Loading data..."):
        try:
            st.session_state.fire_data = load_forest_fire_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

data = st.session_state.fire_data

if data is None:
    st.error("No data available. Please check your data source.")
    st.stop()

# Data filtering options
st.sidebar.header("Data Filters")

# Month filter
if 'month' in data.columns:
    months = sorted(data['month'].unique())
    selected_months = st.sidebar.multiselect(
        "Select Months",
        months,
        default=months,
        help="Filter data by month"
    )
    filtered_data = data[data['month'].isin(selected_months)]
else:
    filtered_data = data

# Area threshold filter
if 'area' in data.columns:
    max_area = float(data['area'].max())
    area_threshold = st.sidebar.slider(
        "Maximum Burned Area (ha)",
        0.0,
        max_area,
        max_area,
        help="Filter fires by maximum burned area"
    )
    filtered_data = filtered_data[filtered_data['area'] <= area_threshold]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_data)}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔥 Fire Analysis", "🌤️ Weather Patterns", "📍 Spatial Analysis"])

with tab1:
    st.header("Data Distributions")
    
    # Numeric columns for analysis
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Burned Area Distribution")
        if 'area' in filtered_data.columns:
            fig_area = px.histogram(
                filtered_data,
                x='area',
                nbins=50,
                title="Distribution of Burned Area",
                labels={'area': 'Burned Area (ha)', 'count': 'Frequency'}
            )
            fig_area.update_layout(showlegend=False)
            st.plotly_chart(fig_area, use_container_width=True)
            
            # Area statistics
            st.write("**Area Statistics:**")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Max', 'Std Dev', 'Zero Area Fires'],
                'Value': [
                    f"{filtered_data['area'].mean():.2f} ha",
                    f"{filtered_data['area'].median():.2f} ha",
                    f"{filtered_data['area'].max():.2f} ha",
                    f"{filtered_data['area'].std():.2f} ha",
                    f"{(filtered_data['area'] == 0).sum()} ({(filtered_data['area'] == 0).mean()*100:.1f}%)"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("Weather Variable Distributions")
        weather_vars = ['temp', 'RH', 'wind', 'rain']
        available_weather = [var for var in weather_vars if var in filtered_data.columns]
        
        if available_weather:
            selected_weather = st.selectbox("Select Weather Variable", available_weather)
            
            fig_weather = px.box(
                filtered_data,
                y=selected_weather,
                title=f"Distribution of {selected_weather}",
                labels={selected_weather: selected_weather.upper()}
            )
            st.plotly_chart(fig_weather, use_container_width=True)
            
            # Weather statistics
            st.write(f"**{selected_weather.upper()} Statistics:**")
            weather_stats = filtered_data[selected_weather].describe()
            st.write(weather_stats)

    # Correlation analysis
    st.subheader("Feature Correlations")
    correlation_vars = st.multiselect(
        "Select variables for correlation analysis",
        numeric_cols,
        default=numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
    )
    
    if len(correlation_vars) > 1:
        corr_fig = create_correlation_heatmap(filtered_data[correlation_vars])
        st.plotly_chart(corr_fig, use_container_width=True)

with tab2:
    st.header("Fire Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fire Severity Classification")
        if 'area' in filtered_data.columns:
            # Create fire severity categories
            def categorize_fire_severity(area):
                if area == 0:
                    return "No Fire"
                elif area <= 1:
                    return "Small (≤1 ha)"
                elif area <= 10:
                    return "Medium (1-10 ha)"
                elif area <= 100:
                    return "Large (10-100 ha)"
                else:
                    return "Very Large (>100 ha)"
            
            filtered_data['severity'] = filtered_data['area'].apply(categorize_fire_severity)
            severity_counts = filtered_data['severity'].value_counts()
            
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Fire Severity Distribution"
            )
            st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        st.subheader("Fire Indices Analysis")
        fire_indices = ['FFMC', 'DMC', 'DC', 'ISI']
        available_indices = [idx for idx in fire_indices if idx in filtered_data.columns]
        
        if available_indices:
            selected_index = st.selectbox("Select Fire Weather Index", available_indices)
            
            # Scatter plot of index vs burned area
            if 'area' in filtered_data.columns:
                fig_index = px.scatter(
                    filtered_data,
                    x=selected_index,
                    y='area',
                    title=f"{selected_index} vs Burned Area",
                    labels={'area': 'Burned Area (ha)'},
                    opacity=0.6
                )
                fig_index.update_layout(showlegend=False)
                st.plotly_chart(fig_index, use_container_width=True)
    
    # Monthly fire activity
    st.subheader("Monthly Fire Activity")
    if 'month' in filtered_data.columns:
        monthly_stats = filtered_data.groupby('month').agg({
            'area': ['count', 'sum', 'mean']
        }).round(2)
        monthly_stats.columns = ['Fire Count', 'Total Area (ha)', 'Avg Area (ha)']
        
        fig_monthly = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Number of Fires by Month', 'Total Burned Area by Month')
        )
        
        fig_monthly.add_trace(
            go.Bar(x=monthly_stats.index, y=monthly_stats['Fire Count'], name='Fire Count'),
            row=1, col=1
        )
        
        fig_monthly.add_trace(
            go.Bar(x=monthly_stats.index, y=monthly_stats['Total Area (ha)'], name='Total Area'),
            row=1, col=2
        )
        
        fig_monthly.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        st.dataframe(monthly_stats, use_container_width=True)

with tab3:
    st.header("Weather Patterns")
    
    weather_vars = ['temp', 'RH', 'wind', 'rain']
    available_weather = [var for var in weather_vars if var in filtered_data.columns]
    
    if available_weather:
        # Weather vs Fire Severity
        st.subheader("Weather Conditions vs Fire Severity")
        
        if 'severity' in filtered_data.columns:
            selected_weather_var = st.selectbox(
                "Select Weather Variable", 
                available_weather,
                key="weather_analysis"
            )
            
            fig_weather_severity = px.box(
                filtered_data,
                x='severity',
                y=selected_weather_var,
                title=f"{selected_weather_var.upper()} by Fire Severity"
            )
            fig_weather_severity.update_xaxes(tickangle=45)
            st.plotly_chart(fig_weather_severity, use_container_width=True)
        
        # Weather correlations with fire area
        st.subheader("Weather-Fire Relationships")
        if 'area' in filtered_data.columns and len(available_weather) >= 2:
            weather_corr = filtered_data[available_weather + ['area']].corr()['area'].sort_values(ascending=False)
            
            fig_corr_bar = px.bar(
                x=weather_corr.index[:-1],
                y=weather_corr.values[:-1],
                title="Correlation of Weather Variables with Burned Area",
                labels={'x': 'Weather Variable', 'y': 'Correlation with Area'}
            )
            st.plotly_chart(fig_corr_bar, use_container_width=True)
        
        # Seasonal weather patterns
        if 'month' in filtered_data.columns:
            seasonal_fig = create_seasonal_analysis(filtered_data, available_weather)
            if seasonal_fig:
                st.plotly_chart(seasonal_fig, use_container_width=True)

with tab4:
    st.header("Spatial Analysis")
    
    if 'X' in filtered_data.columns and 'Y' in filtered_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fire Locations")
            fig_locations = px.scatter(
                filtered_data,
                x='X',
                y='Y',
                color='area' if 'area' in filtered_data.columns else None,
                size='area' if 'area' in filtered_data.columns else None,
                title="Fire Locations and Severity",
                labels={'X': 'X Coordinate', 'Y': 'Y Coordinate'},
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_locations, use_container_width=True)
        
        with col2:
            st.subheader("Spatial Statistics")
            spatial_stats = pd.DataFrame({
                'Coordinate': ['X', 'Y'],
                'Min': [filtered_data['X'].min(), filtered_data['Y'].min()],
                'Max': [filtered_data['X'].max(), filtered_data['Y'].max()],
                'Mean': [filtered_data['X'].mean(), filtered_data['Y'].mean()],
                'Std': [filtered_data['X'].std(), filtered_data['Y'].std()]
            }).round(2)
            st.dataframe(spatial_stats, use_container_width=True)
            
            # Area distribution by location
            if 'area' in filtered_data.columns:
                st.write("**High-Risk Areas (>1 ha):**")
                high_risk = filtered_data[filtered_data['area'] > 1]
                if len(high_risk) > 0:
                    st.write(f"- Number of high-risk fires: {len(high_risk)}")
                    st.write(f"- Average X coordinate: {high_risk['X'].mean():.2f}")
                    st.write(f"- Average Y coordinate: {high_risk['Y'].mean():.2f}")
                else:
                    st.write("No high-risk fires found in filtered data")
    else:
        st.info("Spatial coordinates (X, Y) not available in the dataset")
        
        # Alternative spatial analysis if coordinates exist in different format
        location_cols = [col for col in filtered_data.columns if any(keyword in col.lower() for keyword in ['lat', 'lon', 'coord', 'location'])]
        if location_cols:
            st.write("Available location-related columns:", location_cols)

# Data export
st.header("Data Export")
col1, col2 = st.columns(2)

with col1:
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"forest_fire_data_filtered_{len(filtered_data)}_records.csv",
            mime="text/csv"
        )

with col2:
    st.write(f"**Current Filter Results:**")
    st.write(f"- Total records: {len(filtered_data)}")
    st.write(f"- Date range: {filtered_data['month'].min()} to {filtered_data['month'].max()}" if 'month' in filtered_data.columns else "- No temporal data")
    st.write(f"- Max burned area: {filtered_data['area'].max():.2f} ha" if 'area' in filtered_data.columns else "- No area data")
