import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.database import (
    init_database, store_fire_data, get_fire_data, get_model_history, 
    get_prediction_history, get_database_stats, backup_database_to_csv,
    clear_old_data, store_model_results, store_prediction
)

st.set_page_config(page_title="Database Management", page_icon="🗄️", layout="wide")

st.title("🗄️ Database Management")

st.markdown("""
This page provides comprehensive database management capabilities for the forest fire prediction system,
including data storage, retrieval, analytics, and maintenance operations.
""")

# Initialize database if needed
if st.sidebar.button("Initialize Database"):
    with st.spinner("Initializing database tables..."):
        if init_database():
            st.success("Database initialized successfully!")
        else:
            st.error("Failed to initialize database")

# Database statistics
st.header("Database Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Database Statistics")
    
    try:
        stats = get_database_stats()
        
        if stats:
            # Create metrics display
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Fire Data Records", f"{stats.get('fire_data_records', 0):,}")
            
            with metric_col2:
                st.metric("Model Results", f"{stats.get('model_results', 0):,}")
            
            with metric_col3:
                st.metric("Predictions", f"{stats.get('predictions', 0):,}")
            
            with metric_col4:
                st.metric("Simulations", f"{stats.get('simulations', 0):,}")
            
            if 'data_date_range' in stats:
                st.info(f"📅 Data Range: {stats['data_date_range']}")
        else:
            st.warning("Unable to retrieve database statistics")
    
    except Exception as e:
        st.error(f"Error retrieving database stats: {str(e)}")

with col2:
    st.subheader("Quick Actions")
    
    if st.button("🔄 Refresh Stats"):
        st.rerun()
    
    if st.button("📁 Backup Database"):
        with st.spinner("Creating backup..."):
            if backup_database_to_csv():
                st.success("Database backup completed!")
            else:
                st.error("Backup failed")
    
    days_to_keep = st.number_input("Days to Keep", min_value=1, max_value=365, value=30)
    if st.button("🧹 Clean Old Data"):
        with st.spinner("Cleaning old data..."):
            if clear_old_data(days_to_keep):
                st.success("Data cleanup completed!")
            else:
                st.error("Cleanup failed")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Fire Data", "🤖 Model History", "🎯 Predictions", "📊 Analytics", "⚙️ Maintenance"
])

with tab1:
    st.header("Fire Data Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Stored Fire Data")
        
        # Data filtering options
        with st.expander("Filter Options"):
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                location_filter = st.checkbox("Filter by Location")
                if location_filter:
                    x_range = st.slider("X Coordinate Range", 0.0, 10.0, (0.0, 10.0))
                    y_range = st.slider("Y Coordinate Range", 0.0, 10.0, (0.0, 10.0))
                    location_params = {
                        'x_min': x_range[0], 'x_max': x_range[1],
                        'y_min': y_range[0], 'y_max': y_range[1]
                    }
                else:
                    location_params = None
            
            with filter_col2:
                date_filter = st.checkbox("Filter by Date")
                if date_filter:
                    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
                    end_date = st.date_input("End Date", datetime.now())
                    date_params = {'start_date': start_date, 'end_date': end_date}
                else:
                    date_params = None
        
        # Load and display data
        limit = st.number_input("Max Records to Display", min_value=10, max_value=10000, value=100)
        
        if st.button("Load Fire Data"):
            with st.spinner("Loading fire data from database..."):
                try:
                    fire_data = get_fire_data(
                        limit=limit,
                        location_filter=location_params,
                        date_filter=date_params
                    )
                    
                    if not fire_data.empty:
                        st.success(f"Loaded {len(fire_data)} fire data records")
                        
                        # Display data
                        st.dataframe(fire_data, use_container_width=True)
                        
                        # Data visualization
                        st.subheader("Data Visualization")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Area distribution
                            fig_area = px.histogram(
                                fire_data, x='area', nbins=30,
                                title="Burned Area Distribution"
                            )
                            st.plotly_chart(fig_area, use_container_width=True)
                        
                        with viz_col2:
                            # Location scatter
                            fig_location = px.scatter(
                                fire_data, x='X', y='Y', color='area',
                                title="Fire Locations", size='area',
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig_location, use_container_width=True)
                        
                        # Export option
                        csv = fire_data.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=f"fire_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No fire data found with current filters")
                
                except Exception as e:
                    st.error(f"Error loading fire data: {str(e)}")
    
    with col2:
        st.subheader("Add New Data")
        
        # Manual data entry
        with st.form("add_fire_data"):
            st.write("**Manual Fire Record Entry**")
            
            coord_col1, coord_col2 = st.columns(2)
            with coord_col1:
                x_coord = st.number_input("X Coordinate", value=5.0)
                month = st.selectbox("Month", list(range(1, 13)), index=6)
                ffmc = st.number_input("FFMC", value=85.0)
                temp = st.number_input("Temperature (°C)", value=20.0)
                wind = st.number_input("Wind Speed (km/h)", value=10.0)
            
            with coord_col2:
                y_coord = st.number_input("Y Coordinate", value=5.0)
                day = st.selectbox("Day", ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
                dmc = st.number_input("DMC", value=25.0)
                humidity = st.number_input("Humidity (%)", value=50.0)
                rain = st.number_input("Rain (mm)", value=0.0)
            
            dc = st.number_input("DC", value=150.0)
            isi = st.number_input("ISI", value=5.0)
            area = st.number_input("Burned Area (ha)", value=0.0, min_value=0.0)
            
            if st.form_submit_button("Add Record"):
                # Create DataFrame for single record
                new_record = pd.DataFrame({
                    'X': [x_coord], 'Y': [y_coord], 'month': [month], 'day': [day],
                    'FFMC': [ffmc], 'DMC': [dmc], 'DC': [dc], 'ISI': [isi],
                    'temp': [temp], 'RH': [humidity], 'wind': [wind], 'rain': [rain],
                    'area': [area]
                })
                
                if store_fire_data(new_record, source="manual_entry"):
                    st.success("Fire record added successfully!")
                else:
                    st.error("Failed to add fire record")
        
        # File upload
        st.write("**Upload CSV File**")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                upload_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(upload_data.head(), use_container_width=True)
                
                if st.button("Store Uploaded Data"):
                    with st.spinner("Storing uploaded data..."):
                        if store_fire_data(upload_data, source="file_upload"):
                            st.success(f"Successfully stored {len(upload_data)} records!")
                        else:
                            st.error("Failed to store uploaded data")
            
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")

with tab2:
    st.header("Model Training History")
    
    try:
        model_history = get_model_history()
        
        if not model_history.empty:
            st.dataframe(model_history, use_container_width=True)
            
            # Model performance trends
            st.subheader("Model Performance Trends")
            
            # Group by model type
            regression_models = model_history[model_history['Type'] == 'regression']
            classification_models = model_history[model_history['Type'] == 'classification']
            
            if not regression_models.empty:
                fig_reg = px.line(
                    regression_models, x='Training Date', y='Test Score',
                    color='Model', title='Regression Model Performance Over Time'
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            
            if not classification_models.empty:
                fig_class = px.line(
                    classification_models, x='Training Date', y='Test Score',
                    color='Model', title='Classification Model Performance Over Time'
                )
                st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.info("No model training history found")
    
    except Exception as e:
        st.error(f"Error loading model history: {str(e)}")

with tab3:
    st.header("Prediction History")
    
    try:
        prediction_limit = st.slider("Number of Recent Predictions", 10, 1000, 100)
        prediction_history = get_prediction_history(limit=prediction_limit)
        
        if not prediction_history.empty:
            st.dataframe(prediction_history, use_container_width=True)
            
            # Prediction analytics
            st.subheader("Prediction Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level distribution
                if 'Risk Level' in prediction_history.columns:
                    risk_counts = prediction_history['Risk Level'].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values, names=risk_counts.index,
                        title="Risk Level Distribution"
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                # Model usage
                model_usage = prediction_history['Model'].value_counts()
                fig_models = px.bar(
                    x=model_usage.index, y=model_usage.values,
                    title="Model Usage Frequency"
                )
                st.plotly_chart(fig_models, use_container_width=True)
            
            # Prediction trends over time
            if 'Date' in prediction_history.columns:
                prediction_history['Date'] = pd.to_datetime(prediction_history['Date'])
                daily_predictions = prediction_history.groupby(
                    prediction_history['Date'].dt.date
                ).size().reset_index(name='Count')
                
                fig_trends = px.line(
                    daily_predictions, x='Date', y='Count',
                    title='Daily Prediction Volume'
                )
                st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No prediction history found")
    
    except Exception as e:
        st.error(f"Error loading prediction history: {str(e)}")

with tab4:
    st.header("Database Analytics")
    
    # Data quality metrics
    st.subheader("Data Quality Assessment")
    
    try:
        # Get current fire data for analysis
        current_data = get_fire_data(limit=1000)
        
        if not current_data.empty:
            # Missing data analysis
            missing_data = current_data.isnull().sum()
            missing_pct = (missing_data / len(current_data)) * 100
            
            quality_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            }).sort_values('Missing %', ascending=False)
            
            fig_quality = px.bar(
                quality_df, x='Column', y='Missing %',
                title='Data Completeness by Column'
            )
            st.plotly_chart(fig_quality, use_container_width=True)
            
            # Data distribution analysis
            st.subheader("Data Distribution Analysis")
            
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Select Column for Distribution Analysis", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(
                        current_data, x=selected_col,
                        title=f'{selected_col} Distribution'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(
                        current_data, y=selected_col,
                        title=f'{selected_col} Box Plot'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Feature Correlations")
            
            corr_matrix = current_data[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=True, aspect="auto",
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No data available for analytics")
    
    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")

with tab5:
    st.header("Database Maintenance")
    
    # Database optimization
    st.subheader("Database Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Storage Management**")
        
        if st.button("Analyze Storage Usage"):
            with st.spinner("Analyzing storage..."):
                try:
                    stats = get_database_stats()
                    
                    # Estimate storage usage
                    total_records = sum([
                        stats.get('fire_data_records', 0),
                        stats.get('model_results', 0),
                        stats.get('predictions', 0),
                        stats.get('simulations', 0),
                        stats.get('weather_records', 0)
                    ])
                    
                    st.metric("Total Records", f"{total_records:,}")
                    st.info("Storage analysis completed")
                
                except Exception as e:
                    st.error(f"Error analyzing storage: {str(e)}")
        
        # Data archival
        st.write("**Data Archival**")
        archive_days = st.number_input("Archive data older than (days)", min_value=30, value=365)
        
        if st.button("Archive Old Data"):
            st.info("Archival feature would move old data to long-term storage")
    
    with col2:
        st.write("**Data Validation**")
        
        if st.button("Validate Data Integrity"):
            with st.spinner("Validating data..."):
                try:
                    current_data = get_fire_data(limit=1000)
                    
                    if not current_data.empty:
                        # Basic validation checks
                        issues = []
                        
                        # Check for negative areas
                        negative_areas = (current_data['area'] < 0).sum()
                        if negative_areas > 0:
                            issues.append(f"{negative_areas} records with negative area")
                        
                        # Check coordinate ranges
                        invalid_coords = (
                            (current_data['X'] < 0) | (current_data['X'] > 10) |
                            (current_data['Y'] < 0) | (current_data['Y'] > 10)
                        ).sum()
                        if invalid_coords > 0:
                            issues.append(f"{invalid_coords} records with invalid coordinates")
                        
                        # Check month ranges
                        invalid_months = (
                            (current_data['month'] < 1) | (current_data['month'] > 12)
                        ).sum()
                        if invalid_months > 0:
                            issues.append(f"{invalid_months} records with invalid months")
                        
                        if issues:
                            for issue in issues:
                                st.warning(f"⚠️ {issue}")
                        else:
                            st.success("✅ No data integrity issues found")
                    else:
                        st.info("No data to validate")
                
                except Exception as e:
                    st.error(f"Error validating data: {str(e)}")
        
        # Connection test
        st.write("**Connection Test**")
        
        if st.button("Test Database Connection"):
            with st.spinner("Testing connection..."):
                try:
                    test_stats = get_database_stats()
                    if test_stats:
                        st.success("✅ Database connection successful")
                    else:
                        st.error("❌ Database connection failed")
                
                except Exception as e:
                    st.error(f"Connection test failed: {str(e)}")

# Footer with database status
st.markdown("---")
st.markdown("**Database Status:** Connected to PostgreSQL | Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))