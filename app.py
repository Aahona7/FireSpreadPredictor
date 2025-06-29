import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_forest_fire_data
from utils.visualization import create_overview_map

# Page configuration
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔥 Forest Fire Prediction & Simulation System</h1>', unsafe_allow_html=True)
    
    # Sidebar information
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    Welcome to the Forest Fire Prediction System! This application provides:
    
    - **Data Overview**: Explore historical fire data and patterns
    - **Model Training**: Train and compare ML models
    - **Fire Prediction**: Predict fire risk based on current conditions
    - **Fire Simulation**: Simulate fire spread scenarios
    - **Model Performance**: Evaluate model accuracy and metrics
    """)
    
    # Main dashboard overview
    st.header("System Overview")
    
    # Initialize session state for data
    if 'fire_data' not in st.session_state:
        with st.spinner("Loading forest fire data..."):
            try:
                from utils.database import get_fire_data, init_database, store_fire_data
                
                # Initialize database
                init_database()
                
                # Try loading from database first
                db_data = get_fire_data(limit=1000)
                
                if not db_data.empty:
                    st.session_state.fire_data = db_data
                    st.success(f"Loaded {len(db_data)} fire records from database!")
                else:
                    # Fall back to external data sources
                    data = load_forest_fire_data()
                    if data is not None:
                        st.session_state.fire_data = data
                        # Store in database for future use
                        store_fire_data(data, source="uci_repository")
                        st.success(f"Loaded {len(data)} fire records and stored in database!")
                    else:
                        st.session_state.fire_data = None
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.session_state.fire_data = None
    
    if st.session_state.fire_data is not None:
        data = st.session_state.fire_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Fire Records",
                value=f"{len(data):,}",
                delta=None
            )
        
        with col2:
            total_area = data['area'].sum() if 'area' in data.columns else 0
            st.metric(
                label="Total Burned Area (ha)",
                value=f"{total_area:,.1f}",
                delta=None
            )
        
        with col3:
            avg_temp = data['temp'].mean() if 'temp' in data.columns else 0
            st.metric(
                label="Avg Temperature (°C)",
                value=f"{avg_temp:.1f}",
                delta=None
            )
        
        with col4:
            high_risk_count = len(data[data['area'] > 1]) if 'area' in data.columns else 0
            st.metric(
                label="High Risk Fires (>1ha)",
                value=f"{high_risk_count:,}",
                delta=None
            )
        
        # Quick overview map
        st.subheader("Fire Distribution Overview")
        try:
            overview_map = create_overview_map(data)
            if overview_map:
                st.components.v1.html(overview_map._repr_html_(), height=500)
            else:
                st.info("Map visualization requires location data (X, Y coordinates)")
        except Exception as e:
            st.warning(f"Could not generate map: {str(e)}")
        
        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Dataset information
        with st.expander("Dataset Information"):
            st.write("**Dataset Shape:**", data.shape)
            st.write("**Columns:**", list(data.columns))
            st.write("**Data Types:**")
            st.write(data.dtypes)
            
            if data.isnull().sum().sum() > 0:
                st.write("**Missing Values:**")
                st.write(data.isnull().sum())
    
    else:
        st.error("Unable to load forest fire data. Please check your data source.")
        st.info("""
        The system expects forest fire data with the following features:
        - Weather conditions (temperature, humidity, wind, rain)
        - Fire indices (FFMC, DMC, DC, ISI)
        - Spatial coordinates (X, Y)
        - Burned area (area)
        - Temporal information (month, day)
        """)
    
    # Getting started guide
    st.subheader("Getting Started")
    st.markdown("""
    1. **Explore Data**: Use the 'Data Overview' page to understand the dataset
    2. **Train Models**: Go to 'Model Training' to build prediction models
    3. **Make Predictions**: Use 'Fire Prediction' for risk assessment
    4. **Run Simulations**: Try 'Fire Simulation' for spread modeling
    5. **Evaluate Performance**: Check 'Model Performance' for accuracy metrics
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Forest Fire Prediction System v1.0")

if __name__ == "__main__":
    main()
