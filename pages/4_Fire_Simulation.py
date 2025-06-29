import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from utils.simulation import FireSpreadSimulator, CellularAutomataSimulator
from utils.visualization import create_simulation_animation, plot_spread_statistics

st.set_page_config(page_title="Fire Simulation", page_icon="🔥", layout="wide")

st.title("🔥 Forest Fire Spread Simulation")

st.markdown("""
This page provides interactive fire spread simulation capabilities using cellular automata and 
physics-based models to predict how fires might spread under different conditions.
""")

# Simulation configuration
st.sidebar.header("Simulation Configuration")

# Basic parameters
simulation_type = st.sidebar.radio(
    "Simulation Type",
    ["Cellular Automata", "Physics-Based Model"],
    help="Choose the type of fire spread simulation"
)

# Grid parameters
grid_size = st.sidebar.slider("Grid Size", 20, 100, 50, help="Size of the simulation grid")
time_steps = st.sidebar.slider("Time Steps", 10, 200, 50, help="Number of simulation steps")

# Environmental conditions
st.sidebar.subheader("Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5)
wind_direction = st.sidebar.slider("Wind Direction (degrees)", 0, 360, 90, 5)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 40.0, 1.0)
fuel_moisture = st.sidebar.slider("Fuel Moisture (%)", 5.0, 30.0, 12.0, 0.5)

# Fire parameters
st.sidebar.subheader("Fire Parameters")
ignition_probability = st.sidebar.slider("Ignition Probability", 0.0, 1.0, 0.8, 0.01)
burn_probability = st.sidebar.slider("Burn Probability", 0.0, 1.0, 0.7, 0.01)
spread_probability = st.sidebar.slider("Spread Probability", 0.0, 1.0, 0.6, 0.01)

# Vegetation parameters
st.sidebar.subheader("Vegetation Parameters")
vegetation_type = st.sidebar.selectbox(
    "Vegetation Type",
    ["Mixed Forest", "Coniferous", "Deciduous", "Grassland", "Shrubland"],
    help="Type of vegetation affects fire behavior"
)

vegetation_density = st.sidebar.slider("Vegetation Density", 0.1, 1.0, 0.7, 0.1)

# Main simulation interface
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Interactive Simulation", "📊 Batch Analysis", "📈 Parameter Sensitivity", "🎯 Scenario Comparison"])

with tab1:
    st.header("Interactive Fire Spread Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Simulation Setup")
        
        # Initialize session state for simulation
        if 'simulation_grid' not in st.session_state:
            st.session_state.simulation_grid = None
            st.session_state.simulation_results = None
        
        # Ignition setup
        ignition_mode = st.radio(
            "Ignition Setup",
            ["Single Point", "Multiple Points", "Linear Front"],
            horizontal=True
        )
        
        if ignition_mode == "Single Point":
            col_x, col_y = st.columns(2)
            with col_x:
                ignition_x = st.slider("Ignition X", 0, grid_size-1, grid_size//2)
            with col_y:
                ignition_y = st.slider("Ignition Y", 0, grid_size-1, grid_size//2)
            ignition_points = [(ignition_x, ignition_y)]
            
        elif ignition_mode == "Multiple Points":
            num_ignitions = st.slider("Number of Ignition Points", 2, 10, 3)
            ignition_points = []
            
            for i in range(num_ignitions):
                col_x, col_y = st.columns(2)
                with col_x:
                    x = st.slider(f"Point {i+1} X", 0, grid_size-1, 
                                 (grid_size//4) * (i+1), key=f"ig_x_{i}")
                with col_y:
                    y = st.slider(f"Point {i+1} Y", 0, grid_size-1, 
                                 grid_size//2, key=f"ig_y_{i}")
                ignition_points.append((x, y))
        
        else:  # Linear Front
            front_start = st.slider("Front Start Y", 0, grid_size-1, grid_size//4)
            front_end = st.slider("Front End Y", 0, grid_size-1, 3*grid_size//4)
            front_x = st.slider("Front X Position", 0, grid_size-1, grid_size//10)
            ignition_points = [(front_x, y) for y in range(front_start, front_end+1)]
    
    with col2:
        st.subheader("Current Conditions")
        
        # Calculate fire danger index
        fire_danger = calculate_fire_danger_index(temperature, humidity, wind_speed, fuel_moisture)
        
        # Display danger level
        if fire_danger < 2:
            danger_color = "green"
            danger_text = "Low"
        elif fire_danger < 4:
            danger_color = "orange"
            danger_text = "Moderate"
        elif fire_danger < 6:
            danger_color = "red"
            danger_text = "High"
        else:
            danger_color = "darkred"
            danger_text = "Extreme"
        
        st.metric("Fire Danger Index", f"{fire_danger:.1f}")
        st.markdown(f"**Danger Level:** :{danger_color}[{danger_text}]")
        
        # Environmental summary
        st.write("**Conditions Summary:**")
        st.write(f"- Temperature: {temperature}°C")
        st.write(f"- Humidity: {humidity}%")
        st.write(f"- Wind: {wind_speed} km/h at {wind_direction}°")
        st.write(f"- Fuel Moisture: {fuel_moisture}%")
        st.write(f"- Vegetation: {vegetation_type}")
    
    # Run simulation button
    if st.button("🔥 Run Simulation", type="primary"):
        with st.spinner("Running fire spread simulation..."):
            try:
                if simulation_type == "Cellular Automata":
                    simulator = CellularAutomataSimulator(
                        grid_size=grid_size,
                        wind_speed=wind_speed,
                        wind_direction=wind_direction,
                        temperature=temperature,
                        humidity=humidity,
                        fuel_moisture=fuel_moisture,
                        vegetation_density=vegetation_density
                    )
                else:
                    simulator = FireSpreadSimulator(
                        grid_size=grid_size,
                        wind_speed=wind_speed,
                        wind_direction=wind_direction,
                        temperature=temperature,
                        humidity=humidity,
                        fuel_moisture=fuel_moisture,
                        vegetation_type=vegetation_type
                    )
                
                # Set ignition points
                for x, y in ignition_points:
                    simulator.ignite(x, y)
                
                # Run simulation
                results = simulator.run_simulation(time_steps)
                
                # Store results in session state
                st.session_state.simulation_results = results
                st.session_state.simulator = simulator
                
                st.success("✅ Simulation completed successfully!")
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
    
    # Display simulation results
    if st.session_state.simulation_results is not None:
        results = st.session_state.simulation_results
        simulator = st.session_state.simulator
        
        st.subheader("Simulation Results")
        
        # Animation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_animation = st.checkbox("Show Animation", value=True)
        with col2:
            animation_speed = st.slider("Animation Speed", 50, 500, 200)
        with col3:
            show_wind_vectors = st.checkbox("Show Wind Vectors", value=True)
        
        # Create and display animation
        if show_animation:
            try:
                animation_fig = create_simulation_animation(
                    results, 
                    wind_speed, 
                    wind_direction,
                    show_vectors=show_wind_vectors
                )
                st.plotly_chart(animation_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Animation error: {str(e)}")
        
        # Simulation statistics
        st.subheader("Simulation Statistics")
        
        # Calculate statistics
        final_grid = results[-1]
        total_burned = np.sum(final_grid == 2)  # 2 = burned state
        total_cells = grid_size * grid_size
        burn_percentage = (total_burned / total_cells) * 100
        
        # Fire progression statistics
        burned_over_time = [np.sum(grid == 2) for grid in results]
        burning_over_time = [np.sum(grid == 1) for grid in results]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Burned Area", f"{total_burned:,} cells")
        with col2:
            st.metric("Burn Percentage", f"{burn_percentage:.1f}%")
        with col3:
            max_burning = max(burning_over_time)
            st.metric("Peak Active Fire", f"{max_burning:,} cells")
        with col4:
            fire_duration = len([x for x in burning_over_time if x > 0])
            st.metric("Fire Duration", f"{fire_duration} steps")
        
        # Fire progression chart
        fig_progression = go.Figure()
        
        fig_progression.add_trace(go.Scatter(
            x=list(range(len(burned_over_time))),
            y=burned_over_time,
            name='Burned Area',
            line=dict(color='red')
        ))
        
        fig_progression.add_trace(go.Scatter(
            x=list(range(len(burning_over_time))),
            y=burning_over_time,
            name='Active Fire',
            line=dict(color='orange')
        ))
        
        fig_progression.update_layout(
            title='Fire Progression Over Time',
            xaxis_title='Time Step',
            yaxis_title='Number of Cells',
            height=400
        )
        
        st.plotly_chart(fig_progression, use_container_width=True)
        
        # Fire spread rate analysis
        if len(burned_over_time) > 1:
            spread_rates = np.diff(burned_over_time)
            avg_spread_rate = np.mean(spread_rates[spread_rates > 0])
            max_spread_rate = np.max(spread_rates)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Spread Rate", f"{avg_spread_rate:.1f} cells/step")
                st.metric("Maximum Spread Rate", f"{max_spread_rate:.1f} cells/step")
            
            with col2:
                fig_spread_rate = px.bar(
                    x=list(range(len(spread_rates))),
                    y=spread_rates,
                    title='Fire Spread Rate by Time Step'
                )
                fig_spread_rate.update_layout(height=300)
                st.plotly_chart(fig_spread_rate, use_container_width=True)
        
        # Export simulation results
        if st.button("Export Simulation Data"):
            try:
                # Create export data
                export_data = {
                    'simulation_parameters': {
                        'grid_size': grid_size,
                        'time_steps': time_steps,
                        'wind_speed': wind_speed,
                        'wind_direction': wind_direction,
                        'temperature': temperature,
                        'humidity': humidity,
                        'fuel_moisture': fuel_moisture,
                        'vegetation_type': vegetation_type,
                        'ignition_points': ignition_points
                    },
                    'results': {
                        'total_burned': int(total_burned),
                        'burn_percentage': float(burn_percentage),
                        'fire_duration': int(fire_duration),
                        'burned_over_time': burned_over_time,
                        'burning_over_time': burning_over_time
                    }
                }
                
                import json
                json_data = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download Simulation Results (JSON)",
                    data=json_data,
                    file_name="fire_simulation_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Export error: {str(e)}")

with tab2:
    st.header("Batch Simulation Analysis")
    
    st.markdown("""
    Run multiple simulations with varying parameters to understand fire behavior 
    under different conditions.
    """)
    
    # Batch simulation configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Ranges")
        
        # Wind speed range
        wind_range = st.slider(
            "Wind Speed Range (km/h)",
            0.0, 50.0, (5.0, 25.0),
            help="Range of wind speeds to test"
        )
        
        # Temperature range
        temp_range = st.slider(
            "Temperature Range (°C)",
            0.0, 50.0, (15.0, 35.0),
            help="Range of temperatures to test"
        )
        
        # Humidity range
        humidity_range = st.slider(
            "Humidity Range (%)",
            0.0, 100.0, (20.0, 80.0),
            help="Range of humidity levels to test"
        )
    
    with col2:
        st.subheader("Batch Settings")
        
        num_simulations = st.slider("Number of Simulations", 5, 50, 10)
        batch_grid_size = st.slider("Grid Size", 20, 60, 30)
        batch_time_steps = st.slider("Time Steps", 20, 100, 40)
        
        # Random seed for reproducibility
        random_seed = st.number_input("Random Seed", 0, 9999, 42)
    
    if st.button("Run Batch Simulations"):
        with st.spinner("Running batch simulations... This may take several minutes."):
            try:
                np.random.seed(random_seed)
                
                batch_results = []
                progress_bar = st.progress(0)
                
                for i in range(num_simulations):
                    # Generate random parameters within ranges
                    sim_wind = np.random.uniform(wind_range[0], wind_range[1])
                    sim_temp = np.random.uniform(temp_range[0], temp_range[1])
                    sim_humidity = np.random.uniform(humidity_range[0], humidity_range[1])
                    sim_wind_dir = np.random.uniform(0, 360)
                    sim_fuel_moisture = np.random.uniform(8, 20)
                    
                    # Create simulator
                    simulator = CellularAutomataSimulator(
                        grid_size=batch_grid_size,
                        wind_speed=sim_wind,
                        wind_direction=sim_wind_dir,
                        temperature=sim_temp,
                        humidity=sim_humidity,
                        fuel_moisture=sim_fuel_moisture,
                        vegetation_density=0.7
                    )
                    
                    # Random ignition point
                    ig_x = np.random.randint(5, batch_grid_size-5)
                    ig_y = np.random.randint(5, batch_grid_size-5)
                    simulator.ignite(ig_x, ig_y)
                    
                    # Run simulation
                    results = simulator.run_simulation(batch_time_steps)
                    
                    # Calculate metrics
                    final_grid = results[-1]
                    total_burned = np.sum(final_grid == 2)
                    burn_percentage = (total_burned / (batch_grid_size ** 2)) * 100
                    
                    # Fire progression
                    burned_over_time = [np.sum(grid == 2) for grid in results]
                    max_spread_rate = np.max(np.diff(burned_over_time)) if len(burned_over_time) > 1 else 0
                    
                    batch_results.append({
                        'simulation_id': i,
                        'wind_speed': sim_wind,
                        'temperature': sim_temp,
                        'humidity': sim_humidity,
                        'wind_direction': sim_wind_dir,
                        'fuel_moisture': sim_fuel_moisture,
                        'total_burned': total_burned,
                        'burn_percentage': burn_percentage,
                        'max_spread_rate': max_spread_rate,
                        'fire_danger': calculate_fire_danger_index(sim_temp, sim_humidity, sim_wind, sim_fuel_moisture)
                    })
                    
                    progress_bar.progress((i + 1) / num_simulations)
                
                # Convert to DataFrame
                batch_df = pd.DataFrame(batch_results)
                st.session_state.batch_results = batch_df
                
                st.success("✅ Batch simulations completed!")
                
            except Exception as e:
                st.error(f"Batch simulation error: {str(e)}")
    
    # Display batch results
    if 'batch_results' in st.session_state:
        batch_df = st.session_state.batch_results
        
        st.subheader("Batch Results Analysis")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Burn %", f"{batch_df['burn_percentage'].mean():.1f}%")
        with col2:
            st.metric("Max Burn %", f"{batch_df['burn_percentage'].max():.1f}%")
        with col3:
            st.metric("Avg Spread Rate", f"{batch_df['max_spread_rate'].mean():.1f}")
        
        # Correlation analysis
        st.subheader("Parameter Correlations")
        
        correlation_vars = ['wind_speed', 'temperature', 'humidity', 'fuel_moisture', 'burn_percentage', 'max_spread_rate']
        corr_matrix = batch_df[correlation_vars].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Parameter Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots
        st.subheader("Parameter Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("X Parameter", correlation_vars[:-2], index=0)
            y_param = st.selectbox("Y Parameter", correlation_vars[-2:], index=0)
            
            fig_scatter = px.scatter(
                batch_df,
                x=x_param,
                y=y_param,
                color='fire_danger',
                size='burn_percentage',
                title=f'{y_param} vs {x_param}',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Distribution of burn percentages
            fig_dist = px.histogram(
                batch_df,
                x='burn_percentage',
                nbins=20,
                title='Distribution of Burn Percentages'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("Parameter Sensitivity Analysis")
    
    st.markdown("""
    Analyze how sensitive fire spread is to individual parameters by varying one 
    parameter at a time while keeping others constant.
    """)
    
    # Sensitivity analysis configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Base Conditions")
        base_wind = st.number_input("Base Wind Speed (km/h)", 0.0, 50.0, 15.0)
        base_temp = st.number_input("Base Temperature (°C)", 0.0, 50.0, 25.0)
        base_humidity = st.number_input("Base Humidity (%)", 0.0, 100.0, 50.0)
        base_fuel_moisture = st.number_input("Base Fuel Moisture (%)", 5.0, 30.0, 12.0)
    
    with col2:
        st.subheader("Sensitivity Settings")
        parameter_to_vary = st.selectbox(
            "Parameter to Vary",
            ["wind_speed", "temperature", "humidity", "fuel_moisture"]
        )
        
        num_variations = st.slider("Number of Variations", 5, 20, 10)
        sensitivity_grid_size = st.slider("Grid Size", 20, 50, 30, key="sens_grid")
        sensitivity_time_steps = st.slider("Time Steps", 20, 80, 40, key="sens_time")
    
    # Parameter ranges for sensitivity
    param_ranges = {
        "wind_speed": (0, 40),
        "temperature": (5, 45),
        "humidity": (10, 90),
        "fuel_moisture": (5, 25)
    }
    
    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            try:
                param_range = param_ranges[parameter_to_vary]
                param_values = np.linspace(param_range[0], param_range[1], num_variations)
                
                sensitivity_results = []
                progress_bar = st.progress(0)
                
                for i, param_value in enumerate(param_values):
                    # Set parameters
                    sim_params = {
                        'wind_speed': base_wind,
                        'temperature': base_temp,
                        'humidity': base_humidity,
                        'fuel_moisture': base_fuel_moisture
                    }
                    sim_params[parameter_to_vary] = param_value
                    
                    # Create simulator
                    simulator = CellularAutomataSimulator(
                        grid_size=sensitivity_grid_size,
                        wind_speed=sim_params['wind_speed'],
                        wind_direction=90,  # Fixed wind direction
                        temperature=sim_params['temperature'],
                        humidity=sim_params['humidity'],
                        fuel_moisture=sim_params['fuel_moisture'],
                        vegetation_density=0.7
                    )
                    
                    # Fixed ignition point
                    simulator.ignite(sensitivity_grid_size//2, sensitivity_grid_size//2)
                    
                    # Run simulation
                    results = simulator.run_simulation(sensitivity_time_steps)
                    
                    # Calculate metrics
                    final_grid = results[-1]
                    total_burned = np.sum(final_grid == 2)
                    burn_percentage = (total_burned / (sensitivity_grid_size ** 2)) * 100
                    
                    burned_over_time = [np.sum(grid == 2) for grid in results]
                    max_spread_rate = np.max(np.diff(burned_over_time)) if len(burned_over_time) > 1 else 0
                    
                    sensitivity_results.append({
                        'parameter_value': param_value,
                        'burn_percentage': burn_percentage,
                        'max_spread_rate': max_spread_rate,
                        'total_burned': total_burned
                    })
                    
                    progress_bar.progress((i + 1) / num_variations)
                
                # Convert to DataFrame
                sensitivity_df = pd.DataFrame(sensitivity_results)
                st.session_state.sensitivity_results = sensitivity_df
                st.session_state.sensitivity_parameter = parameter_to_vary
                
                st.success("✅ Sensitivity analysis completed!")
                
            except Exception as e:
                st.error(f"Sensitivity analysis error: {str(e)}")
    
    # Display sensitivity results
    if 'sensitivity_results' in st.session_state:
        sensitivity_df = st.session_state.sensitivity_results
        param_name = st.session_state.sensitivity_parameter
        
        st.subheader(f"Sensitivity to {param_name.replace('_', ' ').title()}")
        
        # Sensitivity plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sens_burn = px.line(
                sensitivity_df,
                x='parameter_value',
                y='burn_percentage',
                title=f'Burn Percentage vs {param_name.replace("_", " ").title()}',
                markers=True
            )
            fig_sens_burn.update_layout(
                xaxis_title=param_name.replace('_', ' ').title(),
                yaxis_title='Burn Percentage (%)'
            )
            st.plotly_chart(fig_sens_burn, use_container_width=True)
        
        with col2:
            fig_sens_rate = px.line(
                sensitivity_df,
                x='parameter_value',
                y='max_spread_rate',
                title=f'Max Spread Rate vs {param_name.replace("_", " ").title()}',
                markers=True
            )
            fig_sens_rate.update_layout(
                xaxis_title=param_name.replace('_', ' ').title(),
                yaxis_title='Max Spread Rate (cells/step)'
            )
            st.plotly_chart(fig_sens_rate, use_container_width=True)
        
        # Sensitivity metrics
        col1, col2, col3 = st.columns(3)
        
        # Calculate sensitivity (rate of change)
        param_diff = sensitivity_df['parameter_value'].diff().iloc[1]
        burn_sensitivity = sensitivity_df['burn_percentage'].diff().iloc[1:].mean() / param_diff
        rate_sensitivity = sensitivity_df['max_spread_rate'].diff().iloc[1:].mean() / param_diff
        
        with col1:
            st.metric("Burn % Sensitivity", f"{burn_sensitivity:.3f}")
        with col2:
            st.metric("Spread Rate Sensitivity", f"{rate_sensitivity:.3f}")
        with col3:
            # Range of values
            burn_range = sensitivity_df['burn_percentage'].max() - sensitivity_df['burn_percentage'].min()
            st.metric("Burn % Range", f"{burn_range:.1f}%")

with tab4:
    st.header("Scenario Comparison")
    
    st.markdown("""
    Compare fire behavior under different predefined scenarios representing 
    typical fire weather conditions.
    """)
    
    # Predefined scenarios
    scenarios = {
        "Low Risk": {
            "wind_speed": 5,
            "temperature": 15,
            "humidity": 70,
            "fuel_moisture": 20,
            "description": "Cool, humid conditions with light wind"
        },
        "Moderate Risk": {
            "wind_speed": 15,
            "temperature": 25,
            "humidity": 50,
            "fuel_moisture": 15,
            "description": "Moderate temperature and humidity"
        },
        "High Risk": {
            "wind_speed": 25,
            "temperature": 35,
            "humidity": 30,
            "fuel_moisture": 10,
            "description": "Hot, dry conditions with strong wind"
        },
        "Extreme Risk": {
            "wind_speed": 40,
            "temperature": 45,
            "humidity": 15,
            "fuel_moisture": 6,
            "description": "Very hot, very dry with extreme wind"
        }
    }
    
    # Scenario selection
    selected_scenarios = st.multiselect(
        "Select Scenarios to Compare",
        list(scenarios.keys()),
        default=list(scenarios.keys())
    )
    
    # Comparison settings
    col1, col2 = st.columns(2)
    
    with col1:
        comp_grid_size = st.slider("Grid Size", 20, 60, 40, key="comp_grid")
        comp_time_steps = st.slider("Time Steps", 20, 100, 50, key="comp_time")
    
    with col2:
        wind_direction_comp = st.slider("Wind Direction", 0, 360, 90, key="comp_wind_dir")
        vegetation_density_comp = st.slider("Vegetation Density", 0.1, 1.0, 0.7, key="comp_veg")
    
    if st.button("Run Scenario Comparison") and selected_scenarios:
        with st.spinner("Running scenario comparison..."):
            try:
                comparison_results = {}
                
                for scenario_name in selected_scenarios:
                    scenario = scenarios[scenario_name]
                    
                    # Create simulator for this scenario
                    simulator = CellularAutomataSimulator(
                        grid_size=comp_grid_size,
                        wind_speed=scenario['wind_speed'],
                        wind_direction=wind_direction_comp,
                        temperature=scenario['temperature'],
                        humidity=scenario['humidity'],
                        fuel_moisture=scenario['fuel_moisture'],
                        vegetation_density=vegetation_density_comp
                    )
                    
                    # Fixed ignition point for fair comparison
                    simulator.ignite(comp_grid_size//2, comp_grid_size//2)
                    
                    # Run simulation
                    results = simulator.run_simulation(comp_time_steps)
                    
                    # Calculate metrics
                    final_grid = results[-1]
                    total_burned = np.sum(final_grid == 2)
                    burn_percentage = (total_burned / (comp_grid_size ** 2)) * 100
                    
                    burned_over_time = [np.sum(grid == 2) for grid in results]
                    burning_over_time = [np.sum(grid == 1) for grid in results]
                    
                    max_spread_rate = np.max(np.diff(burned_over_time)) if len(burned_over_time) > 1 else 0
                    fire_duration = len([x for x in burning_over_time if x > 0])
                    
                    comparison_results[scenario_name] = {
                        'total_burned': total_burned,
                        'burn_percentage': burn_percentage,
                        'max_spread_rate': max_spread_rate,
                        'fire_duration': fire_duration,
                        'burned_over_time': burned_over_time,
                        'final_grid': final_grid,
                        'fire_danger': calculate_fire_danger_index(
                            scenario['temperature'], 
                            scenario['humidity'], 
                            scenario['wind_speed'], 
                            scenario['fuel_moisture']
                        )
                    }
                
                st.session_state.comparison_results = comparison_results
                st.success("✅ Scenario comparison completed!")
                
            except Exception as e:
                st.error(f"Scenario comparison error: {str(e)}")
    
    # Display comparison results
    if 'comparison_results' in st.session_state:
        comp_results = st.session_state.comparison_results
        
        st.subheader("Scenario Comparison Results")
        
        # Create comparison table
        comp_table_data = []
        for scenario_name, results in comp_results.items():
            comp_table_data.append({
                'Scenario': scenario_name,
                'Fire Danger Index': f"{results['fire_danger']:.1f}",
                'Total Burned (cells)': results['total_burned'],
                'Burn Percentage (%)': f"{results['burn_percentage']:.1f}",
                'Max Spread Rate': f"{results['max_spread_rate']:.1f}",
                'Fire Duration (steps)': results['fire_duration']
            })
        
        comp_df = pd.DataFrame(comp_table_data)
        st.dataframe(comp_df, use_container_width=True)
        
        # Comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of burn percentages
            fig_comp_burn = px.bar(
                comp_df,
                x='Scenario',
                y='Burn Percentage (%)',
                title='Burn Percentage by Scenario',
                color='Scenario'
            )
            st.plotly_chart(fig_comp_burn, use_container_width=True)
        
        with col2:
            # Fire progression comparison
            fig_comp_prog = go.Figure()
            
            for scenario_name, results in comp_results.items():
                fig_comp_prog.add_trace(go.Scatter(
                    x=list(range(len(results['burned_over_time']))),
                    y=results['burned_over_time'],
                    name=scenario_name,
                    mode='lines+markers'
                ))
            
            fig_comp_prog.update_layout(
                title='Fire Progression Comparison',
                xaxis_title='Time Step',
                yaxis_title='Burned Area (cells)',
                height=400
            )
            st.plotly_chart(fig_comp_prog, use_container_width=True)
        
        # Final fire state visualization
        st.subheader("Final Fire States")
        
        cols = st.columns(len(comp_results))
        
        for i, (scenario_name, results) in enumerate(comp_results.items()):
            with cols[i]:
                fig_final = px.imshow(
                    results['final_grid'],
                    color_continuous_scale='Reds',
                    title=f'{scenario_name}'
                )
                fig_final.update_layout(height=300)
                st.plotly_chart(fig_final, use_container_width=True)


# Helper functions
def calculate_fire_danger_index(temperature, humidity, wind_speed, fuel_moisture):
    """Calculate a simple fire danger index based on weather conditions."""
    # Normalize inputs
    temp_factor = max(0, (temperature - 10) / 30)  # 0-1 scale
    humidity_factor = max(0, (100 - humidity) / 80)  # 0-1 scale, inverted
    wind_factor = min(1, wind_speed / 40)  # 0-1 scale
    fuel_factor = max(0, (25 - fuel_moisture) / 20)  # 0-1 scale, inverted
    
    # Weighted combination
    danger_index = (temp_factor * 2 + humidity_factor * 2.5 + wind_factor * 1.5 + fuel_factor * 2) / 2
    
    return danger_index * 8  # Scale to 0-8 range
