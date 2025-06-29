"""
Fire Spread Simulation Utilities

This module contains classes and functions for simulating forest fire spread
using cellular automata and physics-based models.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
import streamlit as st

class FireSpreadSimulator:
    """
    Physics-based fire spread simulator using empirical fire behavior models.
    """
    
    def __init__(self, grid_size=50, wind_speed=10, wind_direction=90, 
                 temperature=25, humidity=50, fuel_moisture=12, vegetation_type="Mixed Forest"):
        """
        Initialize the fire spread simulator.
        
        Args:
            grid_size (int): Size of the simulation grid
            wind_speed (float): Wind speed in km/h
            wind_direction (float): Wind direction in degrees (0=North, 90=East)
            temperature (float): Temperature in Celsius
            humidity (float): Relative humidity in percentage
            fuel_moisture (float): Fuel moisture content in percentage
            vegetation_type (str): Type of vegetation
        """
        
        self.grid_size = grid_size
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.temperature = temperature
        self.humidity = humidity
        self.fuel_moisture = fuel_moisture
        self.vegetation_type = vegetation_type
        
        # Initialize grid: 0=unburned, 1=burning, 2=burned
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fuel load map (0-1, where 1 is maximum fuel)
        self.fuel_load = self._initialize_fuel_load()
        
        # Elevation map (affects fire spread)
        self.elevation = self._initialize_elevation()
        
        # Fire intensity map
        self.fire_intensity = np.zeros((grid_size, grid_size))
        
        # Calculate fire behavior parameters
        self.spread_rate = self._calculate_base_spread_rate()
        self.ignition_probability = self._calculate_ignition_probability()
        
    def _initialize_fuel_load(self):
        """Initialize fuel load distribution."""
        np.random.seed(42)
        
        # Base fuel load with some spatial variation
        fuel_load = np.random.normal(0.7, 0.2, (self.grid_size, self.grid_size))
        
        # Apply vegetation type modifier
        vegetation_modifiers = {
            "Mixed Forest": 1.0,
            "Coniferous": 1.2,
            "Deciduous": 0.8,
            "Grassland": 0.6,
            "Shrubland": 0.9
        }
        
        modifier = vegetation_modifiers.get(self.vegetation_type, 1.0)
        fuel_load *= modifier
        
        # Ensure bounds
        fuel_load = np.clip(fuel_load, 0.1, 1.0)
        
        return fuel_load
    
    def _initialize_elevation(self):
        """Initialize elevation map."""
        np.random.seed(43)
        
        # Create simple elevation gradient with some noise
        x, y = np.meshgrid(np.linspace(0, 1, self.grid_size), 
                          np.linspace(0, 1, self.grid_size))
        
        elevation = 100 * (x + y) / 2 + np.random.normal(0, 10, (self.grid_size, self.grid_size))
        elevation = np.clip(elevation, 0, 200)
        
        return elevation
    
    def _calculate_base_spread_rate(self):
        """Calculate base fire spread rate based on conditions."""
        
        # Base spread rate (meters per time step)
        base_rate = 1.0
        
        # Temperature effect
        temp_factor = 1 + (self.temperature - 20) * 0.02
        
        # Humidity effect (inverse relationship)
        humidity_factor = 2 - self.humidity / 100
        
        # Wind effect
        wind_factor = 1 + self.wind_speed * 0.05
        
        # Fuel moisture effect (inverse relationship)
        moisture_factor = 2 - self.fuel_moisture / 20
        
        spread_rate = base_rate * temp_factor * humidity_factor * wind_factor * moisture_factor
        
        return max(0.1, spread_rate)
    
    def _calculate_ignition_probability(self):
        """Calculate base ignition probability."""
        
        # Base ignition probability
        base_prob = 0.1
        
        # Environmental factors
        temp_factor = 1 + (self.temperature - 20) * 0.01
        humidity_factor = 1 - self.humidity / 200
        fuel_factor = 1 - self.fuel_moisture / 50
        
        ignition_prob = base_prob * temp_factor * humidity_factor * fuel_factor
        
        return max(0.01, min(0.8, ignition_prob))
    
    def ignite(self, x, y, intensity=1.0):
        """
        Start a fire at the specified location.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            intensity (float): Initial fire intensity
        """
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1  # Set to burning
            self.fire_intensity[y, x] = intensity
    
    def _calculate_spread_probability(self, from_cell, to_cell):
        """
        Calculate probability of fire spreading from one cell to another.
        
        Args:
            from_cell (tuple): Source cell coordinates (y, x)
            to_cell (tuple): Target cell coordinates (y, x)
            
        Returns:
            float: Spread probability
        """
        
        from_y, from_x = from_cell
        to_y, to_x = to_cell
        
        # Base spread probability
        base_prob = self.spread_rate * 0.1
        
        # Distance factor (adjacent cells only)
        distance = np.sqrt((to_x - from_x)**2 + (to_y - from_y)**2)
        if distance > 1.5:  # Only immediate neighbors
            return 0.0
        
        # Wind effect
        wind_angle = np.radians(self.wind_direction)
        spread_angle = np.arctan2(to_y - from_y, to_x - from_x)
        angle_diff = abs(spread_angle - wind_angle)
        
        # Normalize angle difference to [0, π]
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        # Wind boost for downwind direction
        wind_factor = 1 + (self.wind_speed / 50) * (1 - angle_diff / np.pi)
        
        # Elevation effect (fire spreads faster uphill)
        elevation_diff = self.elevation[to_y, to_x] - self.elevation[from_y, from_x]
        slope_factor = 1 + elevation_diff * 0.01
        
        # Fuel load effect
        fuel_factor = self.fuel_load[to_y, to_x]
        
        # Fire intensity effect
        intensity_factor = self.fire_intensity[from_y, from_x]
        
        # Combined probability
        prob = base_prob * wind_factor * slope_factor * fuel_factor * intensity_factor
        
        return min(0.9, max(0.0, prob))
    
    def _update_fire_intensity(self):
        """Update fire intensity for burning cells."""
        
        burning_mask = (self.grid == 1)
        
        # Intensity increases with fuel load and decreases over time
        self.fire_intensity[burning_mask] *= 0.9  # Decay factor
        self.fire_intensity[burning_mask] += self.fuel_load[burning_mask] * 0.1
        
        # Limit maximum intensity
        self.fire_intensity = np.clip(self.fire_intensity, 0, 2.0)
    
    def step(self):
        """
        Perform one simulation time step.
        
        Returns:
            bool: True if fire is still active, False if extinguished
        """
        
        new_grid = self.grid.copy()
        new_intensity = self.fire_intensity.copy()
        
        # Find all burning cells
        burning_cells = np.where(self.grid == 1)
        
        if len(burning_cells[0]) == 0:
            return False  # No more fire
        
        # For each burning cell, try to spread to neighbors
        for i in range(len(burning_cells[0])):
            y, x = burning_cells[0][i], burning_cells[1][i]
            
            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        # Only spread to unburned cells
                        if self.grid[ny, nx] == 0:
                            spread_prob = self._calculate_spread_probability((y, x), (ny, nx))
                            
                            if np.random.random() < spread_prob:
                                new_grid[ny, nx] = 1
                                new_intensity[ny, nx] = self.fire_intensity[y, x] * 0.8
            
            # Burning cells may burn out
            burnout_prob = 0.1 + (1 - self.fuel_load[y, x]) * 0.1
            if np.random.random() < burnout_prob:
                new_grid[y, x] = 2  # Burned out
                new_intensity[y, x] = 0
        
        # Update grids
        self.grid = new_grid
        self.fire_intensity = new_intensity
        
        # Update fire intensity
        self._update_fire_intensity()
        
        return np.any(self.grid == 1)  # Return True if fire still active
    
    def run_simulation(self, max_steps=100):
        """
        Run the complete simulation.
        
        Args:
            max_steps (int): Maximum number of simulation steps
            
        Returns:
            list: List of grid states at each time step
        """
        
        results = [self.grid.copy()]
        
        for step in range(max_steps):
            fire_active = self.step()
            results.append(self.grid.copy())
            
            if not fire_active:
                break
        
        return results
    
    def get_statistics(self):
        """
        Get current simulation statistics.
        
        Returns:
            dict: Simulation statistics
        """
        
        total_cells = self.grid_size ** 2
        unburned = np.sum(self.grid == 0)
        burning = np.sum(self.grid == 1)
        burned = np.sum(self.grid == 2)
        
        return {
            'total_cells': total_cells,
            'unburned': unburned,
            'burning': burning,
            'burned': burned,
            'burn_percentage': (burned / total_cells) * 100,
            'active_fire': burning > 0
        }


class CellularAutomataSimulator:
    """
    Cellular automata-based fire spread simulator.
    """
    
    def __init__(self, grid_size=50, wind_speed=10, wind_direction=90,
                 temperature=25, humidity=50, fuel_moisture=12, vegetation_density=0.7):
        """
        Initialize the cellular automata simulator.
        
        Args:
            grid_size (int): Size of the simulation grid
            wind_speed (float): Wind speed in km/h
            wind_direction (float): Wind direction in degrees
            temperature (float): Temperature in Celsius
            humidity (float): Relative humidity in percentage
            fuel_moisture (float): Fuel moisture content in percentage
            vegetation_density (float): Vegetation density (0-1)
        """
        
        self.grid_size = grid_size
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.temperature = temperature
        self.humidity = humidity
        self.fuel_moisture = fuel_moisture
        self.vegetation_density = vegetation_density
        
        # Initialize grid: 0=empty, 1=fuel, 2=burning, 3=burned
        self.grid = self._initialize_grid()
        
        # Calculate probabilities
        self.ignition_prob = self._calculate_ignition_probability()
        self.spread_prob = self._calculate_spread_probability()
        self.burnout_prob = self._calculate_burnout_probability()
        
        # Wind effect matrix
        self.wind_matrix = self._create_wind_matrix()
    
    def _initialize_grid(self):
        """Initialize the grid with fuel distribution."""
        np.random.seed(42)
        
        # Create grid with fuel cells based on vegetation density
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Add fuel cells randomly based on vegetation density
        fuel_mask = np.random.random((self.grid_size, self.grid_size)) < self.vegetation_density
        grid[fuel_mask] = 1  # Fuel
        
        return grid
    
    def _calculate_ignition_probability(self):
        """Calculate ignition probability based on conditions."""
        
        base_prob = 0.1
        
        # Environmental factors
        temp_factor = 1 + (self.temperature - 20) * 0.02
        humidity_factor = 1 - self.humidity / 200
        moisture_factor = 1 - self.fuel_moisture / 40
        
        prob = base_prob * temp_factor * humidity_factor * moisture_factor
        
        return max(0.01, min(0.9, prob))
    
    def _calculate_spread_probability(self):
        """Calculate spread probability based on conditions."""
        
        base_prob = 0.4
        
        # Environmental factors
        temp_factor = 1 + (self.temperature - 20) * 0.01
        humidity_factor = 1 - self.humidity / 150
        wind_factor = 1 + self.wind_speed * 0.02
        moisture_factor = 1 - self.fuel_moisture / 30
        
        prob = base_prob * temp_factor * humidity_factor * wind_factor * moisture_factor
        
        return max(0.05, min(0.95, prob))
    
    def _calculate_burnout_probability(self):
        """Calculate burnout probability."""
        
        base_prob = 0.2
        
        # Fuel moisture increases burnout probability
        moisture_factor = 1 + self.fuel_moisture / 50
        
        prob = base_prob * moisture_factor
        
        return max(0.1, min(0.8, prob))
    
    def _create_wind_matrix(self):
        """Create wind effect matrix for directional spread."""
        
        # Create 3x3 matrix representing wind effect on neighboring cells
        wind_matrix = np.ones((3, 3))
        
        if self.wind_speed > 0:
            # Convert wind direction to radians
            wind_rad = np.radians(self.wind_direction)
            
            # Calculate wind effect for each direction
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:  # Center cell
                        continue
                    
                    # Calculate angle from center to this neighbor
                    dy, dx = i - 1, j - 1
                    neighbor_angle = np.arctan2(dy, dx)
                    
                    # Calculate angle difference
                    angle_diff = abs(neighbor_angle - wind_rad)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                    
                    # Wind boost factor (stronger in wind direction)
                    wind_boost = 1 + (self.wind_speed / 50) * (1 - angle_diff / np.pi)
                    wind_matrix[i, j] = wind_boost
        
        return wind_matrix
    
    def ignite(self, x, y):
        """
        Ignite a fire at the specified location.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """
        
        if (0 <= x < self.grid_size and 0 <= y < self.grid_size and 
            self.grid[y, x] == 1):  # Can only ignite fuel cells
            self.grid[y, x] = 2  # Set to burning
    
    def step(self):
        """
        Perform one simulation step using cellular automata rules.
        
        Returns:
            bool: True if fire is still active, False if extinguished
        """
        
        new_grid = self.grid.copy()
        
        # Find all burning cells
        burning_cells = np.where(self.grid == 2)
        
        if len(burning_cells[0]) == 0:
            return False  # No more fire
        
        # Spread fire from burning cells
        for i in range(len(burning_cells[0])):
            y, x = burning_cells[0][i], burning_cells[1][i]
            
            # Check 8-connected neighbors
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:  # Skip center cell
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                        # Only spread to fuel cells
                        if self.grid[ny, nx] == 1:
                            # Calculate spread probability with wind effect
                            wind_effect = self.wind_matrix[dy + 1, dx + 1]
                            effective_prob = self.spread_prob * wind_effect
                            
                            if np.random.random() < effective_prob:
                                new_grid[ny, nx] = 2  # Set to burning
        
        # Handle burnout for currently burning cells
        for i in range(len(burning_cells[0])):
            y, x = burning_cells[0][i], burning_cells[1][i]
            
            if np.random.random() < self.burnout_prob:
                new_grid[y, x] = 3  # Set to burned out
        
        self.grid = new_grid
        
        return np.any(self.grid == 2)  # Return True if fire still active
    
    def run_simulation(self, max_steps=100):
        """
        Run the complete cellular automata simulation.
        
        Args:
            max_steps (int): Maximum number of simulation steps
            
        Returns:
            list: List of grid states at each time step
        """
        
        results = [self.grid.copy()]
        
        for step in range(max_steps):
            fire_active = self.step()
            results.append(self.grid.copy())
            
            if not fire_active:
                break
        
        return results
    
    def get_statistics(self):
        """
        Get current simulation statistics.
        
        Returns:
            dict: Simulation statistics
        """
        
        total_cells = self.grid_size ** 2
        empty = np.sum(self.grid == 0)
        fuel = np.sum(self.grid == 1)
        burning = np.sum(self.grid == 2)
        burned = np.sum(self.grid == 3)
        
        return {
            'total_cells': total_cells,
            'empty': empty,
            'fuel': fuel,
            'burning': burning,
            'burned': burned,
            'burn_percentage': (burned / total_cells) * 100,
            'active_fire': burning > 0
        }


class FirePerimeterTracker:
    """
    Utility class for tracking fire perimeter and calculating spread metrics.
    """
    
    def __init__(self):
        self.perimeter_history = []
        self.area_history = []
        self.spread_rates = []
    
    def update(self, grid):
        """
        Update perimeter tracking with new grid state.
        
        Args:
            grid (np.ndarray): Current fire grid
        """
        
        # Calculate current burned area
        burned_area = np.sum((grid == 2) | (grid == 3))  # Burning or burned
        self.area_history.append(burned_area)
        
        # Calculate perimeter (simplified as edge cells)
        perimeter = self._calculate_perimeter(grid)
        self.perimeter_history.append(perimeter)
        
        # Calculate spread rate
        if len(self.area_history) > 1:
            spread_rate = self.area_history[-1] - self.area_history[-2]
            self.spread_rates.append(max(0, spread_rate))
    
    def _calculate_perimeter(self, grid):
        """
        Calculate fire perimeter length.
        
        Args:
            grid (np.ndarray): Fire grid
            
        Returns:
            int: Perimeter length
        """
        
        # Find fire cells (burning or burned)
        fire_mask = (grid == 2) | (grid == 3)
        
        if not np.any(fire_mask):
            return 0
        
        # Use morphological operations to find perimeter
        from scipy.ndimage import binary_erosion
        
        # Erode the fire mask and subtract to get perimeter
        eroded = binary_erosion(fire_mask)
        perimeter_mask = fire_mask & ~eroded
        
        return np.sum(perimeter_mask)
    
    def get_metrics(self):
        """
        Get spread metrics.
        
        Returns:
            dict: Fire spread metrics
        """
        
        if not self.area_history:
            return {}
        
        return {
            'final_area': self.area_history[-1] if self.area_history else 0,
            'max_area': max(self.area_history) if self.area_history else 0,
            'total_spread': self.area_history[-1] - self.area_history[0] if len(self.area_history) > 1 else 0,
            'avg_spread_rate': np.mean(self.spread_rates) if self.spread_rates else 0,
            'max_spread_rate': max(self.spread_rates) if self.spread_rates else 0,
            'final_perimeter': self.perimeter_history[-1] if self.perimeter_history else 0,
            'max_perimeter': max(self.perimeter_history) if self.perimeter_history else 0
        }


def create_fire_break(grid, x1, y1, x2, y2, width=1):
    """
    Create a fire break in the simulation grid.
    
    Args:
        grid (np.ndarray): Simulation grid
        x1, y1 (int): Starting coordinates
        x2, y2 (int): Ending coordinates
        width (int): Width of the fire break
        
    Returns:
        np.ndarray: Modified grid with fire break
    """
    
    modified_grid = grid.copy()
    
    # Simple line drawing algorithm (Bresenham's line)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        # Clear fuel in fire break area
        for w in range(-width//2, width//2 + 1):
            for h in range(-width//2, width//2 + 1):
                nx, ny = x + w, y + h
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    if modified_grid[ny, nx] == 1:  # Remove fuel
                        modified_grid[ny, nx] = 0
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return modified_grid


def analyze_fire_behavior(simulation_results):
    """
    Analyze fire behavior from simulation results.
    
    Args:
        simulation_results (list): List of simulation grids
        
    Returns:
        dict: Fire behavior analysis
    """
    
    if not simulation_results:
        return {}
    
    # Initialize tracker
    tracker = FirePerimeterTracker()
    
    # Track each time step
    for grid in simulation_results:
        tracker.update(grid)
    
    # Get basic metrics
    metrics = tracker.get_metrics()
    
    # Additional analysis
    final_grid = simulation_results[-1]
    initial_grid = simulation_results[0]
    
    # Fire shape analysis
    if np.any((final_grid == 2) | (final_grid == 3)):
        fire_cells = np.where((final_grid == 2) | (final_grid == 3))
        
        # Bounding box
        min_y, max_y = np.min(fire_cells[0]), np.max(fire_cells[0])
        min_x, max_x = np.min(fire_cells[1]), np.max(fire_cells[1])
        
        fire_width = max_x - min_x + 1
        fire_height = max_y - min_y + 1
        
        # Fire elongation (aspect ratio)
        aspect_ratio = max(fire_width, fire_height) / min(fire_width, fire_height)
        
        metrics.update({
            'fire_width': fire_width,
            'fire_height': fire_height,
            'aspect_ratio': aspect_ratio,
            'bounding_box_area': fire_width * fire_height
        })
    
    # Simulation duration
    metrics['simulation_duration'] = len(simulation_results)
    
    # Fire extinction analysis
    burning_counts = [np.sum(grid == 2) for grid in simulation_results]
    extinction_step = len(burning_counts)
    
    for i, count in enumerate(burning_counts):
        if count == 0 and i > 0:
            extinction_step = i
            break
    
    metrics['extinction_step'] = extinction_step
    metrics['burned_at_extinction'] = np.sum((simulation_results[min(extinction_step, len(simulation_results)-1)] == 2) | 
                                           (simulation_results[min(extinction_step, len(simulation_results)-1)] == 3))
    
    return metrics


def simulate_suppression_efforts(simulator, suppression_points, suppression_effectiveness=0.8):
    """
    Simulate fire suppression efforts.
    
    Args:
        simulator: Fire simulator instance
        suppression_points (list): List of (x, y) coordinates for suppression
        suppression_effectiveness (float): Effectiveness of suppression (0-1)
        
    Returns:
        np.ndarray: Modified grid with suppression effects
    """
    
    modified_grid = simulator.grid.copy()
    
    for x, y in suppression_points:
        if 0 <= x < simulator.grid_size and 0 <= y < simulator.grid_size:
            # Suppression effect in a radius around the point
            radius = 2
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < simulator.grid_size and 0 <= ny < simulator.grid_size:
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        if distance <= radius:
                            # Effectiveness decreases with distance
                            local_effectiveness = suppression_effectiveness * (1 - distance / radius)
                            
                            # Extinguish burning cells
                            if modified_grid[ny, nx] == 2 and np.random.random() < local_effectiveness:
                                modified_grid[ny, nx] = 3  # Burned out
                            
                            # Remove fuel to create fire break
                            elif modified_grid[ny, nx] == 1 and np.random.random() < local_effectiveness * 0.5:
                                modified_grid[ny, nx] = 0  # Remove fuel
    
    return modified_grid


def calculate_fire_danger_rating(temperature, humidity, wind_speed, fuel_moisture):
    """
    Calculate fire danger rating based on weather conditions.
    
    Args:
        temperature (float): Temperature in Celsius
        humidity (float): Relative humidity in percentage
        wind_speed (float): Wind speed in km/h
        fuel_moisture (float): Fuel moisture in percentage
        
    Returns:
        float: Fire danger rating (0-10 scale)
    """
    
    # Normalize inputs to 0-1 scale
    temp_factor = max(0, min(1, (temperature - 5) / 40))
    humidity_factor = max(0, min(1, (100 - humidity) / 80))
    wind_factor = max(0, min(1, wind_speed / 60))
    moisture_factor = max(0, min(1, (30 - fuel_moisture) / 25))
    
    # Weighted combination
    danger_rating = (temp_factor * 0.3 + humidity_factor * 0.3 + 
                    wind_factor * 0.2 + moisture_factor * 0.2) * 10
    
    return danger_rating


def create_evacuation_zones(grid, fire_front_buffer=5):
    """
    Create evacuation zones based on fire location and spread prediction.
    
    Args:
        grid (np.ndarray): Current fire grid
        fire_front_buffer (int): Buffer distance from fire front
        
    Returns:
        np.ndarray: Evacuation zone map (0=safe, 1=watch, 2=warning, 3=evacuation)
    """
    
    evacuation_map = np.zeros_like(grid)
    
    # Find fire cells (burning or burned)
    fire_mask = (grid == 2) | (grid == 3)
    
    if not np.any(fire_mask):
        return evacuation_map
    
    # Calculate distance from fire
    from scipy.ndimage import distance_transform_edt
    
    # Distance from fire cells
    distance_from_fire = distance_transform_edt(~fire_mask)
    
    # Create evacuation zones based on distance
    evacuation_map[distance_from_fire <= 1] = 3  # Immediate evacuation
    evacuation_map[(distance_from_fire > 1) & (distance_from_fire <= 3)] = 2  # Warning
    evacuation_map[(distance_from_fire > 3) & (distance_from_fire <= fire_front_buffer)] = 1  # Watch
    
    return evacuation_map


def export_simulation_data(simulation_results, filename="fire_simulation.csv"):
    """
    Export simulation results to CSV format.
    
    Args:
        simulation_results (list): List of simulation grids
        filename (str): Output filename
        
    Returns:
        pd.DataFrame: Simulation data as DataFrame
    """
    
    data_rows = []
    
    for time_step, grid in enumerate(simulation_results):
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                data_rows.append({
                    'time_step': time_step,
                    'x': x,
                    'y': y,
                    'state': grid[y, x]
                })
    
    df = pd.DataFrame(data_rows)
    
    # Add state labels
    state_labels = {0: 'Unburned/Empty', 1: 'Fuel', 2: 'Burning', 3: 'Burned'}
    df['state_label'] = df['state'].map(state_labels)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    return df


