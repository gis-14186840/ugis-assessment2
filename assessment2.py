"""
Understanding GIS: Assessment 2
@author 14186840
"""
from time import perf_counter

# set start time
start_time = perf_counter()	

# --- NO CODE ABOVE HERE ---

''' --- ALL CODE MUST BE INSIDE HERE --- '''

# Import required libraries
import geopandas as gpd
import rasterio
import numpy as np
from shapely import Point
from numpy.random import uniform, random
from math import radians, sin, cos
from shapely import Point, minimum_bounding_circle, STRtree

# 1.Set running foundation
# Set data path
PARAMS = {
    'tweet_path': './data/wr/level3-tweets-subset.shp', 
    'pop_raster_path': './data/wr/100m_pop_2019.tif', 
    'district_path': './data/wr/gm-districts.shp',}

# Load data
def load_gis_data():
    tweets = gpd.read_file(PARAMS['tweet_path'])
    gm_districts = gpd.read_file(PARAMS['district_path'])
    pop_raster = rasterio.open(PARAMS['pop_raster_path'])
    return tweets, pop_raster, gm_districts
    
# Merge boundaries and get study area
def merge_gm_boundary(gm_districts):
    gm_global_geom = gm_districts.geometry.union_all()
    gm_bounds = gm_global_geom.bounds
    return gm_global_geom, gm_bounds, gm_districts
    
# 2.Generate random points in study area
def generate_random_points(gm_global_geom, n=500):
    
    # Generate repeatable random points
    np.random.seed(42)
    min_x, min_y, max_x, max_y = gm_global_geom.bounds

    # Generate 2 times the number of candidate points
    x_coords = uniform(min_x, max_x, size=n * 2)
    y_coords = uniform(min_y, max_y, size=n * 2)
    
    # Creat empty list
    valid_points = []
    
    # Coordinate pairing
    for x, y in zip(x_coords, y_coords):
        point = Point(x, y)
        
        # Screen the points within the study area
        if point.within(gm_global_geom):
            valid_points.append(point)
            
            # Stop when the required number is reached
            if len(valid_points) == n: break
    return valid_points
    
# 3.Extract population weight value from raster
def get_raster_value(point, pop_raster):

    try:
        # Convert geographic coordinates to raster indexes
        row, col = pop_raster.index(point.x, point.y)
        value = pop_raster.read(1)[row, col]
        # Filter invalid values
        return value if value >= 0 else 0
    
    # Return 0 when exceeding the raster range
    except IndexError:
        return 0
    
# 4.Cartesian coordinate system point offset calculation
def cartesian_offset(point, distance, azimuth):
    
    # Convert azimuth from degrees to radians
    azimuth_rad = radians(azimuth)
    
    # Calculate the offset
    easting = point.x + sin(azimuth_rad) * distance
    northing = point.y + cos(azimuth_rad) * distance
    return Point(easting, northing)

# 5.Single-point weighted iterative relocation
def relocate_point(point, polygon, iterations, max_offset, pop_raster):
    
    # Initialize the best point
    max_weight = -1
    best_point = point
    
    # Generate random offset distance and azimuth
    for _ in range(iterations):
        offset_dist = random() * max_offset
        offset_azimuth = random() * 359.9
        
        # Calculate the new point
        relocated = cartesian_offset(point, offset_dist, offset_azimuth)
        
        # Filter points inside the study area 
        if not relocated.within(polygon):
            continue
        current_weight = get_raster_value(relocated, pop_raster)
        
        # Compare population weight and keep higher weight
        if current_weight > max_weight:
            max_weight = current_weight
            best_point = relocated
            
    return best_point, max_weight

# 6.Seed points batch relocation & high-weight filtering
def select_hotspot_seeds(random_points, gm_global_geom, pop_raster):
    
    # Calculate the minimum bounding circle
    min_circle_poly = minimum_bounding_circle(gm_global_geom)
    
    # Calculate the circle center
    center = min_circle_poly.centroid
    
    # Calculate the radius of the circle
    min_circle_radius = center.distance(Point(min_circle_poly.exterior.coords[0]))
    
    # Define the fuzziness factor is 0.1
    max_offset = min_circle_radius * 0.1
    
    # Creat candidate relocated seed points list
    seed_candidates = []
    
    # Process each random point
    for idx, point in enumerate(random_points):

        # Iterate and relocation 10 times
        relocated_point, weight = relocate_point(point, gm_global_geom, 10, max_offset, pop_raster)
        
        # Store the relocated point's coordinates and weight
        seed_candidates.append((relocated_point.x, relocated_point.y, weight))
    
    # Sort candidate seed points in descending order
    seed_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select the top 35% of high‑weight seed points
    top_seeds = seed_candidates[:int(len(seed_candidates)*0.35)]
    
    # Extract coordinates of selected seed points
    seed_coords = np.array([(x, y) for x, y, w in top_seeds])
    
    # Extract seed points corresponding weights
    seed_weights = np.array([w for x, y, w in top_seeds])
       
    return seed_coords, seed_weights

# 7.Calculate the weighted density
def calculate_weighted_density(seed_coords, seed_weights, gm_bounds, gm_global_geom):

    # Set boundary
    x_min, y_min, x_max, y_max = gm_bounds

    # Set grid parameters
    grid_size = 500
    fuzzy_radius = 1000

    # Create density calculation grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    density = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Calculate average pixel resolution
    pixel_res_x = (x_max - x_min) / grid_size
    pixel_res_y = (y_max - y_min) / grid_size
    avg_pixel_res = (pixel_res_x + pixel_res_y) / 2

    # Convert fuzzy radius in pixels
    fuzzy_radius_px = int(np.ceil(fuzzy_radius / avg_pixel_res))
    
    # Generate linear attenuation fuzzy kernel matrix
    # Set matrix
    kernel_size = 2 * fuzzy_radius_px + 1
    fuzzy_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel_center = fuzzy_radius_px  

    # Assign linear attenuation weights for each pixel
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            # Calculate the pixel offset distance
            dx_px = kx - kernel_center
            dy_px = ky - kernel_center
            geo_dist = np.hypot(dx_px * avg_pixel_res, dy_px * avg_pixel_res)
            
            # Sey linear attenuation rule
            if geo_dist <= fuzzy_radius:
                fuzzy_kernel[ky, kx] = 1 - (geo_dist / fuzzy_radius)

    # Perform weighted superposition of fuzzy kernel matrix
    for seed_idx, (seed_x, seed_y) in enumerate(seed_coords):
        # Get the weight of the current seed point
        seed_weight = seed_weights[seed_idx]
        # Skip seed points with non-positive weight
        if seed_weight <= 0: continue
        
        # Convert seed point coordinates to grid pixel indices
        grid_i = int(np.round((seed_x - x_min) / pixel_res_x))
        grid_j = int(np.round((seed_y - y_min) / pixel_res_y))
        
        # Skip seed points outside the boundary
        if (grid_i < 0 or grid_i >= grid_size or 
            grid_j < 0 or grid_j >= grid_size):
            continue
        
        # Calculate the valid superposition range on the density grid
        # Determine the x direction
        start_i = max(0, grid_i - fuzzy_radius_px)
        end_i = min(grid_size, grid_i + fuzzy_radius_px + 1)
        # Determine the y direction
        start_j = max(0, grid_j - fuzzy_radius_px)
        end_j = min(grid_size, grid_j + fuzzy_radius_px + 1)
        
        # Calculate the valid slice range of the kernel matrix
        kernel_start_i = max(0, fuzzy_radius_px - grid_i)
        kernel_end_i = kernel_size - max(0, (grid_i + fuzzy_radius_px + 1) - grid_size)
        kernel_start_j = max(0, fuzzy_radius_px - grid_j)
        kernel_end_j = kernel_size - max(0, (grid_j + fuzzy_radius_px + 1) - grid_size)
        
        # Weighted superposition and accumulate to density grid
        density[start_j:end_j, start_i:end_i] += (
            fuzzy_kernel[kernel_start_j:kernel_end_j, kernel_start_i:kernel_end_i] * seed_weight)

    return X, Y, density



# Test running
if __name__ == "__main__":
    tweets, pop_raster, gm_districts = load_gis_data()
    gm_global_geom, gm_bounds, gm_districts = merge_gm_boundary(gm_districts)
    random_points = generate_random_points(gm_global_geom)
    seed_coords, seed_weights = select_hotspot_seeds(random_points, gm_global_geom, pop_raster)
    X, Y, density = calculate_weighted_density(seed_coords, seed_weights, gm_bounds, gm_global_geom)
    print(f" {density.min():.2f} ~ {density.max():.2f}")

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")