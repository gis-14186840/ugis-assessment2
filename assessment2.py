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
from shapely import Point, Polygon, minimum_bounding_circle

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

# Test running
if __name__ == "__main__":
    tweets, pop_raster, gm_districts = load_gis_data()
    gm_global_geom, gm_bounds, gm_districts = merge_gm_boundary(gm_districts)
    random_points = generate_random_points(gm_global_geom)
    seed_coords, seed_weights = select_hotspot_seeds(random_points, gm_global_geom, pop_raster)
    print(f" {seed_coords.shape}")

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")