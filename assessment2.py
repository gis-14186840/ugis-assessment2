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
from numpy.random import uniform

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
    
# Merge boundaries and get study area
def merge_gm_boundary(gm_districts):
    gm_global_geom = gm_districts.geometry.union_all()
    gm_bounds = gm_global_geom.bounds
    
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
            if len(valid_points) == n:
                break

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
    
    
# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")