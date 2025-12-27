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

# 1.Data load
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

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")