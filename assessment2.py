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


# 1.Data load
PARAMS = {
    'tweet_path': './data/wr/level3-tweets-subset.shp', 
    'pop_raster_path': './data/wr/100m_pop_2019.tif', 
    'district_path': './data/wr/gm-districts.shp',}




# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")