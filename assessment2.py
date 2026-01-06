"""
Understanding GIS: Assessment 2
@author 14186840
"""
from time import perf_counter

# set start time
start_time = perf_counter()	

# --- NO CODE ABOVE HERE ---

# Import required libraries
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import uniform
from shapely import Point
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import LinearSegmentedColormap

# 1.Load all required GIS data files and Set algorithm parameters

# Algorithm parameters
PARAM_W = 20    # Weighting influence
PARAM_S = 0.5   # Spatial ambiguity

# Load all required GIS data files
def load_gis_data():
    tweets = gpd.read_file("./data/wr/level3-tweets-subset.shp")
    gm_districts = gpd.read_file("./data/wr/gm-districts.shp")
    pop_raster = rasterio.open("./data/wr/100m_pop_2019.tif")
    
    # Read the data into the memory
    pop_data = pop_raster.read(1)
    
    # Ensure Coordinate Reference Systems match
    if tweets.crs != gm_districts.crs:
        tweets = tweets.to_crs(gm_districts.crs)
       
    # Get study area
    area_bounds = gm_districts.total_bounds
    
    return tweets, pop_raster, pop_data, gm_districts, area_bounds
           
# 2.Perform weighted redistribution
def perform_weighted_redistribution(tweets, gm_districts, pop_raster, pop_data):
       
    # List to store final redistributed tweet locations
    final_seeds = []

    # Iterate through each administrative district
    for index, district in gm_districts.iterrows():
        geom = district.geometry
        
        # Select all actual tweets located within this district
        tweets_in_district = tweets[tweets.within(geom)]
        
        # Avoid empty data
        if len(tweets_in_district) == 0: continue
            
        # Calculate dynamic radius for this district
        r_meters = math.sqrt((geom.area * PARAM_S) / math.pi)
        
        # Get bounding box of the district
        minx, miny, maxx, maxy = geom.bounds
        height, width = pop_data.shape
        
        # Process each tweet in this district
        for i in range(len(tweets_in_district)):
            
            # Variable to store best candidate point for tweet
            best_candidate = None
            
            # Tracks the highest population-weight value (initialized to -1)
            max_pop_val = -1
            
            # Counter for number of random candidate points
            candidates_tried = 0
            
            # Generate 'Weighting index' random candidate points
            while candidates_tried < PARAM_W:
                
                # Generate random point 
                rx = uniform(minx, maxx)
                ry = uniform(miny, maxy)
                p_cand = Point(rx, ry)
                
                # Check random points inside the boundary
                if geom.contains(p_cand):
                    
                    # Get population density weight for candidate points
                    row, col = pop_raster.index(p_cand.x, p_cand.y)
                    val = pop_data[row, col] if 0 <= row < height and 0 <= col < width else 0
                    
                    # Keep the candidate with the highest population weight
                    if val > max_pop_val: max_pop_val, best_candidate = val, p_cand
                    
                    # Ensure have at least one candidate
                    if best_candidate is None: best_candidate = p_cand
                        
                    # Record generated candidate point
                    candidates_tried += 1
            
            # Store the best candidate location and the calculated radius
            if best_candidate:
                final_seeds.append((best_candidate.x, best_candidate.y, r_meters))
                
    return final_seeds

# 3.Calculate the weighted density
def calculate_weighted_density(seeds_data, pop_raster, pop_data):

    # Use the native shape of the population raster
    height, width = pop_data.shape
    density = np.zeros((height, width), dtype=np.float32)

    # Get pixel resolution from raster transform
    pixel_res = pop_raster.transform[0]
    
    # Store calculated kernel shapes
    kernel_cache = {}
    
    # Iterate through each relocated seed point
    for seed_x, seed_y, r_meters in seeds_data:

        # Convert radius from meters to pixels
        r_px = int(r_meters / pixel_res)
        
        # Get center grid indices directly using rasterio
        center_row, center_col = pop_raster.index(seed_x, seed_y)
        
        # Define bounds for the splat operation
        start_i = max(0, center_row - r_px)
        end_i = min(height, center_row + r_px + 1)
        start_j = max(0, center_col - r_px)
        end_j = min(width, center_col + r_px + 1)
        
        # Skip if completely out of bounds
        if start_i >= end_i or start_j >= end_j: continue
    
        # Create the Kernel
        if r_px not in kernel_cache:
            
            # Create local coordinate grid for the kernel
            ky, kx = np.ogrid[-r_px:r_px+1, -r_px:r_px+1]
            dist_matrix = np.sqrt(kx**2 + ky**2)
            
            # Calaulate Kernel
            kernel = 1 - (dist_matrix / r_px)
            kernel[kernel < 0] = 0 # Clip values outside circle
            
            # Cache the result
            kernel_cache[r_px] = kernel
               
        # Calculate slicing offsets to map kernel to the density grid
        k_start_i = start_i - (center_row - r_px)
        k_end_i   = k_start_i + (end_i - start_i)
        k_start_j = start_j - (center_col - r_px)
        k_end_j   = k_start_j + (end_j - start_j)
        
        # Retrieve the pre-calculated kernel for this radius
        full_kernel = kernel_cache[r_px]
        
        # Extract the specific portion of the kernel to add
        kernel_slice = full_kernel[k_start_i:k_end_i, k_start_j:k_end_j]
        
        # Add this kernel chunk to the main density map
        density[start_i:end_i, start_j:end_j] += kernel_slice

    return density

# 4.Drawing the map
def visualize_hotspot(gm_districts, density, area_bounds):
   
    # Custom colors: blue → yellow → red (corresponding to low to high density)
    cmap = LinearSegmentedColormap.from_list(None, ['#368fc3', '#fffeca', '#e13024'])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Remove axes
    ax.axis('off')
    
    # Set title
    fig.suptitle('Weighted Redistribution of Royal Wedding Twitter Activity in Greater Manchester', 
                 fontsize=17, ha='center')
    
    # Set display range with a small buffer to show the study area completely
    x_min, y_min, x_max, y_max = area_bounds
    buffer = min(x_max - x_min, y_max - y_min) / 50
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    
    # Plot administrative boundaries
    gm_districts.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, zorder=2)
    
    # Plot density heatmap
    ax.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='upper',
          cmap=cmap, vmin=np.nanmin(density), vmax=np.nanmax(density), alpha=0.9, zorder=1)

    # Add scale bar
    ax.add_artist(ScaleBar(dx=1, units="m", location="lower right",
                       length_fraction=0.25, font_properties={'size':13}))

    # Add legend
    legend_ax = ax.inset_axes([0.02, 0.02, 0.03, 0.15])
    fig.colorbar(ax.images[0], cax=legend_ax, orientation='vertical').set_ticks([])
    legend_ax.text(1.2, 0.95, 'High Tweet Activity', fontsize=14, transform=legend_ax.transAxes)
    legend_ax.text(1.2, 0.05, 'Low Tweet Activity', fontsize=14, transform=legend_ax.transAxes)

    # Add north arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.9),
            arrowprops=dict(facecolor='black', width=4, headwidth=12),
            ha='center', va='center', fontsize=14, xycoords='axes fraction')
    
    # Adjust figure margins to avoid clipping title and elements
    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.08, right=0.95)
    
    # Save the image
    plt.savefig('./out/assessment2.png', dpi=300, bbox_inches='tight')
        
# Main function: program entry point
if __name__ == "__main__":
    # Running functions in sequence
    tweets, pop_raster, pop_data, gm_districts, area_bounds = load_gis_data()
    seeds_data = perform_weighted_redistribution(tweets, gm_districts, pop_raster, pop_data)
    density = calculate_weighted_density(seeds_data, pop_raster, pop_data)
    visualize_hotspot(gm_districts, density, area_bounds)

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")