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
PARAM_W = 20    # w: Weighting influence
PARAM_S = 0.5   # s: Spatial ambiguity

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
    
    # Calculate the total bounds of the study area
    area_bounds = gm_districts.total_bounds
    
    # Get study area
    area_bounds = gm_districts.total_bounds
    
    return tweets, pop_raster, pop_data, gm_districts, area_bounds
       
# 2.Extract population weight value from raster based on the point's geographic coordinates
def get_raster_value(point, pop_raster, pop_data):

    try:
        # Convert geographic coordinates to raster row and column indices
        row, col = pop_raster.index(point.x, point.y)
        
        # Read the pixel value at the corresponding row and column
        if 0 <= row < pop_data.shape[0] and 0 <= col < pop_data.shape[1]:
            return pop_data[row, col]
        else:
            return 0
            
    # Return 0 if the point is out of the raster range (trigger IndexError)
    except IndexError:
        return 0
    
# 3.Perform weighted redistribution
def perform_weighted_redistribution(tweets, districts, pop_raster, pop_data):
       
    # List to store final seed data
    final_seeds = []

    # Iterate through each administrative district
    for index, district in districts.iterrows():
        geom = district.geometry
        
        # Find all actual tweets located within this district
        tweets_in_district = tweets[tweets.within(geom)]
        
        if len(tweets_in_district) == 0:
            continue
            
        # Calculate dynamic radius for this district
        r_meters = math.sqrt((geom.area * PARAM_S) / math.pi)
        
        # Get bounding box of the district for random point generation
        minx, miny, maxx, maxy = geom.bounds
        
        # Process each tweet in this district
        for i in range(len(tweets_in_district)):
            
            best_candidate = None
            max_pop_val = -1
            
            # Generate 'W' random candidate points
            candidates_tried = 0
            while candidates_tried < PARAM_W:
                # Cartesian random point generation (Bounding Box)
                rx = uniform(minx, maxx)
                ry = uniform(miny, maxy)
                p_cand = Point(rx, ry)
                
                # Check if the random point is actually inside the district geometry
                if geom.contains(p_cand):
                    # Get population density weight for this candidate
                    val = get_raster_value(p_cand, pop_raster, pop_data)
                    
                    # Keep the candidate with the highest population weight
                    if val > max_pop_val:
                        max_pop_val = val
                        best_candidate = p_cand
                    
                    # Ensure we have at least one candidate
                    if best_candidate is None:
                        best_candidate = p_cand
                        
                    candidates_tried += 1
            
            # Store the best candidate location and the calculated radius
            if best_candidate:
                final_seeds.append((best_candidate.x, best_candidate.y, r_meters))
                
    return final_seeds

# 4.Calculate the weighted density of seed points
def calculate_weighted_density(seeds_data, pop_raster, pop_data):

    # Use the native shape of the population raster
    height, width = pop_data.shape
    density = np.zeros((height, width), dtype=np.float32)

    # Get pixel resolution from raster transform
    pixel_res = pop_raster.transform[0]
    
    # Kernel Caching
    # Store calculated kernel shapes
    kernel_cache = {}
    
    # Iterate through each relocated seed point
    for seed_x, seed_y, r_meters in seeds_data:

        # Convert radius from meters to pixels
        r_px = int(r_meters / pixel_res)
        if r_px < 1: r_px = 1
        
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
            kernel_cache[r_px] = kernel.astype(np.float32)
        
        full_kernel = kernel_cache[r_px]
        
        # Calculate slicing offsets to map kernel to the density grid
        k_start_i = start_i - (center_row - r_px)
        k_end_i   = k_start_i + (end_i - start_i)
        k_start_j = start_j - (center_col - r_px)
        k_end_j   = k_start_j + (end_j - start_j)
        
        # Add kernel to the density matrix (Vectorized Addition)
        density[start_i:end_i, start_j:end_j] += full_kernel[k_start_i:k_end_i, k_start_j:k_end_j]

    return density

# 5.Drawing the map
def visualize_hotspot(gm_districts, density, area_bounds):
   
    # Custom colors: blue → yellow → red (corresponding to low to high density)
    colors = ['#368fc3', '#fffeca', '#e13024']
    cmap = LinearSegmentedColormap.from_list(None, colors)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Remove axes
    ax.axis('off')
    
    # Set title
    fig.suptitle('Weighted Redistribution of Royal Wedding Twitter Activity in Greater Manchester', 
                 fontsize=16, ha='center')
    
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
                       length_fraction=0.25, width_fraction=0.015,
                       font_properties={'size':12}, color='black'))

    # Add legend
    legend_ax = ax.inset_axes([0.02, 0.02, 0.03, 0.15])
    fig.colorbar(ax.images[0], cax=legend_ax, orientation='vertical').set_ticks([])
    legend_ax.text(1.2, 0.95, 'Most Tweet Activity', ha='left', va='center', fontsize=10, transform=legend_ax.transAxes)
    legend_ax.text(1.2, 0.05, 'Least Tweet Activity', ha='left', va='center', fontsize=10, transform=legend_ax.transAxes)

    # Add north arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.9),
            arrowprops=dict(facecolor='black', width=4, headwidth=12),
            ha='center', va='center', fontsize=14, xycoords='axes fraction')
    
    # Adjust figure margins to avoid clipping title and elements
    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.08, right=0.95)
    
    # Save the image
    plt.savefig('./out/assessment2.png', dpi=300, bbox_inches='tight')
        
# Main function: program entry point (only execute when running this script directly)
if __name__ == "__main__":
    # Call functions in sequence to complete the full workflow
    tweets, pop_raster, pop_data, gm_districts, area_bounds = load_gis_data()
    seeds_data = perform_weighted_redistribution(tweets, gm_districts, pop_raster, pop_data)
    density = calculate_weighted_density(seeds_data, pop_raster, pop_data)
    visualize_hotspot(gm_districts, density, area_bounds)

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")