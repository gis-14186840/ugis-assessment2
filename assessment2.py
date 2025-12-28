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
import shapely
from numpy.random import uniform, random
from math import radians, sin, cos
from shapely import Point, minimum_bounding_circle
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import LinearSegmentedColormap

# 1.Set running foundation
def load_gis_data():
    
    # Load data
    tweets = gpd.read_file("./data/wr/level3-tweets-subset.shp")
    gm_districts = gpd.read_file("./data/wr/gm-districts.shp")
    pop_raster = rasterio.open("./data/wr/100m_pop_2019.tif")
    
    # Merge boundaries to get study area
    gm_global_geom = gm_districts.unary_union
    gm_bounds = gm_global_geom.bounds
    return tweets, pop_raster, gm_districts, gm_global_geom, gm_bounds
    
# 2.Generate random points in study area
def generate_random_points(gm_global_geom, n=500):
    
    # Generate repeatable random points
    np.random.seed(42)
    min_x, min_y, max_x, max_y = gm_global_geom.bounds

    # Creat empty list
    valid_points = []
    
    # Generating random points
    while len(valid_points) < n:
        x, y = uniform(min_x, max_x), uniform(min_y, max_y)
        
        # Check the point inside the boundary
        if gm_global_geom.contains(Point(x, y)):
            valid_points.append(Point(x, y))
    
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
    
# 4.Single-point weighted iterative relocation & high-weight filtering
def relocate_point(random_points, polygon, pop_raster, iterations=10):
    
    # Calculate minimum bounding circle
    min_circle_poly = minimum_bounding_circle(gm_global_geom)
    center = min_circle_poly.centroid
    min_circle_radius = center.distance(Point(min_circle_poly.exterior.coords[0]))
    
    # Define the fuzziness factor
    max_offset = min_circle_radius * 0.1
    
    # Creat candidate relocated seed points list
    seed_candidates = []
    
    # Relocate points and collect weights
    for p in random_points:
        
        # Initialization the optimal point and the optimal weight
        best_point, max_weight = p, -1
        
        # Randomly generate the offset direction and distance
        for _ in range(iterations):
            relocated = Point(
                p.x + sin(radians(random()*360)) * (random()*max_offset),
                p.y + cos(radians(random()*360)) * (random()*max_offset))
            
            # Filter points within the range
            if gm_global_geom.contains(relocated):
                current_weight = get_raster_value(relocated, pop_raster)
                
                # Update the candidate point when the weight is higher
                if current_weight > max_weight:
                    best_point, max_weight = relocated, current_weight
        seed_candidates.append((best_point.x, best_point.y, max_weight))
    
    # Sort candidate seed points in descending order
    seed_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select the top 35% of high‑weight seed points
    top_seeds = seed_candidates[:int(len(seed_candidates)*0.35)]
    
    # Extract coordinates of selected seed points
    seed_coords = np.array([(x, y) for x, y, w in top_seeds])
    
    # Extract seed points corresponding weights
    seed_weights = np.array([w for x, y, w in top_seeds])
       
    return seed_coords, seed_weights

# 5.Calculate the weighted density
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
    avg_pixel_res = ((x_max - x_min) + (y_max - y_min)) / (2 * grid_size)

    # Convert fuzzy radius in pixels
    fuzzy_radius_px = int(np.ceil(fuzzy_radius / avg_pixel_res))
    
    # Calculate the size of the fuzzy kernel
    kernel_size = 2 * fuzzy_radius_px + 1
    
    # Generate index grids
    yk, xk = np.ogrid[-fuzzy_radius_px:fuzzy_radius_px+1, -fuzzy_radius_px:fuzzy_radius_px+1]
    
    # Calculate the distance of kernel cells
    geo_dist = np.hypot(xk*avg_pixel_res, yk*avg_pixel_res)
    
    # Generate fuzzy kernel
    fuzzy_kernel = np.where(geo_dist <= fuzzy_radius, 1 - geo_dist/fuzzy_radius, 0).astype(np.float32)
    
    # Perform weighted superposition of fuzzy kernel matrix
    for (seed_x, seed_y), seed_weight in zip(seed_coords, seed_weights):

        # Skip seed points with non-positive weight
        if seed_weight <= 0: continue
        
        # Convert seed point coordinates to grid pixel indices
        grid_i = int(np.round((seed_x - x_min) / ((x_max - x_min)/grid_size)))
        grid_j = int(np.round((seed_y - y_min) / ((y_max - y_min)/grid_size)))
        
        # Skip seed points outside the boundary
        if 0 <= grid_i < grid_size and 0 <= grid_j < grid_size:
        
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
    
    return X, Y, np.where(shapely.contains_xy(gm_global_geom, X, Y), density, np.nan)

# 6.Drawing the map
def visualize_hotspot(gm_districts, X, Y, density, gm_bounds, gm_global_geom):
   
    # Generate gradient colours
    colors = ['#368fc3', '#fffeca', '#e13024']
    cmap = LinearSegmentedColormap.from_list(None, colors, N=10)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Remove axes
    ax.axis('off')
    
    # Set title
    fig.suptitle('Weighted Redistribution of Royal Wedding Twitter Activity in Greater Manchester', 
                 fontsize=16, ha='center')
    
    # Set boundary
    x_min, y_min, x_max, y_max = gm_bounds
    buffer = min(x_max - x_min, y_max - y_min) / 50
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    
    # Plot administrative boundaries
    gm_districts.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, zorder=2)
    
    # Plot heatmap
    ax.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='lower',
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
    
    # Adjust the spacing
    plt.subplots_adjust(bottom=0.08, top=0.95, left=0.08, right=0.95)
    
    # Save the image
    plt.savefig('./out/assessment2.png', dpi=300, bbox_inches='tight')
        
# Main program
if __name__ == "__main__":
    # Full workflow call
    tweets, pop_raster, gm_districts, gm_global_geom, gm_bounds = load_gis_data()
    random_points = generate_random_points(gm_global_geom)
    seed_coords, seed_weights = relocate_point(random_points, gm_global_geom, pop_raster)
    X, Y, density = calculate_weighted_density(seed_coords, seed_weights, gm_bounds, gm_global_geom)
    visualize_hotspot(gm_districts, X, Y, density, gm_bounds, gm_global_geom)

# --- NO CODE BELOW HERE ---

# report runtime
print(f"completed in: {perf_counter() - start_time} seconds")