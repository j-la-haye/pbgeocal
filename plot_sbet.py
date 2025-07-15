import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

def plot_2d_osm_map(csv_file, zoom=14):
    # Load CSV
    df = pd.read_csv(csv_file)
    required_cols = ['lat', 'lon']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain: {required_cols}")
    # Select every 200th row for performance
    df = df.iloc[::200].reset_index(drop=True)

    # Create GeoDataFrame in WGS84
    gdf_wgs = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon_deg'], df['lat_deg']), crs="EPSG:4979")

    # Convert to Web Mercator
    gdf = gdf_wgs.to_crs(epsg=3857)

    # Compute bounds before plotting
    bounds = gdf.total_bounds  # [xmin, ymin, xmax, ymax]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='red', markersize=5, label='Trajectory')

    # Set limits manually
    ax.set_xlim(bounds[0] - 500, bounds[2] + 500)
    ax.set_ylim(bounds[1] - 500, bounds[3] + 500)

    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"⚠️ Could not load base map: {e}")

    # Finalize plot
    ax.set_title("2D Trajectory on OpenStreetMap")
    ax.set_axis_off()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_osm_altitude(csv_file, zoom=13):
    # Load CSV
    df = pd.read_csv(csv_file)
    required_cols = ['lat_deg', 'lon_deg', 'altitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain: {required_cols}")

    # Select every 500th row for performance
    df = df.iloc[::1000].reset_index(drop=True)

    # Create GeoDataFrame in WGS84
    gdf_wgs = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon_deg'], df['lat_deg']), crs="EPSG:4979")

    # Project to Web Mercator for plotting with contextily
    gdf = gdf_wgs.to_crs(epsg=3857)

    # Create LineString for trajectory path
    line = LineString(gdf.geometry.tolist())

    # Normalize altitude for colormap
    norm = mcolors.Normalize(vmin=gdf['altitude'].min(), vmax=gdf['altitude'].max())
    cmap = plt.get_cmap(name='plasma')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the trajectory line
    gpd.GeoSeries([line], crs=gdf.crs).plot(ax=ax, linewidth=1, color='gray', label='Trajectory Line')

    # Plot points with altitude-based color
    gdf.plot(ax=ax,
             column='altitude',
             cmap=cmap,
             markersize=5,
             legend=True,
             legend_kwds={
             'label': "Altitude (m)",
             'shrink': 0.4,
             'aspect': 15,
             'pad': 0.01
             },
             label='GPS Points')

    # Set map bounds
    buffer = 5000  # meters
    xmin, ymin, xmax, ymax = gdf.total_bounds
    ax.set_xlim(xmin - buffer, xmax + buffer)
    ax.set_ylim(ymin - buffer, ymax + buffer)

    # Add a faster tile provider
    try:
        ctx.add_basemap(ax,
                        crs=gdf.crs,
                        source=ctx.providers.CartoDB.Positron,
                        zoom=zoom)
    except Exception as e:
        print(f"⚠️ Could not load base map: {e}")

    ax.set_title("SBET Trajectory with Altitude ")
    ax.set_axis_off()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    csv_file = '/Volumes/fts-addlidar/fts-addlidar/AVIRIS_4_Mission_Processing/2025/Atlans_CHE_ZRH_250705/SBET_Proc/SBET/sbet_Atlans_A7-20250705-134249.csv'  # Replace with your actual CSV file path
    plot_2d_osm_altitude(csv_file, zoom=10)
    print("Plotting complete.")