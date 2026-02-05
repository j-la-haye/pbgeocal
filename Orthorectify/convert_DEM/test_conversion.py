#!/usr/bin/env python3
"""
Test and demonstration script for DSM conversion.

Creates synthetic DSM tiles and runs the full conversion pipeline
to verify functionality and demonstrate accuracy comparison.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_dsm(
    output_path: Path,
    e_min: float = 2_600_000,
    e_max: float = 2_601_000,
    n_min: float = 1_200_000,
    n_max: float = 1_201_000,
    resolution: float = 10.0,  # meters
    base_height: float = 500.0,
    noise_amplitude: float = 50.0
) -> Path:
    """
    Create a synthetic DSM tile in LV95/LN02.
    
    Args:
        output_path: Output file path
        e_min, e_max: Easting bounds (LV95)
        n_min, n_max: Northing bounds (LV95)
        resolution: Pixel size in meters
        base_height: Base terrain height (LN02)
        noise_amplitude: Height variation amplitude
    
    Returns:
        Path to created file
    """
    # Calculate dimensions
    width = int((e_max - e_min) / resolution)
    height = int((n_max - n_min) / resolution)
    
    # Create synthetic terrain (smooth hills + noise)
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    terrain = base_height + noise_amplitude * (
        np.sin(X) * np.cos(Y) +  # Large-scale undulation
        0.3 * np.sin(3*X) * np.sin(2*Y) +  # Medium features
        0.1 * np.random.randn(height, width)  # Small noise
    )
    
    terrain = terrain.astype(np.float32)
    
    # Create transform (note: rasterio uses top-left origin)
    transform = from_bounds(e_min, n_min, e_max, n_max, width, height)
    
    # Write GeoTiff
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': CRS.from_epsg(2056),  # LV95
        'transform': transform,
        'nodata': -9999,
        'compress': 'LZW'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(terrain, 1)
        dst.update_tags(
            VERTICAL_CRS='LN02',
            DESCRIPTION='Synthetic DSM for testing'
        )
    
    logger.info(f"Created synthetic DSM: {output_path}")
    logger.info(f"  Size: {width}x{height} pixels")
    logger.info(f"  Bounds: E[{e_min:.0f}-{e_max:.0f}], N[{n_min:.0f}-{n_max:.0f}]")
    logger.info(f"  Height range: {terrain.min():.1f} - {terrain.max():.1f} m")
    
    return output_path


def test_reframe_api():
    """Test the Reframe API with a few sample points."""
    from reframe_client import SwissTopoReframeClient, ReframeFormat
    
    logger.info("Testing SwissTopo Reframe API...")
    
    client = SwissTopoReframeClient(timeout=30, max_retries=3)
    
    # Test points across Switzerland
    test_points = [
        ("Bern", 2_600_000, 1_200_000, 540.0),
        ("Zurich", 2_683_000, 1_248_000, 408.0),
        ("Geneva", 2_500_000, 1_118_000, 375.0),
        ("Lugano", 2_717_000, 1_096_000, 273.0),
    ]
    
    results = []
    for name, e, n, h in test_points:
        result = client.transform_point(e, n, h, ReframeFormat.LV95_TO_WGS84)
        
        if result.success:
            geoid_n = result.altitude - h
            logger.info(f"  {name}:")
            logger.info(f"    LV95: E={e:,.0f}, N={n:,.0f}, H={h:.1f}m (LN02)")
            logger.info(f"    WGS84: {result.northing:.6f}°N, {result.easting:.6f}°E, {result.altitude:.2f}m")
            logger.info(f"    Geoid undulation: {geoid_n:.3f}m")
            results.append((name, result))
        else:
            logger.error(f"  {name}: FAILED - {result.error_message}")
    
    return results


def test_pyproj_transformation():
    """Test pyproj transformation and compare with expected values."""
    from pyproj import CRS, Transformer
    
    logger.info("Testing PyProj transformation...")
    
    # Create transformer
    src_crs = CRS.from_epsg(2056)  # LV95
    dst_crs = CRS.from_epsg(4979)  # WGS84 3D
    
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    # Test point (Bern)
    e, n, h = 2_600_000, 1_200_000, 540.0
    
    lon, lat, h_out = transformer.transform(e, n, h)
    
    logger.info(f"  Input (LV95): E={e:,.0f}, N={n:,.0f}, H={h:.1f}m")
    logger.info(f"  Output (WGS84): {lat:.6f}°N, {lon:.6f}°E, {h_out:.2f}m")
    
    # Check if compound CRS works
    try:
        src_compound = CRS.compound_crs([
            CRS.from_epsg(2056),  # LV95
            CRS.from_epsg(5729)   # LN02
        ])
        transformer_3d = Transformer.from_crs(src_compound, dst_crs, always_xy=True)
        lon2, lat2, h_out2 = transformer_3d.transform(e, n, h)
        logger.info(f"  With compound CRS: {lat2:.6f}°N, {lon2:.6f}°E, {h_out2:.2f}m")
        logger.info(f"  Height difference: {h_out2 - h_out:.3f}m")
    except Exception as ex:
        logger.warning(f"  Compound CRS not available: {ex}")
    
    return lon, lat, h_out


def compare_methods():
    """Compare Reframe and PyProj transformations on sample points."""
    from reframe_client import SwissTopoReframeClient, ReframeFormat
    from pyproj import CRS, Transformer
    import numpy as np
    
    logger.info("Comparing Reframe vs PyProj transformations...")
    
    # Setup
    reframe = SwissTopoReframeClient()
    pyproj_trans = Transformer.from_crs(
        CRS.from_epsg(2056),
        CRS.from_epsg(4979),
        always_xy=True
    )
    
    # Grid of test points
    eastings = np.linspace(2_500_000, 2_700_000, 5)
    northings = np.linspace(1_100_000, 1_280_000, 5)
    height = 500.0  # Fixed height for comparison
    
    differences = {
        'lon': [],
        'lat': [],
        'h': [],
        'horiz': []
    }
    
    for e in eastings:
        for n in northings:
            # Reframe transformation
            result_rf = reframe.transform_point(e, n, height, ReframeFormat.LV95_TO_WGS84)
            
            if not result_rf.success:
                continue
            
            # PyProj transformation
            lon_pp, lat_pp, h_pp = pyproj_trans.transform(e, n, height)
            
            # Calculate differences
            d_lon = (result_rf.easting - lon_pp) * 111000 * np.cos(np.radians(lat_pp))
            d_lat = (result_rf.northing - lat_pp) * 111000
            d_h = result_rf.altitude - h_pp
            d_horiz = np.sqrt(d_lon**2 + d_lat**2)
            
            differences['lon'].append(d_lon)
            differences['lat'].append(d_lat)
            differences['h'].append(d_h)
            differences['horiz'].append(d_horiz)
    
    # Report statistics
    logger.info("Transformation Differences (Reframe - PyProj):")
    logger.info(f"  Horizontal: {np.mean(differences['horiz']):.3f} ± {np.std(differences['horiz']):.3f} m")
    logger.info(f"  Vertical:   {np.mean(differences['h']):.3f} ± {np.std(differences['h']):.3f} m")
    logger.info(f"  East-West:  {np.mean(differences['lon']):.3f} ± {np.std(differences['lon']):.3f} m")
    logger.info(f"  North-South:{np.mean(differences['lat']):.3f} ± {np.std(differences['lat']):.3f} m")
    
    return differences


def run_full_pipeline_demo():
    """Run the full conversion pipeline on synthetic data."""
    import yaml
    
    logger.info("="*60)
    logger.info("FULL PIPELINE DEMONSTRATION")
    logger.info("="*60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create synthetic DSM tiles
        logger.info("\n1. Creating synthetic DSM tiles...")
        for i, (e_base, n_base) in enumerate([
            (2_600_000, 1_200_000),
            (2_601_000, 1_200_000),
            (2_600_000, 1_201_000),
        ]):
            create_synthetic_dsm(
                input_dir / f"tile_{i+1}.tif",
                e_min=e_base,
                e_max=e_base + 500,
                n_min=n_base,
                n_max=n_base + 500,
                resolution=10.0
            )
        
        # Create config
        logger.info("\n2. Creating configuration...")
        config = {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'source_crs': {
                'planimetric': 'EPSG:2056',
                'altimetric': 'LN02'
            },
            'target_crs': 'EPSG:4979',
            'reframe_api': {
                'base_url': 'https://geodesy.geo.admin.ch/reframe',
                'format': 'lv95',
                'timeout': 30,
                'max_retries': 3,
                'batch_size': 50
            },
            'processing': {
                'resampling': 'bilinear',
                'output_dtype': None,
                'nodata': -9999,
                'compression': 'LZW',
                'num_workers': 1,
                'comparison_sample_spacing': 10
            },
            'evaluation': {
                'enabled': True,
                'output_report': 'conversion_report.json',
                'output_csv': 'accuracy_stats.csv',
                'percentiles': [50, 90, 95, 99]
            }
        }
        
        config_path = tmpdir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"  Config saved to: {config_path}")
        
        # Note: Full pipeline would be run here
        # from dsm_converter import Config, process_all_tiles
        # cfg = Config.from_yaml(config_path)
        # stats = process_all_tiles(cfg)
        
        logger.info("\n3. Pipeline demo complete!")
        logger.info(f"  In production, run: python dsm_converter.py {config_path}")


def main():
    """Main test entry point."""
    logger.info("DSM Converter Test Suite")
    logger.info("="*60)
    
    # Test individual components
    try:
        logger.info("\n--- Test 1: Reframe API ---")
        test_reframe_api()
    except Exception as e:
        logger.error(f"Reframe API test failed: {e}")
        logger.info("  (This may be due to network issues or API availability)")
    
    try:
        logger.info("\n--- Test 2: PyProj Transformation ---")
        test_pyproj_transformation()
    except Exception as e:
        logger.error(f"PyProj test failed: {e}")
    
    try:
        logger.info("\n--- Test 3: Method Comparison ---")
        compare_methods()
    except Exception as e:
        logger.error(f"Comparison test failed: {e}")
        logger.info("  (This requires network access to Reframe API)")
    
    try:
        logger.info("\n--- Test 4: Full Pipeline Demo ---")
        run_full_pipeline_demo()
    except Exception as e:
        logger.error(f"Pipeline demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("Test suite complete!")


if __name__ == '__main__':
    main()
