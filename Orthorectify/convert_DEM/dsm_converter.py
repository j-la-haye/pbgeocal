#!/usr/bin/env python3
"""
DSM Tile Converter: Swiss LV95/LN02 to WGS84/ETRS89

Converts GeoTiff DSM tiles using:
1. SwissTopo Reframe API (official high-accuracy transformation)
2. Pyproj (for comparison)

Evaluates transformation accuracy differences between methods.

"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from pyproj import CRS, Transformer
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration container."""
    input_dir: Path
    output_dir: Path
    source_planimetric: str
    source_altimetric: str
    target_crs: str
    transformation_method: str  # "auto", "reframe", or "pyproj"
    reframe_base_url: str
    reframe_format: str
    reframe_timeout: int
    reframe_max_retries: int
    reframe_retry_delay: float
    reframe_batch_size: int
    reframe_proxy: Optional[str]  # Optional proxy URL
    reframe_parallel_requests: int  # Number of concurrent API requests
    reframe_use_interpolation: bool  # Use interpolation for large rasters
    reframe_sample_spacing: int  # Sample spacing for interpolation
    resampling: str
    output_dtype: Optional[str]
    nodata: float
    compression: str
    compression_predictor: Optional[int] 
    num_workers: int
    comparison_sample_spacing: int
    evaluation_enabled: bool
    output_report: str
    output_csv: str
    percentiles: list = field(default_factory=lambda: [50, 90, 95, 99])

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            input_dir=Path(cfg['input_dir']),
            output_dir=Path(cfg['output_dir']),
            source_planimetric=cfg['source_crs']['planimetric'],
            source_altimetric=cfg['source_crs']['altimetric'],
            target_crs=cfg['target_crs'],
            transformation_method=cfg.get('transformation_method', 'pyproj'),
            reframe_base_url=cfg['reframe_api']['base_url'],
            reframe_format=cfg['reframe_api']['format'],
            reframe_timeout=cfg['reframe_api']['timeout'],
            reframe_max_retries=cfg['reframe_api']['max_retries'],
            reframe_retry_delay=cfg['reframe_api'].get('retry_delay', 1.0),
            reframe_batch_size=cfg['reframe_api']['batch_size'],
            reframe_proxy=cfg['reframe_api'].get('proxy', None),
            reframe_parallel_requests=cfg['reframe_api'].get('parallel_requests', 10),
            reframe_use_interpolation=cfg['reframe_api'].get('use_interpolation', True),
            reframe_sample_spacing=cfg['reframe_api'].get('sample_spacing', 50),
            resampling=cfg['processing']['resampling'],
            output_dtype=cfg['processing']['output_dtype'],
            nodata=cfg['processing']['nodata'],
            compression=cfg['processing']['compression'],
            compression_predictor=cfg['processing'].get('compression_predictor', None),
            num_workers=cfg['processing']['num_workers'],
            comparison_sample_spacing=cfg['processing']['comparison_sample_spacing'],
            evaluation_enabled=cfg['evaluation']['enabled'],
            output_report=cfg['evaluation']['output_report'],
            output_csv=cfg['evaluation']['output_csv'],
            percentiles=cfg['evaluation'].get('percentiles', [50, 90, 95, 99])
        )


class ReframeAPI:
    """SwissTopo Reframe API client for coordinate transformations with parallel support."""
    
    def __init__(self, config: Config):
        self.base_url = config.reframe_base_url
        self.timeout = config.reframe_timeout
        self.max_retries = config.reframe_max_retries
        self.retry_delay = config.reframe_retry_delay
        self.batch_size = config.reframe_batch_size
        self.parallel_requests = config.reframe_parallel_requests
        self.session = requests.Session()
        self._api_available = None  # Cache API availability
        
        # Configure proxy settings
        self._configure_proxy(config)
    
    def _configure_proxy(self, config):
        """
        Configure proxy settings for the session.
        
        Proxy can be set via:
        1. Config file (reframe_api.proxy)
        2. Environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
        3. System proxy settings
        """
        # Check for proxy in config
        proxy_url = getattr(config, 'reframe_proxy', None)
        
        if proxy_url:
            self.session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            logger.info(f"Using configured proxy: {proxy_url}")
        elif os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY'):
            # Let requests use environment variables automatically
            logger.info("Using proxy from environment variables")
        else:
            # Try to bypass proxy for Swiss government domains
            # This works if NO_PROXY is respected
            no_proxy = os.environ.get('NO_PROXY', '')
            if 'geo.admin.ch' not in no_proxy:
                logger.debug("Consider adding 'geo.admin.ch' to NO_PROXY if experiencing proxy issues")
    
    def check_api_available(self) -> bool:
        """Check if the Reframe API is reachable."""
        if self._api_available is not None:
            return self._api_available
        
        try:
            # Test with a simple request
            response = self.session.get(
                f"{self.base_url}/lv95towgs84",
                params={'easting': '2600000', 'northing': '1200000', 'altitude': '500', 'format': 'json'},
                timeout=10
            )
            self._api_available = response.status_code == 200
        except Exception:
            self._api_available = False
        
        if not self._api_available:
            logger.warning("Reframe API is not available - will use pyproj fallback")
        else:
            logger.info(f"Reframe API available (parallel_requests={self.parallel_requests})")
        
        return self._api_available
    
    def _transform_single_point(self, e: float, n: float, h: float) -> tuple:
        """
        Transform a single point via the API. Returns (lon, lat, h_ellip, success).
        
        This is the core function called by parallel workers.
        """
        if np.isnan(h):
            return (np.nan, np.nan, np.nan, False)
        
        params = {
            'easting': f'{e:.3f}',
            'northing': f'{n:.3f}',
            'altitude': f'{h:.3f}',
            'format': 'json'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/lv95towgs84",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                lon = float(data.get('easting', data.get('longitude')))
                lat = float(data.get('northing', data.get('latitude')))
                h_out = float(data.get('altitude', h))
                return (lon, lat, h_out, True)
                
            except requests.exceptions.RequestException as ex:
                delay = self.retry_delay * (2 ** attempt)
                
                if attempt < self.max_retries - 1:
                    if hasattr(ex, 'response') and ex.response is not None:
                        status = ex.response.status_code
                        if status in (429, 500, 502, 503, 504):
                            time.sleep(delay)
                            continue
                    else:
                        time.sleep(delay)
                        continue
        
        return (np.nan, np.nan, np.nan, False)
    
    def transform_points_parallel(
        self, 
        easting: np.ndarray, 
        northing: np.ndarray, 
        height: np.ndarray,
        progress_callback=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform points using parallel API requests.
        
        This is MUCH faster than sequential requests for large point sets.
        
        Args:
            easting: LV95 easting coordinates (E)
            northing: LV95 northing coordinates (N)
            height: LN02 heights (orthometric/geoid-based)
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Tuple of (longitude, latitude, ellipsoidal_height) in WGS84
        """
        n_points = len(easting)
        lon_out = np.full(n_points, np.nan)
        lat_out = np.full(n_points, np.nan)
        h_out = np.full(n_points, np.nan)
        
        # Filter valid points
        valid_mask = ~np.isnan(height)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return lon_out, lat_out, h_out
        
        logger.info(f"Transforming {n_valid} points with {self.parallel_requests} parallel requests...")
        
        # Create tasks for parallel execution
        tasks = [
            (i, easting[i], northing[i], height[i]) 
            for i in valid_indices
        ]
        
        completed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._transform_single_point, e, n, h): idx
                for idx, e, n, h in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    lon, lat, h_ellip, success = future.result()
                    if success:
                        lon_out[idx] = lon
                        lat_out[idx] = lat
                        h_out[idx] = h_ellip
                    else:
                        failed += 1
                except Exception as ex:
                    logger.warning(f"Error transforming point {idx}: {ex}")
                    failed += 1
                
                completed += 1
                if progress_callback and completed % 100 == 0:
                    progress_callback(completed, n_valid)
        
        if failed > 0:
            logger.warning(f"Failed to transform {failed}/{n_valid} points")
        
        return lon_out, lat_out, h_out
    
    def transform_points(
        self, 
        easting: np.ndarray, 
        northing: np.ndarray, 
        height: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform points from LV95/LN02 to ETRF93/WGS84 using Reframe API.
        
        Automatically uses parallel requests if parallel_requests > 1.
        
        Args:
            easting: LV95 easting coordinates (E)
            northing: LV95 northing coordinates (N)
            height: LN02 heights (orthometric/geoid-based)
        
        Returns:
            Tuple of (longitude, latitude, ellipsoidal_height) in ETRF93/WGS84
        """
        if self.parallel_requests > 1:
            return self.transform_points_parallel(easting, northing, height)
        else:
            return self._transform_points_sequential(easting, northing, height)
    
    def _transform_points_sequential(
        self, 
        easting: np.ndarray, 
        northing: np.ndarray, 
        height: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform points sequentially (original slow method).
        
        Use parallel_requests > 1 for faster processing.
        """
        n_points = len(easting)
        lon_out = np.full(n_points, np.nan)
        lat_out = np.full(n_points, np.nan)
        h_out = np.full(n_points, np.nan)
        
        # Track consecutive failures for early abort
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        for i in range(n_points):
            if np.isnan(height[i]):
                continue
            
            lon, lat, h_ellip, success = self._transform_single_point(
                easting[i], northing[i], height[i]
            )
            
            if success:
                lon_out[i] = lon
                lat_out[i] = lat
                h_out[i] = h_ellip
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Reframe API failing consistently after {i} points - aborting")
                    break
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Transformed {i + 1}/{n_points} points...")
        
        return lon_out, lat_out, h_out


class PyProjTransformer:
    """Pyproj-based coordinate transformation."""
    
    def __init__(self, config: Config):
        self.source_crs = CRS.from_string(config.source_planimetric)
        self.target_crs = CRS.from_string(config.target_crs)
        
        # Create transformer with grid-based transformation if available
        # This uses CHENyx06 grid for planimetric transformation
        self.transformer = Transformer.from_crs(
            self.source_crs,
            self.target_crs,
            always_xy=True
        )
        
        # For height transformation (LN02 geoid to ellipsoidal)
        # PyProj can use the CHGeo2004 geoid model if available
        self._setup_height_transformer(config)
    
    def _setup_height_transformer(self, config: Config):
        """Setup height transformation from LN02 to ellipsoidal heights."""
        # LN02 uses the CHGeo2004 geoid model
        # EPSG:5728 is LN02 (Swiss vertical datum)
        try:
            # Create a compound CRS for proper 3D transformation
            # LV95 + LN02 -> WGS84 3D
            # Method 1: Use CompoundCRS class (pyproj >= 3.0)
            from pyproj.crs import CompoundCRS
            
            source_compound = CompoundCRS(
                name="LV95 + LN02",
                components=[
                    CRS.from_epsg(2056),  # LV95 (horizontal)
                    CRS.from_epsg(5728)   # LN02 (vertical)
                ]
            )
            
            self.transformer_3d = Transformer.from_crs(
                source_compound,
                self.target_crs,
                always_xy=True
            )
            
            # Test if geoid grid is actually available
            test_h_in = 500.0
            _, _, test_h_out = self.transformer_3d.transform(2600000, 1200000, test_h_in)
            
            if abs(test_h_out - test_h_in) > 1.0:  # Swiss geoid ~49m, so any change > 1m means it's working
                self.use_compound = True
                self.geoid_available = True
                logger.info("Using compound CRS transformation with CHGeo2004 geoid (LV95+LN02 -> WGS84)")
            else:
                logger.warning("Compound CRS created but geoid grid not applied (missing ch_swisstopo grids)")
                logger.warning("Heights will be approximate. Install grids with: projsync --source-id ch_swisstopo")
                self.use_compound = False
                self.geoid_available = False
                
        except ImportError:
            logger.warning("CompoundCRS not available in this pyproj version")
            self.use_compound = False
            self.geoid_available = False
        except Exception as e:
            logger.warning(f"Could not create compound CRS transformer: {e}")
            self.use_compound = False
            self.geoid_available = False
        
        # If geoid not available, we'll need to apply approximate correction
        if not getattr(self, 'geoid_available', False):
            self._setup_approximate_geoid()
    
    def _setup_approximate_geoid(self):
        """
        Setup approximate geoid undulation for Switzerland.
        
        The Swiss geoid (CHGeo2004) varies from ~47m to ~53m across the country.
        This provides a simple linear approximation when the proper grid is unavailable.
        """
        # Approximate geoid undulation model for Switzerland
        # Based on CHGeo2004: N ≈ 49.5m + small spatial variation
        # 
        # More accurate: N varies roughly linearly with position
        # N ≈ 49.5 + 0.000015*(E - 2600000) + 0.000020*(N - 1200000)
        #
        # This gives ~1-2m accuracy across Switzerland vs ~49m systematic error without it
        
        self.approx_geoid_base = 49.5  # meters (approximate mean for Switzerland)
        self.approx_geoid_de = 0.000015  # m per m easting
        self.approx_geoid_dn = 0.000020  # m per m northing
        self.approx_geoid_e0 = 2600000
        self.approx_geoid_n0 = 1200000
        
        logger.info("Using approximate geoid model (~1-2m accuracy)")
        logger.info("For cm-level accuracy, install: projsync --source-id ch_swisstopo")
    
    def _get_approximate_geoid_undulation(self, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
        """Calculate approximate geoid undulation for given coordinates."""
        return (
            self.approx_geoid_base + 
            self.approx_geoid_de * (easting - self.approx_geoid_e0) +
            self.approx_geoid_dn * (northing - self.approx_geoid_n0)
        )
    
    def transform_points(
        self, 
        easting: np.ndarray, 
        northing: np.ndarray, 
        height: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform points from LV95/LN02 to target CRS using pyproj.
        
        Args:
            easting: LV95 easting coordinates
            northing: LV95 northing coordinates
            height: LN02 heights
        
        Returns:
            Tuple of (x, y, z) in target CRS
        """
        if self.use_compound and self.geoid_available:
            # Use compound CRS transformation for full 3D accuracy
            return self.transformer_3d.transform(easting, northing, height)
        else:
            # Transform planimetric coordinates
            x, y, _ = self.transformer.transform(easting, northing, height)
            
            # Apply geoid correction to heights
            if hasattr(self, 'approx_geoid_base'):
                # Use approximate geoid model
                geoid_n = self._get_approximate_geoid_undulation(easting, northing)
                z = height + geoid_n  # h_ellipsoidal = H_orthometric + N
            else:
                # No geoid correction available - heights unchanged (LESS ACCURATE)
                z = height
                
            return x, y, z


class DSMConverter:
    """Main DSM tile converter class."""
    
    def __init__(self, config: Config):
        self.config = config
        self.reframe = ReframeAPI(config)
        self.pyproj = PyProjTransformer(config)
        self.resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'lanczos': Resampling.lanczos
        }
    
    def convert_tile(
        self, 
        input_path: Path,
        use_reframe: bool = True
    ) -> dict:
        """
        Convert a single DSM tile.
        
        Args:
            input_path: Path to input GeoTiff
            use_reframe: If True, use Reframe API; otherwise use pyproj
        
        Returns:
            Dictionary with conversion statistics
        """
        output_path = self.config.output_dir / input_path.name
        suffix = "_reframe" if use_reframe else "_pyproj"
        output_path = output_path.with_stem(output_path.stem + suffix)
        
        stats = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'method': 'reframe' if use_reframe else 'pyproj',
            'success': False
        }
        
        try:
            with rasterio.open(input_path) as src:
                # Get source metadata
                src_crs = src.crs
                src_transform = src.transform
                src_data = src.read(1)
                src_nodata = src.nodata if src.nodata is not None else self.config.nodata
                
                # Create mask for valid data
                valid_mask = src_data != src_nodata
                
                # Get coordinates for all pixels
                rows, cols = np.meshgrid(
                    np.arange(src.height),
                    np.arange(src.width),
                    indexing='ij'
                )
                
                # Convert pixel coordinates to LV95
                xs, ys = rasterio.transform.xy(src_transform, rows, cols)
                easting = np.array(xs).flatten()
                northing = np.array(ys).flatten()
                height = src_data.flatten()
                
                # Transform coordinates
                transformer = self.reframe if use_reframe else self.pyproj
                
                if use_reframe:
                    if self.config.reframe_use_interpolation:
                        # Sample points and interpolate (fast, recommended for large tiles)
                        logger.info(f"    Using interpolation mode (sample_spacing={self.config.reframe_sample_spacing})")
                        lon, lat, h_ellip = self._transform_with_interpolation(
                            easting, northing, height, valid_mask.flatten(),
                            src.height, src.width, transformer
                        )
                    else:
                        # Transform all points directly (slower but exact)
                        # Uses parallel requests if parallel_requests > 1
                        n_valid = valid_mask.sum()
                        logger.info(f"    Transforming all {n_valid} valid pixels via API "
                                   f"(parallel_requests={self.config.reframe_parallel_requests})")
                        lon, lat, h_ellip = transformer.transform_points(
                            easting, northing, height
                        )
                else:
                    # For pyproj, transform all points directly (fast, local computation)
                    lon, lat, h_ellip = transformer.transform_points(
                        easting, northing, height
                    )
                
                # Reshape to grid
                lon_grid = lon.reshape(src.height, src.width)
                lat_grid = lat.reshape(src.height, src.width)
                h_grid = h_ellip.reshape(src.height, src.width)
                
                # Apply nodata mask
                h_grid[~valid_mask] = self.config.nodata
                
                # Calculate output bounds and transform
                valid_lon = lon_grid[valid_mask]
                valid_lat = lat_grid[valid_mask]
                
                if len(valid_lon) == 0:
                    raise ValueError("No valid data points in tile")
                
                dst_bounds = (
                    valid_lon.min(), valid_lat.min(),
                    valid_lon.max(), valid_lat.max()
                )
                
                # Calculate output transform and dimensions
                dst_crs = CRS.from_string(self.config.target_crs)
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs,
                    src.width, src.height,
                    *src.bounds
                )
                
                # Prepare output metadata
                dst_meta = src.meta.copy()
                dst_meta.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'nodata': self.config.nodata,
                    'compress': self.config.compression,
                    'predictor': self.config.compression_predictor
                })
                
                if self.config.output_dtype:
                    dst_meta['dtype'] = self.config.output_dtype
                
                # Create output with reprojected heights
                dst_data = np.full((dst_height, dst_width), self.config.nodata, 
                                   dtype=dst_meta['dtype'])
                
                reproject(
                    source=h_grid,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata=self.config.nodata,
                    dst_nodata=self.config.nodata,
                    resampling=self.resampling_methods.get(
                        self.config.resampling, Resampling.bilinear
                    )
                )
                
                # Write output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, 'w', **dst_meta) as dst:
                    dst.write(dst_data, 1)
                
                stats.update({
                    'success': True,
                    'src_bounds': src.bounds,
                    'dst_bounds': dst_bounds,
                    'src_crs': str(src_crs),
                    'dst_crs': str(dst_crs),
                    'width': dst_width,
                    'height': dst_height,
                    'valid_pixels': int(valid_mask.sum())
                })
                
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _transform_with_interpolation(
        self,
        easting: np.ndarray,
        northing: np.ndarray,
        height: np.ndarray,
        valid_mask: np.ndarray,
        nrows: int,
        ncols: int,
        transformer: ReframeAPI
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform coordinates using sample grid + interpolation.
        
        This reduces API calls by transforming a sparse grid and interpolating.
        Much faster for large rasters while maintaining accuracy for smooth geoid.
        """
        spacing = self.config.reframe_sample_spacing
        
        # Create sample indices
        row_idx = np.arange(0, nrows, spacing)
        col_idx = np.arange(0, ncols, spacing)
        
        # Ensure we include the last row/col
        if row_idx[-1] != nrows - 1:
            row_idx = np.append(row_idx, nrows - 1)
        if col_idx[-1] != ncols - 1:
            col_idx = np.append(col_idx, ncols - 1)
        
        n_sample_points = len(row_idx) * len(col_idx)
        logger.info(f"    Using interpolation: {n_sample_points} sample points "
                   f"(spacing={spacing}, grid={len(row_idx)}x{len(col_idx)})")
        
        # Get sample coordinates
        sample_rows, sample_cols = np.meshgrid(row_idx, col_idx, indexing='ij')
        sample_flat_idx = sample_rows.flatten() * ncols + sample_cols.flatten()
        
        sample_e = easting[sample_flat_idx]
        sample_n = northing[sample_flat_idx]
        sample_h = height[sample_flat_idx]
        
        # Transform sample points (this uses parallel requests if configured)
        sample_lon, sample_lat, sample_h_ellip = transformer.transform_points(
            sample_e, sample_n, sample_h
        )
        
        # Reshape for interpolation
        sample_lon = sample_lon.reshape(len(row_idx), len(col_idx))
        sample_lat = sample_lat.reshape(len(row_idx), len(col_idx))
        sample_h_ellip = sample_h_ellip.reshape(len(row_idx), len(col_idx))
        
        # Create interpolators
        interp_lon = RegularGridInterpolator(
            (row_idx, col_idx), sample_lon, 
            method='linear', bounds_error=False, fill_value=np.nan
        )
        interp_lat = RegularGridInterpolator(
            (row_idx, col_idx), sample_lat,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        
        # Height correction (geoid undulation) interpolation
        h_correction = sample_h_ellip - sample_h.reshape(len(row_idx), len(col_idx))
        interp_h_corr = RegularGridInterpolator(
            (row_idx, col_idx), h_correction,
            method='linear', bounds_error=False, fill_value=0
        )
        
        # Interpolate to full grid
        all_rows, all_cols = np.meshgrid(
            np.arange(nrows), np.arange(ncols), indexing='ij'
        )
        query_points = np.column_stack([all_rows.flatten(), all_cols.flatten()])
        
        lon_full = interp_lon(query_points)
        lat_full = interp_lat(query_points)
        h_corr_full = interp_h_corr(query_points)
        
        # Apply height correction to original heights
        h_ellip_full = height + h_corr_full
        
        return lon_full, lat_full, h_ellip_full
    
    def compare_transformations(
        self, 
        input_path: Path
    ) -> dict:
        """
        Compare Reframe and pyproj transformations for a single tile.
        
        Returns statistics on the differences.
        """
        stats = {
            'input_file': str(input_path),
            'success': False
        }
        
        try:
            with rasterio.open(input_path) as src:
                src_data = src.read(1)
                src_nodata = src.nodata if src.nodata is not None else self.config.nodata
                valid_mask = src_data != src_nodata
                
                # Sample points for comparison
                spacing = self.config.comparison_sample_spacing
                rows = np.arange(0, src.height, spacing)
                cols = np.arange(0, src.width, spacing)
                sample_rows, sample_cols = np.meshgrid(rows, cols, indexing='ij')
                
                # Get coordinates
                xs, ys = rasterio.transform.xy(
                    src.transform, sample_rows.flatten(), sample_cols.flatten()
                )
                easting = np.array(xs)
                northing = np.array(ys)
                
                # Get heights at sample points
                height = src_data[sample_rows.flatten().astype(int), 
                                  sample_cols.flatten().astype(int)]
                
                # Filter valid points
                valid_idx = height != src_nodata
                easting = easting[valid_idx]
                northing = northing[valid_idx]
                height = height[valid_idx]
                
                if len(easting) == 0:
                    raise ValueError("No valid sample points")
                
                # Transform with both methods
                logger.info(f"Transforming {len(easting)} sample points...")
                
                # PyProj transformation
                lon_pyproj, lat_pyproj, h_pyproj = self.pyproj.transform_points(
                    easting, northing, height
                )
                
                # Reframe transformation
                lon_reframe, lat_reframe, h_reframe = self.reframe.transform_points(
                    easting, northing, height
                )
                
                # Calculate differences
                # Convert to approximate meters for horizontal differences
                # At Swiss latitudes (~47°), 1 degree ≈ 111km lat, ~75km lon
                lat_mean = np.nanmean(lat_pyproj)
                m_per_deg_lat = 111000
                m_per_deg_lon = 111000 * np.cos(np.radians(lat_mean))
                
                d_lon = (lon_reframe - lon_pyproj) * m_per_deg_lon
                d_lat = (lat_reframe - lat_pyproj) * m_per_deg_lat
                d_h = h_reframe - h_pyproj
                
                # Horizontal distance
                d_horiz = np.sqrt(d_lon**2 + d_lat**2)
                
                # Filter NaN values
                valid_diff = ~(np.isnan(d_lon) | np.isnan(d_lat) | np.isnan(d_h))
                d_lon = d_lon[valid_diff]
                d_lat = d_lat[valid_diff]
                d_h = d_h[valid_diff]
                d_horiz = d_horiz[valid_diff]
                
                # Compute statistics
                stats.update({
                    'success': True,
                    'n_sample_points': len(easting),
                    'n_valid_comparisons': int(valid_diff.sum()),
                    'horizontal': {
                        'mean_m': float(np.mean(d_horiz)),
                        'std_m': float(np.std(d_horiz)),
                        'min_m': float(np.min(d_horiz)),
                        'max_m': float(np.max(d_horiz)),
                        'percentiles': {
                            str(p): float(np.percentile(d_horiz, p))
                            for p in self.config.percentiles
                        }
                    },
                    'vertical': {
                        'mean_m': float(np.mean(d_h)),
                        'std_m': float(np.std(d_h)),
                        'min_m': float(np.min(d_h)),
                        'max_m': float(np.max(d_h)),
                        'percentiles': {
                            str(p): float(np.percentile(d_h, p))
                            for p in self.config.percentiles
                        }
                    },
                    'easting_diff': {
                        'mean_m': float(np.mean(d_lon)),
                        'std_m': float(np.std(d_lon))
                    },
                    'northing_diff': {
                        'mean_m': float(np.mean(d_lat)),
                        'std_m': float(np.std(d_lat))
                    }
                })
                
        except Exception as e:
            logger.error(f"Error comparing transformations for {input_path}: {e}")
            stats['error'] = str(e)
        
        return stats


def process_all_tiles(config: Config) -> dict:
    """Process all tiles in the input directory."""
    
    # Find all GeoTiff files
    input_files = list(config.input_dir.glob('*.tif')) + \
                  list(config.input_dir.glob('*.tiff')) + \
                  list(config.input_dir.glob('*.TIF'))
    
    if not input_files:
        logger.warning(f"No GeoTiff files found in {config.input_dir}")
        return {}
    
    logger.info(f"Found {len(input_files)} tiles to process")
    logger.info(f"Transformation method: {config.transformation_method}")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = DSMConverter(config)
    
    # Determine which methods to use
    use_reframe = config.transformation_method in ('auto', 'reframe')
    use_pyproj = config.transformation_method in ('auto', 'pyproj')
    
    # Check Reframe API availability if needed
    reframe_available = False
    if use_reframe:
        reframe_available = converter.reframe.check_api_available()
        if not reframe_available:
            logger.warning("Reframe API unavailable")
            if config.transformation_method == 'reframe':
                logger.error("Reframe API required but not available - aborting")
                return {}
            else:
                logger.warning("Falling back to pyproj only")
                use_reframe = False
                use_pyproj = True
    
    # Process tiles and collect statistics
    all_stats = {
        'tiles_pyproj': [],
        'tiles_reframe': [],
        'comparisons': [],
        'summary': {},
        'config': {
            'transformation_method': config.transformation_method,
            'target_crs': config.target_crs,
            'reframe_used': use_reframe and reframe_available,
            'pyproj_used': use_pyproj
        }
    }
    
    # Process each tile
    for input_file in tqdm(input_files, desc="Processing tiles"):
        logger.info(f"Processing {input_file.name}")
        
        # Convert with pyproj
        if use_pyproj:
            logger.info(f"  Converting with pyproj...")
            pyproj_stats = converter.convert_tile(input_file, use_reframe=False)
            all_stats['tiles_pyproj'].append(pyproj_stats)
            if pyproj_stats['success']:
                logger.info(f"  ✓ PyProj output: {pyproj_stats['output_file']}")
            else:
                logger.error(f"  ✗ PyProj conversion failed: {pyproj_stats.get('error', 'unknown')}")
        
        # Convert with Reframe (in auto mode, produce both outputs for comparison)
        if use_reframe and reframe_available:
            logger.info(f"  Converting with Reframe API...")
            reframe_stats = converter.convert_tile(input_file, use_reframe=True)
            all_stats['tiles_reframe'].append(reframe_stats)
            if reframe_stats['success']:
                logger.info(f"  ✓ Reframe output: {reframe_stats['output_file']}")
            else:
                logger.error(f"  ✗ Reframe conversion failed: {reframe_stats.get('error', 'unknown')}")
        
        # Compare transformations if evaluation is enabled AND both methods produced outputs
        if config.evaluation_enabled and use_reframe and use_pyproj and reframe_available:
            comparison = converter.compare_transformations(input_file)
            all_stats['comparisons'].append(comparison)
    
    # Compute summary statistics across all tiles
    if all_stats['comparisons']:
        valid_comparisons = [c for c in all_stats['comparisons'] if c.get('success')]
        
        if valid_comparisons:
            # Aggregate horizontal differences
            all_horiz_means = [c['horizontal']['mean_m'] for c in valid_comparisons]
            all_vert_means = [c['vertical']['mean_m'] for c in valid_comparisons]
            
            all_stats['summary'] = {
                'total_tiles': len(input_files),
                'successful_pyproj': len([t for t in all_stats['tiles_pyproj'] if t.get('success')]),
                'successful_reframe': len([t for t in all_stats['tiles_reframe'] if t.get('success')]),
                'successful_comparisons': len(valid_comparisons),
                'mean_horizontal_diff_m': float(np.mean(all_horiz_means)),
                'std_horizontal_diff_m': float(np.std(all_horiz_means)),
                'mean_vertical_diff_m': float(np.mean(all_vert_means)),
                'std_vertical_diff_m': float(np.std(all_vert_means)),
                'notes': [
                    "Positive vertical diff = Reframe height > pyproj height",
                    "Differences primarily due to geoid model accuracy",
                    "Reframe uses official SwissTopo transformation parameters",
                    "PyProj accuracy depends on available grid files (CHENyx06, CHGeo2004)",
                    "Both _pyproj.tif and _reframe.tif outputs saved for direct comparison"
                ]
            }
            
            logger.info("\n" + "="*60)
            logger.info("TRANSFORMATION COMPARISON SUMMARY")
            logger.info("="*60)
            logger.info(f"Tiles processed: {len(input_files)}")
            logger.info(f"PyProj outputs:  {all_stats['summary']['successful_pyproj']}")
            logger.info(f"Reframe outputs: {all_stats['summary']['successful_reframe']}")
            logger.info(f"Mean horizontal difference: {all_stats['summary']['mean_horizontal_diff_m']:.4f} m")
            logger.info(f"Mean vertical difference: {all_stats['summary']['mean_vertical_diff_m']:.4f} m")
            logger.info("="*60)
    else:
        # Summary for single-method processing
        all_stats['summary'] = {
            'total_tiles': len(input_files),
            'successful_pyproj': len([t for t in all_stats['tiles_pyproj'] if t.get('success')]),
            'successful_reframe': len([t for t in all_stats['tiles_reframe'] if t.get('success')]),
            'method': config.transformation_method,
            'notes': []
        }
        
        if not use_reframe:
            all_stats['summary']['notes'].append("Reframe comparison disabled or API unavailable")
            all_stats['summary']['notes'].append("Using pyproj with approximate geoid model if grids not installed")
            all_stats['summary']['notes'].append("For full accuracy, install: projsync --source-id ch_swisstopo")
    
    # Save reports
    report_path = config.output_dir / config.output_report
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Saved detailed report to {report_path}")
    
    # CSV summary for tiles
    csv_data = []
    
    # Add pyproj tiles
    for tile in all_stats['tiles_pyproj']:
        if tile.get('success'):
            csv_data.append({
                'file': Path(tile['input_file']).name,
                'method': 'pyproj',
                'output_file': Path(tile['output_file']).name,
                'src_crs': tile.get('src_crs', ''),
                'dst_crs': tile.get('dst_crs', ''),
                'valid_pixels': tile.get('valid_pixels', 0)
            })
    
    # Add reframe tiles
    for tile in all_stats['tiles_reframe']:
        if tile.get('success'):
            csv_data.append({
                'file': Path(tile['input_file']).name,
                'method': 'reframe',
                'output_file': Path(tile['output_file']).name,
                'src_crs': tile.get('src_crs', ''),
                'dst_crs': tile.get('dst_crs', ''),
                'valid_pixels': tile.get('valid_pixels', 0)
            })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = config.output_dir / config.output_csv
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved tile statistics to {csv_path}")
    
    # Additional comparison CSV if comparisons exist
    if all_stats['comparisons']:
        comp_csv_data = []
        for comp in all_stats['comparisons']:
            if comp.get('success'):
                comp_csv_data.append({
                    'file': Path(comp['input_file']).name,
                    'n_points': comp['n_valid_comparisons'],
                    'horiz_mean_m': comp['horizontal']['mean_m'],
                    'horiz_std_m': comp['horizontal']['std_m'],
                    'horiz_max_m': comp['horizontal']['max_m'],
                    'vert_mean_m': comp['vertical']['mean_m'],
                    'vert_std_m': comp['vertical']['std_m'],
                    'vert_max_m': comp['vertical']['max_m']
                })
        
        if comp_csv_data:
            df_comp = pd.DataFrame(comp_csv_data)
            comp_csv_path = config.output_dir / "comparison_stats.csv"
            df_comp.to_csv(comp_csv_path, index=False)
            logger.info(f"Saved comparison statistics to {comp_csv_path}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert Swiss DSM tiles from LV95/LN02 to WGS84/ETRS89'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['auto', 'reframe', 'pyproj'],
        help='Override transformation method from config'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    #define config path for testing
    args.config = 'Orthorectify/convert_DEM/config.yaml'
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override method if specified
    if args.method:
        config.transformation_method = args.method
    
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Target CRS: {config.target_crs}")
    logger.info(f"Transformation method: {config.transformation_method}")
    
    # Process all tiles
    stats = process_all_tiles(config)
    
    return stats


if __name__ == '__main__':
    main()
