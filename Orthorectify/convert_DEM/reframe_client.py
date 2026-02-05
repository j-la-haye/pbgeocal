#!/usr/bin/env python3
"""
SwissTopo Reframe API Client

Accurate implementation of the SwissTopo REFRAME web service for
coordinate transformations in Switzerland.

API Documentation: https://www.swisstopo.admin.ch/en/rest-api-reframe

The REFRAME service provides:
- Planimetric transformations (LV03 <-> LV95 <-> ETRF93/WGS84)
- Altimetric transformations (LN02 <-> LHN95 <-> Ellipsoidal)
- Combined 3D transformations
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class ReframeFormat(Enum):
    """Available transformation formats in the Reframe API."""
    LV03_TO_LV95 = "lv03tolv95"
    LV95_TO_LV03 = "lv95tolv03"
    LV03_TO_ETRF93 = "lv03toetrf93"
    LV95_TO_ETRF93 = "lv95toetrf93"
    ETRF93_TO_LV03 = "etrf93tolv03"
    ETRF93_TO_LV95 = "etrf93tolv95"
    LV03_TO_WGS84 = "lv03towgs84"
    LV95_TO_WGS84 = "lv95towgs84"
    WGS84_TO_LV03 = "wgs84tolv03"
    WGS84_TO_LV95 = "wgs84tolv95"


class AltimetricFrame(Enum):
    """Altimetric reference frames."""
    LN02 = "ln02"      # Levelling Network 1902 (geoid-based)
    LHN95 = "lhn95"    # Levelling Height Network 1995
    ELLIPSOIDAL = "ellipsoidal"  # GRS80/WGS84 ellipsoid


@dataclass
class ReframeResult:
    """Result of a Reframe transformation."""
    easting: float      # Output X/E/Longitude
    northing: float     # Output Y/N/Latitude
    altitude: float     # Output height
    source_frame: str   # Source reference frame
    target_frame: str   # Target reference frame
    success: bool = True
    error_message: Optional[str] = None


class SwissTopoReframeClient:
    """
    Client for the SwissTopo REFRAME coordinate transformation service.
    
    This service provides high-accuracy transformations using official
    Swiss geodetic parameters and models:
    
    - CHENyx06: Planimetric transformation grid (LV03 <-> LV95/ETRF93)
    - CHGeo2004: Geoid model for height transformations (LN02 <-> ellipsoidal)
    - FINELTRA: Fine transformation for cadastral accuracy
    
    API Base URL: https://geodesy.geo.admin.ch/reframe/
    
    Example API call:
        GET https://geodesy.geo.admin.ch/reframe/lv95towgs84?
            easting=2600000&northing=1200000&altitude=500&format=json
    """
    
    BASE_URL = "https://geodesy.geo.admin.ch/reframe"
    
    def __init__(
        self,
        timeout: float = 30,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        """
        Initialize the Reframe API client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'DSM-Converter/1.0'
        })
    
    def transform_point(
        self,
        easting: float,
        northing: float,
        altitude: float,
        transformation: ReframeFormat = ReframeFormat.LV95_TO_WGS84
    ) -> ReframeResult:
        """
        Transform a single point using the Reframe API.
        
        Args:
            easting: Input easting/X/longitude
            northing: Input northing/Y/latitude
            altitude: Input altitude/height
            transformation: Transformation type (e.g., LV95_TO_WGS84)
        
        Returns:
            ReframeResult with transformed coordinates
        """
        params = {
            'easting': f'{easting:.6f}',
            'northing': f'{northing:.6f}',
            'altitude': f'{altitude:.6f}',
            'format': 'json'
        }
        
        url = f"{self.BASE_URL}/{transformation.value}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                return ReframeResult(
                    easting=float(data['easting']),
                    northing=float(data['northing']),
                    altitude=float(data['altitude']),
                    source_frame=transformation.value.split('to')[0],
                    target_frame=transformation.value.split('to')[1]
                )
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 400:
                    # Bad request - likely coordinates out of bounds
                    return ReframeResult(
                        easting=np.nan,
                        northing=np.nan,
                        altitude=np.nan,
                        source_frame=transformation.value.split('to')[0],
                        target_frame=transformation.value.split('to')[1],
                        success=False,
                        error_message=f"Coordinates out of valid range: {e}"
                    )
                raise
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    return ReframeResult(
                        easting=np.nan,
                        northing=np.nan,
                        altitude=np.nan,
                        source_frame=transformation.value.split('to')[0],
                        target_frame=transformation.value.split('to')[1],
                        success=False,
                        error_message=str(e)
                    )
    
    def transform_points_lv95_to_wgs84(
        self,
        easting: np.ndarray,
        northing: np.ndarray,
        altitude: np.ndarray,
        progress_callback=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform arrays of points from LV95/LN02 to WGS84.
        
        This includes:
        - Planimetric: LV95 (EPSG:2056) -> WGS84 (EPSG:4326)
        - Altimetric: LN02 (geoid) -> Ellipsoidal heights (WGS84)
        
        Args:
            easting: LV95 easting coordinates (E)
            northing: LV95 northing coordinates (N)
            altitude: LN02 heights (orthometric)
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Tuple of (longitude, latitude, ellipsoidal_height)
        """
        n_points = len(easting)
        lon_out = np.full(n_points, np.nan)
        lat_out = np.full(n_points, np.nan)
        h_out = np.full(n_points, np.nan)
        
        for i in range(n_points):
            if np.isnan(altitude[i]):
                continue
            
            result = self.transform_point(
                easting[i],
                northing[i],
                altitude[i],
                ReframeFormat.LV95_TO_WGS84
            )
            
            if result.success:
                lon_out[i] = result.easting
                lat_out[i] = result.northing
                h_out[i] = result.altitude
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, n_points)
        
        return lon_out, lat_out, h_out
    
    def transform_points_wgs84_to_lv95(
        self,
        longitude: np.ndarray,
        latitude: np.ndarray,
        altitude: np.ndarray,
        progress_callback=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform arrays of points from WGS84 to LV95/LN02.
        
        Args:
            longitude: WGS84 longitude
            latitude: WGS84 latitude
            altitude: Ellipsoidal heights (WGS84)
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Tuple of (easting, northing, ln02_height)
        """
        n_points = len(longitude)
        e_out = np.full(n_points, np.nan)
        n_out = np.full(n_points, np.nan)
        h_out = np.full(n_points, np.nan)
        
        for i in range(n_points):
            if np.isnan(altitude[i]):
                continue
            
            result = self.transform_point(
                longitude[i],
                latitude[i],
                altitude[i],
                ReframeFormat.WGS84_TO_LV95
            )
            
            if result.success:
                e_out[i] = result.easting
                n_out[i] = result.northing
                h_out[i] = result.altitude
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, n_points)
        
        return e_out, n_out, h_out
    
    def get_geoid_undulation(
        self,
        easting: float,
        northing: float
    ) -> float:
        """
        Get the geoid undulation (N) at a point.
        
        The geoid undulation is the difference between ellipsoidal height
        and orthometric (LN02) height: N = h_ellipsoidal - H_orthometric
        
        This is derived from the CHGeo2004 geoid model.
        
        Args:
            easting: LV95 easting
            northing: LV95 northing
        
        Returns:
            Geoid undulation in meters
        """
        # Transform with a reference height
        ref_height = 0.0
        result = self.transform_point(
            easting, northing, ref_height,
            ReframeFormat.LV95_TO_WGS84
        )
        
        if result.success:
            # The difference is the geoid undulation
            return result.altitude - ref_height
        else:
            return np.nan
    
    def validate_coordinates(
        self,
        easting: float,
        northing: float
    ) -> bool:
        """
        Check if coordinates are within valid Swiss bounds.
        
        LV95 bounds (approximate):
        - Easting: 2,485,000 - 2,834,000
        - Northing: 1,074,000 - 1,296,000
        """
        return (
            2_485_000 <= easting <= 2_834_000 and
            1_074_000 <= northing <= 1_296_000
        )


class GeoidInterpolator:
    """
    Efficient geoid undulation interpolator using sparse API calls.
    
    This class builds a local geoid model by sampling the Reframe API
    at grid points and interpolating for full-resolution data.
    """
    
    def __init__(self, reframe_client: SwissTopoReframeClient):
        self.client = reframe_client
        self._grid_cache = {}
    
    def build_geoid_grid(
        self,
        e_min: float,
        e_max: float,
        n_min: float,
        n_max: float,
        grid_spacing: float = 1000.0  # meters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a geoid undulation grid over an area.
        
        Args:
            e_min, e_max: Easting bounds (LV95)
            n_min, n_max: Northing bounds (LV95)
            grid_spacing: Grid point spacing in meters
        
        Returns:
            Tuple of (easting_grid, northing_grid, undulation_grid)
        """
        # Create grid
        e_grid = np.arange(e_min, e_max + grid_spacing, grid_spacing)
        n_grid = np.arange(n_min, n_max + grid_spacing, grid_spacing)
        
        E, N = np.meshgrid(e_grid, n_grid)
        undulation = np.zeros_like(E)
        
        logger.info(f"Building geoid grid: {len(e_grid)}x{len(n_grid)} = {E.size} points")
        
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                undulation[i, j] = self.client.get_geoid_undulation(E[i, j], N[i, j])
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{E.shape[0]} rows")
        
        return e_grid, n_grid, undulation
    
    def interpolate_undulation(
        self,
        easting: np.ndarray,
        northing: np.ndarray,
        e_grid: np.ndarray,
        n_grid: np.ndarray,
        undulation_grid: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate geoid undulation at arbitrary points.
        """
        from scipy.interpolate import RegularGridInterpolator
        
        interp = RegularGridInterpolator(
            (n_grid, e_grid),
            undulation_grid,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        points = np.column_stack([northing, easting])
        return interp(points)


def demo():
    """Demonstrate the Reframe API client."""
    client = SwissTopoReframeClient()
    
    # Test point: Bern (approximate)
    e_bern = 2_600_000  # LV95 easting
    n_bern = 1_200_000  # LV95 northing
    h_bern = 540.0      # LN02 height (meters)
    
    print("SwissTopo Reframe API Demo")
    print("="*50)
    print(f"Input (LV95/LN02):")
    print(f"  Easting:  {e_bern:,.1f} m")
    print(f"  Northing: {n_bern:,.1f} m")
    print(f"  Height:   {h_bern:.1f} m (LN02)")
    print()
    
    result = client.transform_point(e_bern, n_bern, h_bern, ReframeFormat.LV95_TO_WGS84)
    
    if result.success:
        print(f"Output (WGS84):")
        print(f"  Longitude: {result.easting:.8f}°")
        print(f"  Latitude:  {result.northing:.8f}°")
        print(f"  Height:    {result.altitude:.3f} m (ellipsoidal)")
        print()
        print(f"Geoid undulation: {result.altitude - h_bern:.3f} m")
    else:
        print(f"Transformation failed: {result.error_message}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
