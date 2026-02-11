#!/usr/bin/env python3
"""
CSV Point Converter: Swiss LV95/LN02 to WGS84

Converts CSV files containing Swiss coordinates (LV95 + LN02 heights)
to WGS84 with ellipsoidal heights.

Input format:
    #gcp name,E,N,H
    point1,2600000.0,1200000.0,500.0
    
Output format:
    gcp_name,lon,lat,alt
    point1,7.438632,46.951083,549.62

Usage:
    python convert_csv_points.py input.csv output.csv
    python convert_csv_points.py input.csv output.csv --method reframe
    python convert_csv_points.py input.csv output.csv --method pyproj
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Point:
    """A point with name and coordinates."""
    name: str
    easting: float
    northing: float
    height: float
    # Output coordinates (populated after transformation)
    lon: Optional[float] = None
    lat: Optional[float] = None
    alt: Optional[float] = None


class ReframeTransformer:
    """Transform points using SwissTopo Reframe API."""
    
    BASE_URL = "https://geodesy.geo.admin.ch/reframe"
    
    def __init__(
        self,
        timeout: float = 30,
        max_retries: int = 3,
        parallel_requests: int = 10
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.parallel_requests = parallel_requests
        self.session = requests.Session()
    
    def transform_single(self, e: float, n: float, h: float) -> tuple:
        """Transform a single point. Returns (lon, lat, alt, success)."""
        params = {
            'easting': f'{e:.6f}',
            'northing': f'{n:.6f}',
            'altitude': f'{h:.6f}',
            'format': 'json'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/lv95towgs84",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                lon = float(data['easting'])
                lat = float(data['northing'])
                alt = float(data['altitude'])
                return (lon, lat, alt, True)
                
            except Exception as ex:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    logger.warning(f"Failed to transform ({e}, {n}, {h}): {ex}")
                    return (None, None, None, False)
        
        return (None, None, None, False)
    
    def transform_points(self, points: list[Point]) -> list[Point]:
        """Transform multiple points using parallel requests."""
        logger.info(f"Transforming {len(points)} points via Reframe API "
                   f"(parallel_requests={self.parallel_requests})...")
        
        def transform_point(point: Point) -> Point:
            lon, lat, alt, success = self.transform_single(
                point.easting, point.northing, point.height
            )
            if success:
                point.lon = lon
                point.lat = lat
                point.alt = alt
            return point
        
        results = []
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            futures = {executor.submit(transform_point, p): p for p in points}
            
            for future in as_completed(futures):
                point = future.result()
                results.append(point)
                if point.lon is None:
                    failed += 1
        
        if failed > 0:
            logger.warning(f"Failed to transform {failed}/{len(points)} points")
        else:
            logger.info(f"Successfully transformed all {len(points)} points")
        
        # Preserve original order
        results.sort(key=lambda p: points.index(p) if p in points else 0)
        return results


class PyProjTransformer:
    """Transform points using pyproj."""
    
    def __init__(self):
        from pyproj import CRS, Transformer
        from pyproj.crs import CompoundCRS
        
        self.source_crs = CRS.from_epsg(2056)  # LV95
        self.target_crs = CRS.from_epsg(4979)  # WGS84 3D
        
        # Try to use compound CRS for proper height transformation
        try:
            source_compound = CompoundCRS(
                name="LV95 + LN02",
                components=[
                    CRS.from_epsg(2056),  # LV95
                    CRS.from_epsg(5728)   # LN02 (note: 5728 for heights, 5729 for datum)
                ]
            )
            
            self.transformer = Transformer.from_crs(
                source_compound, self.target_crs, always_xy=True
            )
            
            # Test if geoid is actually applied
            _, _, test_h = self.transformer.transform(2600000, 1200000, 500)
            if abs(test_h - 500) > 1.0:
                self.geoid_available = True
                logger.info("Using pyproj with CHGeo2004 geoid grid")
            else:
                self.geoid_available = False
                logger.warning("Geoid grid not available - using approximate model")
                self._setup_simple_transformer()
        except Exception as ex:
            logger.warning(f"Compound CRS failed: {ex} - using approximate model")
            self.geoid_available = False
            self._setup_simple_transformer()
    
    def _setup_simple_transformer(self):
        """Setup simple transformer with approximate geoid."""
        from pyproj import Transformer
        
        self.transformer = Transformer.from_crs(
            self.source_crs, self.target_crs, always_xy=True
        )
        
        # Approximate geoid model for Switzerland
        # N ≈ 49.5 + spatial variation
        self.approx_geoid_base = 49.5
        self.approx_geoid_de = 0.000015
        self.approx_geoid_dn = 0.000020
        self.approx_geoid_e0 = 2600000
        self.approx_geoid_n0 = 1200000
    
    def _get_geoid_undulation(self, e: float, n: float) -> float:
        """Get approximate geoid undulation."""
        return (
            self.approx_geoid_base +
            self.approx_geoid_de * (e - self.approx_geoid_e0) +
            self.approx_geoid_dn * (n - self.approx_geoid_n0)
        )
    
    def transform_points(self, points: list[Point]) -> list[Point]:
        """Transform multiple points using pyproj."""
        logger.info(f"Transforming {len(points)} points via pyproj...")
        
        for point in points:
            lon, lat, alt = self.transformer.transform(
                point.easting, point.northing, point.height
            )
            
            # Apply approximate geoid if needed
            if not self.geoid_available:
                geoid_n = self._get_geoid_undulation(point.easting, point.northing)
                alt = point.height + geoid_n
            
            point.lon = lon
            point.lat = lat
            point.alt = alt
        
        logger.info(f"Successfully transformed all {len(points)} points")
        return points


def read_csv(input_path: Path) -> list[Point]:
    """
    Read points from CSV file.
    
    Supports formats:
    - #gcp name,E,N,H (with comment header)
    - gcp_name,E,N,H (standard header)
    - name,easting,northing,height (verbose header)
    """
    points = []
    
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        # Detect delimiter and header
        sample = f.read(2048)
        f.seek(0)
        
        # Try to detect delimiter
        if '\t' in sample:
            delimiter = '\t'
        elif ';' in sample:
            delimiter = ';'
        else:
            delimiter = ','
        
        reader = csv.reader(f, delimiter=delimiter)
        
        # Read header
        header = next(reader)
        
        # Clean header (remove # prefix if present)
        header = [h.strip().lstrip('#').lower() for h in header]
        
        # Map column names to indices
        col_map = {}
        for i, col in enumerate(header):
            col_lower = col.lower()
            if col_lower in ('gcp name', 'gcp_name', 'name', 'point', 'id', 'label'):
                col_map['name'] = i
            elif col_lower in ('e', 'easting', 'x', 'east'):
                col_map['easting'] = i
            elif col_lower in ('n', 'northing', 'y', 'north'):
                col_map['northing'] = i
            elif col_lower in ('h', 'height', 'z', 'alt', 'altitude', 'elev', 'elevation'):
                col_map['height'] = i
        
        # Validate required columns
        required = ['name', 'easting', 'northing', 'height']
        missing = [c for c in required if c not in col_map]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {header}")
        
        # Read data rows
        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == '' for cell in row):
                continue  # Skip empty rows
            
            try:
                point = Point(
                    name=row[col_map['name']].strip(),
                    easting=float(row[col_map['easting']]),
                    northing=float(row[col_map['northing']]),
                    height=float(row[col_map['height']])
                )
                points.append(point)
            except (ValueError, IndexError) as ex:
                logger.warning(f"Skipping invalid row {row_num}: {row} ({ex})")
    
    logger.info(f"Read {len(points)} points from {input_path}")
    return points


def write_csv(points: list[Point], output_path: Path, precision: int = 8):
    """Write transformed points to CSV file."""
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['gcp_name', 'lon', 'lat', 'alt'])
        
        # Write data
        for point in points:
            if point.lon is not None:
                writer.writerow([
                    point.name,
                    f'{point.lon:.{precision}f}',
                    f'{point.lat:.{precision}f}',
                    f'{point.alt:.6f}'
                ])
            else:
                writer.writerow([point.name, 'ERROR', 'ERROR', 'ERROR'])
    
    logger.info(f"Wrote {len(points)} points to {output_path}")


def convert_csv(
    input_path: Path,
    output_path: Path,
    method: str = 'reframe',
    parallel_requests: int = 10
) -> list[Point]:
    """
    Convert CSV file from LV95/LN02 to WGS84.
    
    Args:
        input_path: Input CSV file path
        output_path: Output CSV file path
        method: 'reframe' (API) or 'pyproj' (local)
        parallel_requests: Number of parallel API requests (for reframe)
    
    Returns:
        List of transformed points
    """
    # Read input
    points = read_csv(input_path)
    
    if not points:
        logger.error("No valid points found in input file")
        return []
    
    # Transform
    if method == 'reframe':
        transformer = ReframeTransformer(parallel_requests=parallel_requests)
    else:
        transformer = PyProjTransformer()
    
    points = transformer.transform_points(points)
    
    # Write output
    write_csv(points, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRANSFORMATION RESULTS")
    print("=" * 70)
    print(f"{'GCP Name':<25} {'E':>12} {'N':>12} {'H':>10} -> "
          f"{'Lon':>12} {'Lat':>11} {'Alt':>10}")
    print("-" * 70)
    
    for p in points:
        if p.lon is not None:
            geoid_n = p.alt - p.height
            print(f"{p.name:<25} {p.easting:>12.2f} {p.northing:>12.2f} {p.height:>10.3f} -> "
                  f"{p.lon:>12.8f} {p.lat:>11.8f} {p.alt:>10.3f}")
        else:
            print(f"{p.name:<25} {p.easting:>12.2f} {p.northing:>12.2f} {p.height:>10.3f} -> "
                  f"{'FAILED':>12} {'':>11} {'':>10}")
    
    print("-" * 70)
    
    # Print geoid summary
    valid_points = [p for p in points if p.lon is not None]
    if valid_points:
        geoid_vals = [p.alt - p.height for p in valid_points]
        print(f"\nGeoid undulation (N): mean={np.mean(geoid_vals):.3f}m, "
              f"range=[{np.min(geoid_vals):.3f}, {np.max(geoid_vals):.3f}]m")
    
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Method: {method}")
    print("=" * 70)
    
    return points


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV points from Swiss LV95/LN02 to WGS84',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_csv_points.py input.csv output.csv
    python convert_csv_points.py input.csv output.csv --method pyproj
    python convert_csv_points.py gcps.csv gcps_wgs84.csv --parallel 20

Input CSV format:
    #gcp name,E,N,H
    point1,2600000.0,1200000.0,500.0
    point2,2600100.0,1200100.0,510.0

Output CSV format:
    gcp_name,lon,lat,alt
    point1,7.438632,46.951083,549.62
    point2,7.439958,46.951982,559.58
        """
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help='Input CSV file (LV95/LN02 coordinates)'
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Output CSV file (WGS84 coordinates)'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['reframe', 'pyproj'],
        default='reframe',
        help='Transformation method (default: reframe)'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=10,
        help='Number of parallel API requests for reframe (default: 10)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    convert_csv(
        args.input,
        args.output,
        method=args.method,
        parallel_requests=args.parallel
    )


if __name__ == '__main__':
    main()
