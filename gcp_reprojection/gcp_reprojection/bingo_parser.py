"""
BINGO format parser module.

Parses BINGO-style correspondence files and timing files.

BINGO Correspondence File Format:
    Each GCP block consists of:
    - Header line: `gcp_id gcp_name`
    - One or more observation lines: `image_id U V`
    - Delimiter: `-99`
    
    Example:
        1 L4_UVT + 69_3_bldgs
        1061  431.97  4.65812
        -99
        2 L4_UVT + 67_dmnd_bldg
        1059  518.88  14.0513
        -99

    Note: U, V coordinates are typically in a photo-coordinate system
    with origin at image center. Positive U is right, positive V can be
    up (photogrammetric convention) or down depending on convention.

Timing File Format:
    CSV or space-delimited with columns: image_id, time
    Time can be GPS time, Unix timestamp, or any monotonic time reference.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BINGOObservation:
    """A single image observation of a GCP in BINGO format."""
    gcp_id: int
    gcp_name: str
    image_id: int
    u: float  # Photo-coordinate U (typically origin at image center)
    v: float  # Photo-coordinate V


@dataclass
class BINGOGCPBlock:
    """A GCP with all its image observations."""
    gcp_id: int
    gcp_name: str
    observations: List[Tuple[int, float, float]]  # List of (image_id, u, v)


class BINGOParser:
    """
    Parser for BINGO format correspondence files.
    
    The BINGO format uses blocks delimited by -99:
        gcp_id gcp_name
        image_id U V
        image_id U V
        ...
        -99
    
    Coordinate Convention:
        BINGO typically uses photo-coordinates with origin at image center:
        - U: positive right
        - V: positive up (photogrammetric) or down (depends on setup)
        
        This parser preserves the original coordinates. Conversion to
        pixel coordinates (origin top-left) should be done separately
        using image dimensions and knowledge of the V-axis convention.
    """
    
    def __init__(self, v_axis_up: bool = True):
        """
        Initialize parser.
        
        Args:
            v_axis_up: If True, V is positive upward (photogrammetric convention).
                      If False, V is positive downward.
        """
        self.v_axis_up = v_axis_up
    
    def parse_file(self, filepath: str) -> List[BINGOObservation]:
        """
        Parse a BINGO correspondence file.
        
        Args:
            filepath: Path to the BINGO correspondence file
            
        Returns:
            List of BINGOObservation objects
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BINGO file not found: {filepath}")
        
        observations = []
        blocks = self._parse_blocks(filepath)
        
        for block in blocks:
            for image_id, u, v in block.observations:
                obs = BINGOObservation(
                    gcp_id=block.gcp_id,
                    gcp_name=block.gcp_name,
                    image_id=image_id,
                    u=u,
                    v=v,
                )
                observations.append(obs)
        
        logger.info(f"Parsed {len(observations)} observations from {len(blocks)} GCPs")
        return observations
    
    def _parse_blocks(self, filepath: str) -> List[BINGOGCPBlock]:
        """
        Parse file into GCP blocks.
        
        Args:
            filepath: Path to the BINGO file
            
        Returns:
            List of BINGOGCPBlock objects
        """
        blocks = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Skip if this is a delimiter
            if line == '-99':
                i += 1
                continue
            
            # Try to parse as GCP header (id name)
            header_match = self._parse_header(line)
            if header_match:
                gcp_id, gcp_name = header_match
                observations = []
                i += 1
                
                # Read observations until -99
                while i < len(lines):
                    obs_line = lines[i].strip()
                    
                    if obs_line == '-99':
                        i += 1
                        break
                    
                    if not obs_line:
                        i += 1
                        continue
                    
                    obs = self._parse_observation(obs_line)
                    if obs:
                        observations.append(obs)
                    i += 1
                
                if observations:
                    block = BINGOGCPBlock(
                        gcp_id=gcp_id,
                        gcp_name=gcp_name,
                        observations=observations,
                    )
                    blocks.append(block)
            else:
                i += 1
        
        return blocks
    
    def _parse_header(self, line: str) -> Optional[Tuple[int, str]]:
        """
        Parse a GCP header line.
        
        Format: gcp_id gcp_name
        Example: "1 L4_UVT + 69_3_bldgs"
        
        Returns:
            Tuple of (gcp_id, gcp_name) or None if not a valid header
        """
        parts = line.split(None, 1)  # Split on first whitespace
        
        if len(parts) < 2:
            return None
        
        try:
            gcp_id = int(parts[0])
            gcp_name = parts[1].strip()
            return (gcp_id, gcp_name)
        except ValueError:
            return None
    
    def _parse_observation(self, line: str) -> Optional[Tuple[int, float, float]]:
        """
        Parse an observation line.
        
        Format: image_id U V
        Example: "1061  431.97  4.65812"
        
        Returns:
            Tuple of (image_id, u, v) or None if not valid
        """
        parts = line.split()
        
        if len(parts) < 3:
            return None
        
        try:
            image_id = int(parts[0])
            u = float(parts[1])
            v = float(parts[2])
            return (image_id, u, v)
        except ValueError:
            return None
    
    def to_pixel_coordinates(
        self,
        u: float,
        v: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:
        """
        Convert BINGO photo-coordinates to pixel coordinates.
        
        BINGO coordinates have origin at image center.
        Pixel coordinates have origin at top-left.
        
        Args:
            u: BINGO U coordinate (positive right)
            v: BINGO V coordinate 
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (pixel_u, pixel_v) with origin at top-left
        """
        # U coordinate: add half width to shift origin to left edge
        pixel_u = u + image_width / 2.0
        
        # V coordinate: depends on convention
        if self.v_axis_up:
            # Photogrammetric: V positive up, need to flip for pixel coords
            pixel_v = image_height / 2.0 - v
        else:
            # V positive down (same as pixel coords)
            pixel_v = v + image_height / 2.0
        
        return pixel_u, pixel_v


class TimingFileParser:
    """
    Parser for image timing files.
    
    Supports formats:
        - CSV: image_id,time
        - Space-delimited: image_id time
        
    Time values should be in a consistent reference frame
    (e.g., GPS seconds, Unix timestamp).
    """
    
    def parse_file(self, filepath: str) -> Dict[int, float]:
        """
        Parse a timing file.
        
        Args:
            filepath: Path to the timing file
            
        Returns:
            Dictionary mapping image_id to time
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Timing file not found: {filepath}")
        
        timings: Dict[int, float] = {}
        
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Try comma-separated first, then space-separated
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line {line_num}: {line}")
                    continue
                
                try:
                    image_id = int(parts[0].strip())
                    time = float(parts[1].strip())
                    timings[image_id] = time
                except ValueError as e:
                    logger.warning(f"Parse error on line {line_num}: {e}")
                    continue
        
        logger.info(f"Parsed {len(timings)} timing records from {filepath}")
        return timings
    
    def get_time_range(self, timings: Dict[int, float]) -> Tuple[float, float]:
        """Get the time range covered by the timing data."""
        if not timings:
            return (0.0, 0.0)
        
        times = list(timings.values())
        return (min(times), max(times))


def parse_bingo_file(
    filepath: str,
    v_axis_up: bool = True,
) -> List[BINGOObservation]:
    """
    Convenience function to parse a BINGO file.
    
    Args:
        filepath: Path to the BINGO correspondence file
        v_axis_up: Whether V is positive upward
        
    Returns:
        List of observations
    """
    parser = BINGOParser(v_axis_up=v_axis_up)
    return parser.parse_file(filepath)


def parse_timing_file(filepath: str) -> Dict[int, float]:
    """
    Convenience function to parse a timing file.
    
    Args:
        filepath: Path to timing file
        
    Returns:
        Dictionary mapping image_id to time
    """
    parser = TimingFileParser()
    return parser.parse_file(filepath)
