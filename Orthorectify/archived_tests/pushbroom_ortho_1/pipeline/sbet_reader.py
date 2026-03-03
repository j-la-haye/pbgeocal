"""
sbet_reader.py — Read Applanix SBET binary trajectory files.

SBET Format (POSPac / Applanix NED body frame):
  17 fields × float64 (8 bytes each) = 136 bytes per record at 200 Hz.

Fields (all float64):
  ┌─────┬──────────────────────┬───────────────────────┐
  │  #  │ Field                │ Units                 │
  ├─────┼──────────────────────┼───────────────────────┤
  │  0  │ time                 │ GPS seconds of week   │
  │  1  │ latitude             │ radians               │
  │  2  │ longitude            │ radians               │
  │  3  │ altitude             │ metres (ellipsoidal)  │
  │  4  │ x_velocity           │ m/s (north)           │
  │  5  │ y_velocity           │ m/s (east)            │
  │  6  │ z_velocity           │ m/s (down)            │
  │  7  │ roll                 │ radians               │
  │  8  │ pitch                │ radians               │
  │  9  │ platform_heading     │ radians               │
  │ 10  │ wander_angle         │ radians               │
  │ 11  │ x_body_accel         │ m/s² (forward)        │
  │ 12  │ y_body_accel         │ m/s² (right)          │
  │ 13  │ z_body_accel         │ m/s² (down)           │
  │ 14  │ x_body_ang_rate      │ rad/s                 │
  │ 15  │ y_body_ang_rate      │ rad/s                 │
  │ 16  │ z_body_ang_rate      │ rad/s                 │
  └─────┴──────────────────────┴───────────────────────┘

Note: Attitudes (roll, pitch, heading) are in the NED navigation frame.
      Heading uses the wander azimuth mechanisation; true heading =
      platform_heading + wander_angle.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


# Structured dtype for one SBET record
SBET_DTYPE = np.dtype([
    ('time',              np.float64),
    ('latitude',          np.float64),
    ('longitude',         np.float64),
    ('altitude',          np.float64),
    ('x_velocity',        np.float64),
    ('y_velocity',        np.float64),
    ('z_velocity',        np.float64),
    ('roll',              np.float64),
    ('pitch',             np.float64),
    ('platform_heading',  np.float64),
    ('wander_angle',      np.float64),
    ('x_body_accel',      np.float64),
    ('y_body_accel',      np.float64),
    ('z_body_accel',      np.float64),
    ('x_body_ang_rate',   np.float64),
    ('y_body_ang_rate',   np.float64),
    ('z_body_ang_rate',   np.float64),
])

RECORD_SIZE = SBET_DTYPE.itemsize  # 136 bytes


@dataclass
class SBETData:
    """
    Parsed SBET trajectory data with convenience arrays.

    Attributes
    ----------
    time : (N,) GPS seconds of week
    lat  : (N,) latitude  [rad]
    lon  : (N,) longitude [rad]
    alt  : (N,) ellipsoidal altitude [m]
    roll : (N,) roll  [rad]
    pitch: (N,) pitch [rad]
    heading: (N,) true heading [rad]  (platform_heading + wander_angle)
    vel_ned: (N,3) velocity in NED [m/s]
    raw  : structured numpy array with all 17 fields
    """
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    heading: np.ndarray       # true heading = platform_heading + wander_angle
    vel_ned: np.ndarray       # (N, 3) north, east, down
    raw: np.ndarray           # full structured array


def read_sbet(filepath: str) -> SBETData:
    """
    Read an Applanix SBET binary file.

    Parameters
    ----------
    filepath : path to the .out / .sbet binary file

    Returns
    -------
    SBETData with all fields extracted and true heading computed.

    Raises
    ------
    FileNotFoundError, ValueError on bad file size.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SBET file not found: {filepath}")

    file_size = path.stat().st_size
    if file_size % RECORD_SIZE != 0:
        raise ValueError(
            f"SBET file size ({file_size} bytes) is not a multiple of "
            f"record size ({RECORD_SIZE} bytes). Check file integrity."
        )

    num_records = file_size // RECORD_SIZE
    raw = np.fromfile(str(path), dtype=SBET_DTYPE, count=num_records)

    # Compute true heading from platform heading + wander angle
    true_heading = raw['platform_heading'] + raw['wander_angle']

    # Normalise heading to [0, 2π)
    true_heading = true_heading % (2.0 * np.pi)

    vel_ned = np.column_stack([
        raw['x_velocity'],   # north
        raw['y_velocity'],   # east
        raw['z_velocity'],   # down
    ])

    return SBETData(
        time=raw['time'].copy(),
        lat=raw['latitude'].copy(),
        lon=raw['longitude'].copy(),
        alt=raw['altitude'].copy(),
        roll=raw['roll'].copy(),
        pitch=raw['pitch'].copy(),
        heading=true_heading,
        vel_ned=vel_ned,
        raw=raw,
    )


def trim_sbet(sbet: SBETData, t_start: float, t_end: float,
              margin: float = 1.0) -> SBETData:
    """
    Trim SBET to a time window with a safety margin for interpolation.

    Parameters
    ----------
    sbet    : full SBETData
    t_start : earliest exposure time [GPS s]
    t_end   : latest exposure time [GPS s]
    margin  : extra time on each side [s]  (default 1 s = 200 records at 200 Hz)

    Returns
    -------
    New SBETData containing only the relevant time span.
    """
    mask = (sbet.time >= t_start - margin) & (sbet.time <= t_end + margin)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No SBET records in [{t_start - margin:.3f}, {t_end + margin:.3f}]"
        )

    return SBETData(
        time=sbet.time[idx],
        lat=sbet.lat[idx],
        lon=sbet.lon[idx],
        alt=sbet.alt[idx],
        roll=sbet.roll[idx],
        pitch=sbet.pitch[idx],
        heading=sbet.heading[idx],
        vel_ned=sbet.vel_ned[idx],
        raw=sbet.raw[idx],
    )
