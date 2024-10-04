# av4-extract-time-pose: 
- extract GPS times (GPS TOD in 10usec) from binary AV4 flight lines contained in specified `mission-path`  and interpolate poses from processed trajectory. 


```
Example usage: python av4-extract-time-pose [path/to/mission-dir/] [/path/to/trajectory_file]\
              [/path/to/imu_file] --int-pose False --out-dir L1-geo-data --ext [default .bin]
```


