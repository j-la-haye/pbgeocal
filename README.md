# av4-extract-time-pose: 
- extract GPS times (GPS TOD in 10usec) from binary AV4 flight lines contained in specified `mission-path`  and interpolate poses for each line from processed trajectory sbet file (NED-DEG). 


```
Example usage: 

Command line: python av4-extract-time-pose [path/to/mission-dir/] [/path/to/trajectory_file]\
              [/path/to/imu_file] --int-pose False --out-dir L1-geo-data --ext [default .bin]

Config .ini file : python av4-extract-time-pose --config [path/to/mission-dir/]
```


