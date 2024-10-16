import os
import struct

# Constants
data_t = 'H'  # Unsigned short (16-bit integer)
aviris4img_channels = 327  # Does not include the band with time tags
aviris4img_resolution = 1280
aviris4img_headerlinelen = aviris4img_resolution * struct.calcsize(data_t)
aviris4img_linelen = aviris4img_resolution * (aviris4img_channels + 1) * struct.calcsize(data_t)
aviris4img_linedatalen = aviris4img_resolution * aviris4img_channels * struct.calcsize(data_t)

sysTimeOffset = 0
statusFlagOffset = 80
statusFlagExpected = 0xBABE
utcTowOffset = 116
sysTimePPSOffset = 164

def load_frame_times(frame_file_path,read_first_time_only = False):
    # Check file size and existence
    if not os.path.exists(frame_file_path):
        return []

    file_size = os.path.getsize(frame_file_path)

    if file_size % aviris4img_linelen != 0:
        return []  # Unexpected file size

    n_lines = file_size // aviris4img_linelen

    # Structure to hold line timing info
    class LineTimingInfos:
        def __init__(self, internal_time, gps_time_last_pps, internal_time_last_pps, is_babe):
            self.internal_time = internal_time
            self.gps_time_last_pps = gps_time_last_pps
            self.internal_time_last_pps = internal_time_last_pps
            self.is_babe = is_babe

    infos = [None] * n_lines
    ret = [0] * n_lines

    with open(frame_file_path, 'rb') as f:
        header_data = bytearray(aviris4img_headerlinelen)

        for i in range(n_lines):
            # Read the header line
            f.seek(i * aviris4img_linelen)
            f.readinto(header_data)

            # Extract various data from header
            line_internal_time = struct.unpack_from('<I', header_data, sysTimeOffset)[0]  # Little endian 4 bytes
            flag = struct.unpack_from('<H', header_data, statusFlagOffset)[0]  # 2 bytes for the status flag
            is_babe = (flag ^ statusFlagExpected) == 0

            gps_validity_time = struct.unpack_from('>I', header_data, utcTowOffset)[0]  # Big endian 4 bytes

            # Fixing the custom byte order for ppsInternalTime as described
            pps_bytes = header_data[sysTimePPSOffset:sysTimePPSOffset + 4]
            pps_internal_time = (pps_bytes[2] | (pps_bytes[3] << 8) | 
                                 (pps_bytes[0] << 16) | (pps_bytes[1] << 24))

            # Store the timing info
            infos[i] = LineTimingInfos(line_internal_time, gps_validity_time, pps_internal_time, is_babe)
            # if is_babe and read_first_time_only, exit the if loop
            if is_babe and read_first_time_only:
                lines_read = i
                break
                

    # Fill in missing values for gps and internal times
    if not read_first_time_only:
        babe_idxs = [i for i in range(n_lines) if infos[i].is_babe]
    else:
        babe_idxs = [lines_read] #[i for i in range(lines_read) if infos[i].is_babe]

    if not babe_idxs:
        return []  # No valid BABE flags found

    previous_babe_idx = babe_idxs[0]
    next_babe_idx = babe_idxs[0]
    current_babe_idx_pos = 0

    if not read_first_time_only:
        for i in range(n_lines):
            delta_prev = abs(i - previous_babe_idx)
            delta_next = abs(i - next_babe_idx)

            if delta_prev < delta_next:
                infos[i].gps_time_last_pps = infos[previous_babe_idx].gps_time_last_pps
                infos[i].internal_time_last_pps = infos[previous_babe_idx].internal_time_last_pps
            else:
                infos[i].gps_time_last_pps = infos[next_babe_idx].gps_time_last_pps
                infos[i].internal_time_last_pps = infos[next_babe_idx].internal_time_last_pps

            if i == next_babe_idx:
                previous_babe_idx = next_babe_idx
                current_babe_idx_pos += 1
                if current_babe_idx_pos >= len(babe_idxs):
                    current_babe_idx_pos = len(babe_idxs) - 1
                next_babe_idx = babe_idxs[current_babe_idx_pos]

        # Calculate the final times
        for i in range(n_lines):
            delta_t = infos[i].internal_time - infos[i].internal_time_last_pps
            ret[i] = infos[i].gps_time_last_pps * 10 + delta_t
    else:
        infos[0].gps_time_last_pps = infos[lines_read].gps_time_last_pps
        infos[0].internal_time_last_pps = infos[lines_read].internal_time_last_pps
        
        delta_t = infos[0].internal_time - infos[0].internal_time_last_pps
        ret = [infos[0].gps_time_last_pps * 10 + delta_t]

    return ret


def main():
    import sys

    if len(sys.argv) != 3:
        print("Wrong number of arguments provided, only the input file is required!", file=sys.stderr)
        return 1

    file_path = sys.argv[1]

    read_first_time_only= sys.argv[2] if len(sys.argv) == 3 else False

    if not os.path.exists(file_path):
        print(f"Provided input file: '{file_path}' does not exist!", file=sys.stderr)
        return 1
    print(f"Provided input file: '{file_path}',{read_first_time_only}", file=sys.stderr)
    times = load_frame_times(file_path,read_first_time_only)

    if not times:
        print(f"Could not load time data from file: '{file_path}'!", file=sys.stderr)
        return 1

    for time in times:
        print(time)

    return 0

if __name__ == "__main__":

    #raw_path = '/Users/jlahaye/Work/AVIRIS4/AV4_Thun/AV4_Raw_Data/L102/102_Locked/RawDataCube_Line_0102_2.bin'
    #times = load_frame_times(raw_path, read_first_time_only=True)

    main()
