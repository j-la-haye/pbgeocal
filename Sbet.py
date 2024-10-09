#!~/Work/Students/JulienB/hygeocal/bin/python

import math, struct

datagram_size = 136 # 8*17 bytes per datagram

field_names = ('time', 'latitude', 'longitude', 'altitude', \
          'x_vel', 'y_vel', 'z_vel', \
          'roll', 'pitch', 'heading', 'wander_angle', \
          'x_acceleration', 'y_acceleration', 'z_acceleration', \
          'x_angular_rate', 'y_angular_rate', 'z_angular')

class Sbet(object):
    #add option to convert to deg or not 

    def __init__(self, filename, deg=True):
        with open(filename, 'rb') as sbet_file:
            self.data = sbet_file.read()
        #sbet_file = open(filename)
        #self.data = sbet_file.read()

        # Make sure the file is sane
        assert(len(self.data)%datagram_size == 0)

        self.num_datagrams = len(self.data) / datagram_size

    def decode(self, offset=0,deg=True):
        'Return a dictionary for an SBet datagram starting at offset'

        values = struct.unpack('17d',self.data[ offset : offset+datagram_size ])

        sbet_values = dict(zip (field_names, values))

        if deg:
            sbet_values['lat_deg'] = math.degrees(sbet_values['latitude'])
            sbet_values['lon_deg'] = math.degrees(sbet_values['longitude'])
            sbet_values['roll'] = math.degrees(sbet_values['roll'])
            sbet_values['pitch'] = math.degrees(sbet_values['pitch'])
            sbet_values['heading'] = math.degrees(sbet_values['heading'])
            sbet_values['time_10usec'] = sbet_values['time'] * 1e5

        return sbet_values
    
    def get_offset(self, datagram_index):
        return datagram_index * datagram_size

    def get_datagram(self, datagram_index):
        offset = self.get_offset(datagram_index)
        values = self.decode(offset)
        return values
    
    def __iter__(self):
        'start iteration'
        self.iter_position = 0
        return self

    def __next__(self):
        'Take the next step in the iteration'
        if self.iter_position >= self.num_datagrams:
            raise StopIteration

        values = self.get_datagram(self.iter_position)
    
        self.iter_position += 1

        return values

    def sbet_print(self, sbet_values):
        'Print out all the values of a SBET dictionary'
        print ('results:')
        for key in sbet_values:
            print ('    ', key, sbet_values[key])

    #datagram_size = 136 # 8*17 bytes per datagram

    def saveSbet2csv(self, filename):
        'Save the SBET data to a CSV file'
        # Write the header
        with open(filename, 'w') as csv_file:
            csv_file.write('#time_10usec,lat_deg,lon_deg,altitude,roll,pitch,heading,'\
                           #'x_acceleration,y_acceleration,z_acceleration,x_angular_rate,y_angular_rate,'z_angular_rate,wander_angle,'\
                            'x_vel,y_vel,z_vel\n')
            for datagram in self:
                csv_file.write(f"{datagram['time_10usec']:.0f},{datagram['lat_deg']},{datagram['lon_deg']},{datagram['altitude']},"
                               f"{datagram['roll']},{datagram['pitch']},{datagram['heading']},{datagram['x_vel']},{datagram['y_vel']},{datagram['z_vel']}\n")
                
            
            # for datagram in self:
            #     csv_file.write(datagram['time_10usec'], datagram['lat_deg'], datagram['lon_deg'], datagram['altitude'], \
            #     datagram['roll'], datagram['pitch'], datagram['heading'],\
            #     #datagram['x_acceleration'], datagram['y_acceleration'], datagram['z_acceleration'], \
            #     #datagram['x_angular_rate'], datagram['y_angular_rate'], datagram['z_angular'],,datagram['wander_angle']\
            #     datagram['x_vel'], datagram['y_vel'], datagram['z_vel'],'\n')
            #     #csv_file.write('\n')
            #     #f"{datagram['time_10usec']:.0f}"
def main():
    print('Datagram Number, Time, x, y')

    sbet = Sbet('av_4_ortho_rect/Test_SBET/Atlans_A7-20240720-100407_Thun_POSTPROCESSING_V2_SBET_TOD.out',deg=True)
    # for datagram in sbet:
    #     #print(datagram['time_10usec'],datagram['lon_deg'], datagram['lat_deg'])
    #     #print time_10usec with 3 decimal places
    #     print(f"{datagram['time_10usec']:.0f}", datagram['lon_deg'], datagram['lat_deg'])


    #Save the SBET data to a CSV file
    sbet.saveSbet2csv('av_4_ortho_rect/Test_SBET/Atlans_A7-20240720-100407_Thun_SBET_10usec.csv')

if __name__=='__main__':
    main()