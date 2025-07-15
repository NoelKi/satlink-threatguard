"""
This is a python library for inter-satellite communicaiton detection. 
Author: Kieran Noel Mai
Date: 2024
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.special import jn
import warnings
import ephem
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import ITRF_to_GCRS2
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime
import folium
import requests
from geopy.geocoders import Nominatim
import time
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj, Transformer

class Satellite(object):
    def __init__(self,name,x,y,z,velo,lat,lon):
        """
        Initialize a Satellite object.

        :param name: Satellite name
        :type name: str 
        :param x: X-coordinate
        :type x: list[float]
        :param y: Y-coordinate
        :type y: list[float]
        :param z: Z-coordinate
        :type z: list[float]
        :param velo: Velocity
        :type velo: list[float]
        :param lat: Latitude
        :type lat: list[float]
        :param lon: Longitude
        :type lon: list[float]
        """
        self.name = name
        self.x = x
        self.y = y
        self.z = z 
        self.frequency = None
        self.velo = velo
        self.lat = lat
        self.lon = lon
        self.doppler = None
        self.distance = None
        self.anglespeed = None         
        self.los = None
        self.periods = None
        self.flightvecs = None
        self.theta = None
        self.ocean = None
        self.area = None 
        self.threshold = None 
        self.contacts = None 
        self.theta_3dB = None
        self.theta_attack = None
        self.frequency_range = None
        self.overlapping_frequencys = []

    def set_theta_3dB(self,theta):
        """
        (not used)
        Set the theta_3dB attribute for the satellite.

        :param theta: value to set for theta_3dB
        """
        self.theta_3dB = theta

    def set_theta_attack(self,theta):
        """
        (not used)
        Set the theta_attack attribute for the satellite.

        :param theta: value to set for theta_3dB
        """
        self.theta_attack = theta

    def set_frequency_range(self,frequency_range):
        """
        (not used)
        Set the frequency range attribute for the satellite. 

        :param frequency_range: tuple from frequency range
        """
        self.frequency_range = frequency_range

    def check_frequency_overlap(self, other):
        """
        (not used)
        Check if there is a frequency overlap with another satellite and record it.

        :param other: Another Satellite object to compare with
        """
        min1, max1 = self.frequency_range
        min2, max2 = other.frequency_range
        if max(min1, min2) <= min(max1, max2):
            if other.name not in self.overlapping_frequencys:
                self.overlapping_frequencys.append(other.name)
    
    def set_time_step(self,time_step):
        """
        (not used)
        Set timestep attribute for satellite. 

        :param time_step: timestep [sek]
        """
        self.time_step = time_step

def set_theta_attack_loop(sat_list,theta):
    """
    Set for all satellites in sat_list a value for theta.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]

    :param theta: theta from antenna 
    :type theta_3dB: float or int
    """
    for sat in sat_list:
        sat.set_theta_attack(theta)
        
def set_theta_3dB_loop(sat_list,theta):
    """
    Set for all satellites in sat_list a value for theta.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]

    :param theta: theta from antenna 
    :type theta: float or int
    """
    for sat in sat_list:
        sat.set_theta_3dB(theta)

def set_frequency_future(sat_list,theta_target,theta_attack):
    """
    Set for all satellites in sat_list a value for both thetas. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]

    :param theta_target: theta from target satellite 
    :type theta_target: float or int 
    :param theta_attack: theta from attacking satellite 
    :type theta_attack: float or int
    """
    set_theta_3dB_loop(sat_list,theta_target)
    set_theta_attack_loop(sat_list,theta_attack)

def set_frequency_past(sat_list,theta_target):
    """
    Set for all satellites in sat_list a value for both thetas. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]

    :param theta_target: theta from target satellite 
    :type theta_target: float or int 
    :param theta_attack: theta from attacking satellite 
    :type theta_attack: float or int 
    """
    set_theta_3dB_loop(sat_list,theta_target)

def add_satellite_from_earth(earth_sat_list):
    """
    Create Satellite objects from a list of EarthSatellite objects.

    :param earth_sat_list: List of EarthSatellite objects
    :type earth_sat_list: list[EarthSatellite]
    :return: List of Satellite objects
    :ytype: list[Satellite]
    """
    return [Satellite(s.name, s.x, s.y, s.z, s.velo, s.lat, s.lon) for s in earth_sat_list]

def earthsatellite_extension(sat_list:list[EarthSatellite]) ->list[Satellite]:
    """
    This function extends the attributes of a list of class objects to x,y,z.  

    :param sat_list: list of objects of the EarthSatellite class.
    :type sat_list: list[Satellite]
    :return sat_list: list of objects of the EarthSatellite class.
    :rtype: list[Satellite]
    """
    for sat in sat_list:  
        sat.x = []
        sat.y = []
        sat.z = []
        sat.velo = []
        sat.lat = []
        sat.lon = []
    return sat_list

def name_determination(sat_list:list[Satellite]):
    """
    This function extract the name of the Satellites out of a 
    list of objects in order of declaration and saves it as a 
    list. 

    :param sat_list: list of objects of the Satellite class.
    :type sat_list: list[Satellite]
    :return name_list: list of names from satellites.  
    :rtype: list
    """
    name_list = []
    for sat in sat_list:
        name_list.append(sat.name)
    return name_list

def plot(sat_list:list[Satellite]):
    """
    This function visualizes the position of the satellites determined by the 
    pos_determination function. Plot is interactive. 

    :param sat_list:  list of objects of the Satellite class 
    :type sat_list: list[Satellite]
    """
    max_range = 60000 # determine shown axis interval 
    names = name_determination(sat_list) # create list of satellite names for plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]]) 
    
    # add earth
    a = b = 6356.752  # axis in x,y-direction
    c = 6378.137  # axis in z-direction
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x0 = a * np.outer(np.cos(u), np.sin(v))
    y0 = b * np.outer(np.sin(u), np.sin(v))
    z0 = c * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Scatter3d(x=x0.flatten(), y=y0.flatten(), z=z0.flatten(), mode='markers', name='Earth', marker=dict(size=1, color='blue')))

    for index,sat in enumerate(sat_list): 
            x = sat_list[index].x
            y = sat_list[index].y
            z = sat_list[index].z
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',name=names[index]), row=1, col=1) # create orbit
    fig.update_layout(scene=dict(xaxis_title='X (km)', 
                                 yaxis_title='Y (km)',
                                 zaxis_title='Z (km)',
                                 aspectmode='cube',
                                 xaxis=dict(range=[-max_range, max_range]),  # arrange plot axis for identical scales 
                                 yaxis=dict(range=[-max_range, max_range]), 
                                 zaxis=dict(range=[-max_range, max_range]),
                                   ),
                      title='Satellite Orbits')                                                  
    fig.show()

def sim_time(sim_start:datetime,sim_end:datetime):
    """
    This function creates two time instances from the time class of Skyfield. 
    The two instances are dedicated to the start and end of the simulation.

    :param sim_start: Datetime for simulation beginn
    :type sim_start: datetime
    :param sim_end: Datetime for simulation end
    :type sim_end: datetime
    :return: Simulation start and end time, in Timescale class (Skyfield)
    :rtype: tuple
    """
    ts = load.timescale()                                           # load timescale
    start_time = ts.utc(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second)   # create starttime instance 
    end_time = ts.utc(sim_end.year,sim_end.month,sim_end.day,sim_end.hour,sim_end.minute,sim_end.second)      # create endtime instance 
    time = start_time, end_time                                     
    return time

def generate_satellite_motion_data(time_start,time_end,time_step,sat_list:list[Satellite]):
    """
    Calculates the position in ECI Coordinates for satellite inside the 
    sat_list. The position is calulated over a time interval (see sim_time)
    for predetermined finite steps. Also the velocity and the latitude and 
    longitude been calculated and stored for every object in sat_list. 

    :param time_start: Start time of the simulation (Skyfield timescale)
    :type time_start: Time
    :param time_end: End time of the simulation (Skyfield timescale)
    :type time_end: Time
    :param time_step: Time step in seconds.
    :type time_step: int or float
    :param sat_list: List of objects of the EarthSatellite class
    :type sat_list: list[Satellite]
    :return: List of satellites with new attributes x, y, z, velo, lat, lon.
    :rtype: list[Satellite]
    """
    timescale = load.timescale() #load timescale
    time_current = time_start # set time_current
    while time_current.tt < time_end.tt: # while-loop
        for sat in sat_list: # for-loop
            geocentric = sat.at(time_current) 
            geocentric_pos_km = geocentric.position.km
            lat, lon = wgs84.latlon_of(geocentric)
            velocity = geocentric.velocity.km_per_s
            x, y, z = geocentric_pos_km
            sat.x.append(x)             # x-coordinate element
            sat.y.append(y)             # y-coordinate element
            sat.z.append(z)             # z-coordinate element 
            sat.velo.append(velocity)   # velocity
            sat.lat.append(lat.degrees)               # latitude
            sat.lon.append(lon.degrees)               # londitude
        # increment time by time_step
        time_current = time_current.utc_datetime() + timedelta(seconds=time_step)
        time_current = timescale.utc(time_current)
    return

def frequency_add(frequency_list,sat_list:list[Satellite]):
    """
    (not used)
    This function stores the frequency from frequency_tracker for every attribute of 
    EarthSatellite Class. 

    :param frequency_list: list of frequency bands from satellites 
    :type frequency_list: list
    :param sat_list: list of satellites from class EarthSatellite
    :type sat_list: list[Satellite]
    """
    for index,sat in enumerate(sat_list):
        sat.frequency = frequency_list[index]
    return

def calculate_distance(sat0_position, sat1_position):
    """
    This function calculates the distance between two points. 

    :param sat1_position: (x,y,z) from point 1 [km,m]
    :type sat1_postion: list
    :param sat2_position: (x,y,z) from point 2 [km,m]
    :type sat2_position: list
    :return distance: distance
    :rtype: list
    """
    vector_between = tuple(p2 - p1 for p1, p2 in zip(sat0_position, sat1_position))
    distance = np.linalg.norm(vector_between)
    return distance

def relative_speed(sat_list:list[Satellite]):
    """
    This function calculates the relative speed between two satellites.

    :param sat_list: list of objects of satellites with velocity
    :type sat_list: list[Satellite]
    :return sat_list: list of sat objects with added reference velocity 
    :rtype: list
    """

    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].velo)

    for sat in sat_list:
        sat.doppler = [[] for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
                for x in range(len_cal):
                    # i, j represent satellites, x represents timesteps
                    v_x = sat_list[i].velo[x][0]-sat_list[j].velo[x][0]
                    v_y = sat_list[i].velo[x][1]-sat_list[j].velo[x][1]
                    v_z = sat_list[i].velo[x][2]-sat_list[j].velo[x][2]
                    relative_vel = np.sqrt(v_x**2+v_y**2+v_z**2)
                    sat_list[i].doppler[j].append(relative_vel)
    return 

def line_of_sight_points(sat_list:list[Satellite]):
    """
    This function calculates points on a vector between satellites for every timestep and saves them into a list. 

    :param sat_list: list of objects of satellites
    :type sat_list: list[Satellite]
    """
    num_points_on_line = 333 # actually one more, defined in loop
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)

    for sat in sat_list:
        sat.points = [[] for _ in range(len_cal)] 

    for x in range(len_cal):
        for i in range(num_satellites):
            for j in range(num_satellites):
                if i == j:
                    points_on_line = ['self']
                else:
                    points_on_line = []
                    for k in range(1+num_points_on_line):
                        x_coord = sat_list[i].x[x] + (sat_list[j].x[x] - sat_list[i].x[x]) * k / num_points_on_line
                        y_coord = sat_list[i].y[x] + (sat_list[j].y[x] - sat_list[i].y[x]) * k / num_points_on_line 
                        z_coord = sat_list[i].z[x] + (sat_list[j].z[x] - sat_list[i].z[x]) * k / num_points_on_line      
                        points_on_line.append((x_coord, y_coord, z_coord))
                sat_list[i].points[x].append(points_on_line)

    return 

def line_of_sight_earth_sphere(sat_list,atmosphere):
    """
    Modulates the Earth as a sphere and checks if a line of sight between satellites is possible.

    The function calculates whether a line of sight between satellites is possible, assuming the Earth
    is a sphere with given atmosphere height. It saves the Boolean states in the .los attribute of 
    each satellite object.

    :param sat_list: List of Satellite objects.
    :type sat_list: list[Satellite]
    :param atmosphere: Height of the assumed atmosphere.
    :type atmosphere: float
    :return: List of satellite objects with updated attribute 'los'.
    :rtype: list[Satellite]
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)
    num_points = len(sat_list[0].points[0][1])  # amont points on vektor
    a = 6371 + atmosphere  # axis in x-direction
    b = a # axis in y-direction
    c = a # axis in z-direction

    for sat in sat_list:
        sat.los = [[]for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                sat_list[i].los[j] = [False] * len_cal
            else:
                for l in range(len_cal):
                    los_points = []
                    for k in range(num_points):  
                        x, y, z = sat_list[i].points[l][j][k]
                        is_inside = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) <= 1
                        los_points.append(is_inside)   
                    If_True_than_False = not any(los_points)
                    sat_list[i].los[j].append(If_True_than_False)

    return sat_list

def line_of_sight_ellipsoid(sat_list,atmosphere,start_time:datetime,time_step):
    """
    Modulates the Earth as an ellipsoid and checks if a line of sight between satellites is possible.

    The function calculates whether a line of sight between satellites is possible, assuming the Earth
    is an ellipsoid with given atmosphere height. It saves the Boolean states in the 'los' attribute of 
    each satellite object.

    :param sat_list: List of Satellite objects.
    :type sat_list: list[Satellite]
    :param atmosphere: Height of the assumed lowest line of sight above the Earth surface.
    :type atmosphere: float or int
    :param start_time: Time of simulation start in datetime format.
    :type start_time: datetime.datetime
    :param time_step: Size of the time step in seconds.
    :type time_step: int
    :return: List of satellite objects with added attribute 'los'.
    :rtype: list[Satellite]
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)
    num_points = len(sat_list[0].points[0][1])  # amont points on vektor
    a = 6378.137 + atmosphere
    b = a  
    c = 6356.752 + atmosphere 

    for sat in sat_list:
        sat.los = [[]for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                sat_list[i].los[j] = [False] * len_cal
            else:
                for l in range(len_cal):
                    time = start_time + timedelta(seconds=time_step*l)
                    los_points = []
                    for k in range(num_points):  
                        x, y, z = sat_list[i].points[l][j][k]
                        is_inside = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) <= 1
                        los_points.append(is_inside)   
                    If_True_than_False = not any(los_points)
                    sat_list[i].los[j].append(If_True_than_False)

def line_of_sight_with_rot(sat_list,atmosphere,start_time:datetime,time_step):
    """
    This function modulates the Earth as a wgs 84 ellipsoid and checks if a line of sight between satellites is possible,
    the function saves the Boolean states in .los attribute as: [[states for satellite 0],[states for satellite 1],...]
    
    :param sat_list: list of objects of satellites attribute 'points' from satellite_vektor_connection
    :param atmosphere: height of assumed lowest LOS above Earth surface
    :param start_time: time of simulation start in datetime
    :param time_step: size of timestep [sek]
    :return sat_list: list of satellite objects with added attribute 'los'
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)
    num_points = len(sat_list[0].points[0][1])  # amont points on vektor
    a = 6378.137 + atmosphere
    b = a  
    c = 6356.752 + atmosphere 

    for sat in sat_list:
        sat.los = [[]for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                sat_list[i].los[j] = [False] * len_cal
            else:
                for l in range(len_cal):
                    time = start_time + timedelta(seconds=time_step*l)
                    lst = calculate_gmst_with_skyfield(time) 
                    lst = -lst
                    los_points = []
                    for k in range(num_points):  
                        x, y, z = sat_list[i].points[l][j][k]
                        x_rot, y_rot = rotate_eci_coordinates(x, y, lst)
                        is_inside = (x_rot**2 / a**2 + y_rot**2 / b**2 + z**2 / c**2) <= 1
                        los_points.append(is_inside)   
                    If_True_than_False = not any(los_points)
                    sat_list[i].los[j].append(If_True_than_False)

def calculate_gmst(datetime):
    """
    Calculate the Local Sidereal Time (LST) in radians for a given observation time, assuming the observer is at the Greenwich Meridian.

    :param observation_time: A datetime object representing the date and time of observation.
    :return: The LST in radians.
    """
    jd = 2451545.0 + datetime.toordinal() + (datetime.hour - 12) / 24 + datetime.minute / 1440 + datetime.second / 86400
    T = (jd - 2451545.0) / 36525.0
    GMST_deg = 280.46061837 + 360.98564736629 * (jd - 2451545) + T**2 * (0.000387933 - T / 38710000)
    GMST_rad = math.radians(GMST_deg % 360)
    
    return GMST_rad 

def rotate_eci_coordinates(x_eci, y_eci, lst):
    """
    Rotate ECI coordinates based on Local Sidereal Time (LST).

    :param x_eci: ECI x-coordinate.
    :type x_eci: float
    :param y_eci: ECI y-coordinate.
    :type y_eci: float
    :param z_eci: ECI z-coordinate.
    :type z_eci: float
    :param lst: Local Sidereal Time (LST) in radians.
    :return: Tuple (x_rotated, y_rotated, z_rotated) with rotated ECI coordinates.
    :rtype: (float, float, float)
    """
    x_rotated = x_eci * math.cos(lst) - y_eci * math.sin(lst)
    y_rotated = x_eci * math.sin(lst) + y_eci * math.cos(lst)

    return x_rotated, y_rotated

def is_inside_ellpsioid(x,y,z):
    """
    This function checks if a point is inside the wgs 84 ellipsoid.

    :param x: x-coordinate
    :type x: float
    :param y: y-coordinate
    :typer y: float
    :param z: z-coordinate
    :type z: float
    :return is_inside_point:
    :rtype: boolen
    """
    a = 6378.137  # Halbachse in x-Richtung
    b = a  # Halbachse in y-Richtung
    c = 6356.752  # Halbachse in z-Richtung

    is_inside_point = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) <= 1
    return is_inside_point

def get_position_of_true(list):
    """
    This function returns the position of a Boolean if it's True, from a list.
    Works for sat_list[x].los[y].

    :param lst: List of objects of satellites with points of vector.
    :type lst: list
    :return: List of positions of True values in the list.
    :rtype: list
    """
    bool = True
    positions = [index for index, value in enumerate(list) if value == bool]
    return positions

def find_continuous_sequences(list):
    """
    This funciton checks continuity of values in a list and returns in case the first and last
    value of the continity stored in a list. Also works for multiple series of continity. 

    :param list: list of digits
    :type list: list
    :return sequences: list of beginning and end position of continity series in committed list
    :rtype: list[tuple]
    """
    sequences = []
    if len(list) == 0: 
        print('----------------------------------------------------------------------')
        print('These two satellites do no have any line of sight in the given period.')
        print('----------------------------------------------------------------------')
    else:
        current_sequence = [list[0]]

        for i in range(1, len(list)):
            if list[i] == list[i - 1] + 1:
                current_sequence.append(list[i])
            else:
                if len(current_sequence) > 1:
                    sequences.append((current_sequence[0], current_sequence[-1]))
                current_sequence = [list[i]]

        if len(current_sequence) > 1:                                           # check if last sequence is included
            sequences.append((current_sequence[0], current_sequence[-1]))

    return sequences

def find_continuous_sequences_01(list):
    """
    This funciton checks continuity of values in a list and returns in case the first and last
    value of the continity stored in a list. Also works for multiple series of continity. 

    :param list: list of digits
    :type list: list
    :return sequences: list of beginning and end position of continity series in committed list.
    :rtype: list[tuple]
    """
    sequences = []
    if len(list) == 0: 
        return sequences
    else:
        current_sequence = [list[0]]

        for i in range(1, len(list)):
            if list[i] == list[i - 1] + 1:
                current_sequence.append(list[i])
            else:
                if len(current_sequence) > 1:
                    sequences.append((current_sequence[0], current_sequence[-1]))
                current_sequence = [list[i]]

        if len(current_sequence) > 1:                                           # check if last sequence is included
            sequences.append((current_sequence[0], current_sequence[-1]))

    return sequences

def iteration_list_extraction_01(input_list, list_of_intervals):
    """
    This function iterates the list from find_continuous_sequence(_01) function.

    :param input_list: List of values
    :type input_list: list
    :param list_of_intervals: list of intervalls with start and end 
    :type list_of_intervals: list[tuple]
    :return: List of sublists containing values from the specified intervals.
    :rtype: list
    """
    result_list = []
    for start, end in list_of_intervals:
        result_list.append(input_list[start:end+1])
    return result_list

def iteration_list_extraction_02(input_list, list_of_intervals):
    """
    This function iterates the list from find_continuous_sequences(_01) function.

    :param input_list: List of values.
    :type input_list: list
    :param list_of_intervals: List of tuples defining intervals.
    :type list_of_intervals: list
    :return: List containing values from the specified intervals.
    :rtype: list
    """
    result_list = []
    for start, end in list_of_intervals:
        result_list.extend(input_list[start:end+1])
    return result_list

def find_number_of_satellite(sat_list:list[Satellite],name):
    """
    This function finds the number of a satellite from a tle-list by name.
    
    :param sat_list: list of objects of satellites
    :type sat_list: list[Satellite]
    :param name: Name of Satellite
    :type name: str
    :return a: return just the frequencies as a list
    :rtype: list
    """
    names = name_determination(sat_list)
    a = []
    for i in range(1,len(names)):
        if name == names[i]:
            a.append(i)
    if not a:
        print('Waring: No satellite with this name could be found.\n        Note upper and lower case and space characters ')
        return
    return a 

def get_time(S_year,S_month,S_day,S_hour,S_min,S_sek,time_step,calc_step):
    """
    This function determines the point of time out of a calculations step.
    DO: 
    time.utc_strftime()             - UTC
    time.tt_strftime()              - TT
    time.tai_strftime()             - TAI
    {:.10f}'.format(time.tdb)       - TDB Julian date
    {:.1f}'.format(time.J)          - Julian century

    :param S_year: year from start time
    :param S_month: month for the start time
    :param S_day: day for the start time
    :param S_hour: hour for the start time
    :param S_min: minute for the start time
    :param S_sec: second for the start time
    :param calc_step: number of time_step you want to know.
    :return time: time class from skyfield
    """
    ts = load.timescale() 
    time = ts.utc(S_year,S_month,S_day,S_hour,S_min,S_sek+time_step*calc_step)
    return time

def parse_datetime(input_string):
    """
    This function parses the string of the time completely.

    :param input_string: time in the format: '2023-08-31 16:09:30 UTC'
    :return parsed_time: list of [year,month,day,hour,min,sek]
    """
    try:
        date_format = '%Y-%m-%d %H:%M:%S %Z'

        parsed_datetime = datetime.strptime(input_string, date_format)

        year = parsed_datetime.year
        month = parsed_datetime.month
        day = parsed_datetime.day

        hour = parsed_datetime.hour
        minute = parsed_datetime.minute
        second = parsed_datetime.second

        parsed_time = [year, month, day, hour, minute, second]
        return parsed_time
    except ValueError:
        return None 
    
def parse_date(input_string):
    """
    This function parses the string of the time sequently.

    :param input_string: time in the format: '2023-08-31 16:09:30 UTC'
    :return parsed_time: list of [month,day]
    """
    try:
        date_format = '%Y-%m-%d %H:%M:%S %Z'

        parsed_datetime = datetime.strptime(input_string, date_format)

        month = parsed_datetime.month
        day = parsed_datetime.day

        parsed_time = [month, day]
        return parsed_time
    except ValueError:
        return None 
    
def show_ground_track_map(sat_list:list[Satellite],sat_no):
    """
    This function showns the ground track of two satellites for a list. 

    :param sat_list: list of objects of satellites with latitude and longitude
    :type sat_list: list[Satellite]
    :param sat_no: number of satellite to examined
    :return m: retruns html map
    :rtype: html
    """
    m = folium.Map(location=[0, 0], zoom_start=1.5) 

    for lat, lon in zip(sat_list[sat_no].lat, sat_list[sat_no].lon):
        folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)

    m.save('map.html') 
    return m
    
def show_ground_track_map_twosats(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function showns the ground track of two satellites for a list. 

    :param sat_list: list of objects of satellites with latitude and longitude
    :type sat_list: list[Satellite]
    :param sat_no_1,sat_no_2: number of satellite to examined
    :return m: retruns html map
    :rtype: html
    """
    m = folium.Map(location=[0, 0], zoom_start=1.5) 

    for lat, lon in zip(sat_list[sat_no_1].lat, sat_list[sat_no_1].lon):
        folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)
    for lat, lon in zip(sat_list[sat_no_2].lat, sat_list[sat_no_2].lon):
        folium.Marker([lat, lon], icon=folium.Icon(color='blue')).add_to(m)

    m.save('map.html') 
    return m

def show_ground_track_while_los(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function shows the ground track of two satellites for all periodes of los. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite to examine
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite to examine 
    :type sat_no_2: int
    :return m: html map of groundtrack
    :rtype: html
    """

    list_of_intervals = find_continuous_sequences(get_position_of_true(sat_list[sat_no_1].los[sat_no_2]))
    lon1 = []
    lat1 = []
    lon2 = []
    lat2 = []
    lon1 = iteration_list_extraction_02(sat_list[sat_no_1].lon,list_of_intervals)
    lat1 = iteration_list_extraction_02(sat_list[sat_no_1].lat,list_of_intervals)
    lon2 = iteration_list_extraction_02(sat_list[sat_no_2].lon,list_of_intervals)
    lat2 = iteration_list_extraction_02(sat_list[sat_no_2].lat,list_of_intervals)

    m = folium.Map(location=[0, 0], zoom_start=1.5) 

    for lat, lon in zip(lat1, lon1):
        folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)
    for lat, lon in zip(lat2, lon2):
        folium.Marker([lat, lon], icon=folium.Icon(color='blue')).add_to(m)

    m.save('map.html') 
    return m

def show_ground_track_while_los_specific(sat_list:list[Satellite],sat_no_1,sat_no_2,period_no):
    """
    This function shows the ground track of two satellites for a specific periodes of los. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite to examine
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite to examine 
    :type sat_no_2: int
    :param period: select number of period to show 
    :type period: int
    :return m: html map of groundtrack
    :rtype: html
    """
    list_of_intervals = find_continuous_sequences(get_position_of_true(sat_list[sat_no_1].los[sat_no_2]))
    if len(list_of_intervals) == 0:
        return 
    else:
        lon1 = []
        lat1 = []
        lon2 = []
        lat2 = []
        lon1 = iteration_list_extraction_01(sat_list[sat_no_1].lon,list_of_intervals)
        lat1 = iteration_list_extraction_01(sat_list[sat_no_1].lat,list_of_intervals)
        lon2 = iteration_list_extraction_01(sat_list[sat_no_2].lon,list_of_intervals)
        lat2 = iteration_list_extraction_01(sat_list[sat_no_2].lat,list_of_intervals)

        if period_no > len(lat1):
            raise ValueError(f"Period number {period_no} is out of range. Please select a valid period number.")

        m = folium.Map(location=[0, 0], zoom_start=1.5) 
        
        for lat, lon in zip(lat1[period_no], lon1[period_no]):
            folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)
        for lat, lon in zip(lat2[period_no], lon2[period_no]):
            folium.Marker([lat, lon], icon=folium.Icon(color='blue')).add_to(m)

        m.save('map.html') 
        return m

def show_ground_track_for_specific_communication_period(sat_list:list[Satellite],orientation,sat_no_1,sat_no_2,com_periods,period_no):
    """
    This function shows the ground track of two satellites for a specific periodes of los. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite to examine
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite to examine 
    :type sat_no_1: int
    :param com_periods: list of tuple of communication periods 
    :type com_periods: list
    :param period: select number of period to show 
    :type period: int
    :return m: html map of groundtrack
    """
    list_of_intervals = com_periods
    if len(list_of_intervals) == 0:
        return 
    else:
        lon1 = []
        lat1 = []
        lon2 = []
        lat2 = []
        lon1 = iteration_list_extraction_01(sat_list[sat_no_1].lon,list_of_intervals)
        lat1 = iteration_list_extraction_01(sat_list[sat_no_1].lat,list_of_intervals)
        lon2 = iteration_list_extraction_01(sat_list[sat_no_2].lon,list_of_intervals)
        lat2 = iteration_list_extraction_01(sat_list[sat_no_2].lat,list_of_intervals)

        if period_no > len(lat1):
            raise ValueError(f"Period number {period_no} is out of range. Please select a valid period number.")

        m = folium.Map(location=[0, 0], zoom_start=1.5) 
        
        for lat, lon in zip(lat1[period_no], lon1[period_no]):
            folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)
        for lat, lon in zip(lat2[period_no], lon2[period_no]):
            folium.Marker([lat, lon], icon=folium.Icon(color='blue')).add_to(m)

        m.save('map.html') 
        return m

def print_los_info(sat_list:list[Satellite],sat_no_1,sat_no_2,sim_start,time_step,max_anglespeed,max_distance,max_velo):
    """
    This function prints intervals from line of sight of satellite x and y.

    :param sat_list: list of objects of satellites
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite
    :type sat_no_1: int
    :param S_year: year from start time
    :param S_month: month for the start time
    :param S_day: day for the start time
    :param S_hour: hour for the start time
    :param S_min: minute for the start time
    :param S_sec: second for the start time
    :param max_anglespeed: list of max medium anglespeeds of periods
    :type max_anglespeed: list
    """
    list_of_intervals = find_continuous_sequences(get_position_of_true(sat_list[sat_no_1].los[sat_no_2]))
    if len(list_of_intervals) == 0:
        print(f'There is no communication between {sat_list[sat_no_2].name} with {sat_list[sat_no_1].name} ')
        return
    else: 
        print(f"Detailed information of all line of sight periods for {sat_list[sat_no_2].name} with {sat_list[sat_no_1].name}:")
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('Period - Starttime -------------------- Endtime ------------------ Duration --- max Rotspeed -- max Distance -- max Speed -')
        print('---------------------------------------------------------------------------------------------------------------------------')
        for index,period in enumerate(list_of_intervals):
            start,end = period
            end = end + 1
            dt = (end - start)*time_step
            print(f"No. {index} \t from {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}    {dt} sek \t{max_anglespeed[index]:.4f} deg/s \t{max_distance[index]:.2f} km \t{max_velo[index]:.2f} km/s")
    return 

def print_contact_info(sat_list,sat_no_1,sat_no_2,sim_start,time_step,max_anglespeed,max_distance,max_velo):
    """
    This function prints intervals from line of sight of satellite x and y.

    :param sat_list: list of objects of satellites
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite
    :type sat_no_2: int
    :param S_year: year from start time
    :param S_month: month for the start time
    :param S_day: day for the start time
    :param S_hour: hour for the start time
    :param S_min: minute for the start time
    :param S_sec: second for the start time
    :param max_anglespeed: list of max medium anglespeeds of periods
    """
    list_of_intervals = sat_list[sat_no_1].contacts[sat_no_2]
    if len(list_of_intervals) == 0:
        print(f'There is no communication between {sat_list[sat_no_2].name} with {sat_list[sat_no_1].name} ')
        return
    else: 
        print(f"Detailed information to the communication periods for {sat_list[sat_no_2].name} with {sat_list[sat_no_1].name}:")
        print('---------------------------------------------------------------------------------------------------------------------------')
        print('Period - Starttime -------------------- Endtime ------------------ Duration --- max Rotspeed -- max Distance -- max Speed -')
        print('---------------------------------------------------------------------------------------------------------------------------')
        for index,period in enumerate(list_of_intervals):
            start,end = period
            dt = (end - start)*time_step
            print(f"No. {index} \t from {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}    {dt} sek \t{max_anglespeed[index]:.4f} deg/s \t{max_distance[index]:.2f} km \t{max_velo[index]:.2f} km/s")
    return 

def all_los_periods(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function parses the sequence of line of sight into a list of tuple.

    :param sat_list: list of objects of the Satellite class
    :type sat_list: list[Satellite]
    :param sat_no_1,sat_no_2: number of satellite to examined
    :type sat_no_1,sat_no_2: int
    :return: list of start and end timepoint for los 
    """
    list_of_intervals = find_continuous_sequences(get_position_of_true(sat_list[sat_no_1].los[sat_no_2]))
    return list_of_intervals

def all_communication_periods(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function parses the sequence of communication into a list of tuple.

    :param sat_list: list of objects of the Satellite class
    :type sat_list: list[Satellite]
    :param sat_no_1,sat_no_2: number of satellite to examined
    :type sat_no_1,sat_no_2: int
    :return: list of start and end timepoint for los 
    """
    list_of_intervals = sat_list[sat_no_1].contacts[sat_no_2]
    return list_of_intervals

def los_periods(sat_list:list[Satellite]):
    """
    This function parses the sequence of line of sight into a list of tuple.

    :param sat_list: list of objects of the Satellite class
    :type sat_list: list[Satellite]
    """
    for i in range(len(sat_list)):
        sat_list[i].periods = [] 
        for j in range(len(sat_list)):
            list_of_intervals = find_continuous_sequences_01(get_position_of_true(sat_list[i].los[j]))
            sat_list[i].periods.append(list_of_intervals)

def plot_los_with_earth(sat_list:list[Satellite],sat_no_1,sat_no_2,list_of_intervals,period):
    """
    This function visualises the position of satellites created with the function 
    pos_determination. Interactive plot is 

    :param sat_list: list of objects of the Satellite class.
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite
    :type sat_no_2: int
    :param period: specific intervall from list of intervall. (Value starts at 1)
    """
    max_range = 60000
    names = name_determination(sat_list) # create list of satellite names for plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]]) 

    if len(list_of_intervals) == 0:
        print('No line of sight available!')
        return
    else: 
        a = b = 6356.752  # Halbachse in x,y-Richtung
        c = 6378.137  # Halbachse in z-Richtung
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x0 = a * np.outer(np.cos(u), np.sin(v))
        y0 = b * np.outer(np.sin(u), np.sin(v))
        z0 = c * np.outer(np.ones_like(u), np.cos(v))

        start,end = list_of_intervals[period]

        x1 = sat_list[sat_no_1].x[start:end]
        y1 = sat_list[sat_no_1].y[start:end]
        z1 = sat_list[sat_no_1].z[start:end]
        x2 = sat_list[sat_no_2].x[start:end]
        y2 = sat_list[sat_no_2].y[start:end]
        z2 = sat_list[sat_no_2].z[start:end]
        fig.add_trace(go.Scatter3d(x=x0.flatten(), y=y0.flatten(), z=z0.flatten(), mode='markers', name='Earth', marker=dict(size=1, color='blue')))
        fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines',name=names[sat_no_1]), row=1, col=1) # create orbit
        fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines',name=names[sat_no_2]), row=1, col=1) # create orbit
        fig.update_layout(scene=dict(xaxis_title='X (km)', 
                                    yaxis_title='Y (km)',
                                    zaxis_title='Z (km)',
                                    aspectmode='cube',
                                    xaxis=dict(range=[-max_range, max_range]),  # arrange plot axis for identical scales 
                                    yaxis=dict(range=[-max_range, max_range]), 
                                    zaxis=dict(range=[-max_range, max_range]),
                                    ),
                        title='Specific Line of Sight Period')                                                  
    fig.show()

def plot_commmunication_period_with_earth(sat_list:list[Satellite],orientation,sat_no_1:int,sat_no_2:int,period:int):
    """
    This function visualises the position of satellites created with the function 
    pos_determination. Interactive plot is 

    :param sat_list: list of objects of the Satellite class.
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite
    :type sat_no_1,2: int
    :param sat_no_2: Number of secound satellite
    :param period: specific intervall from list of intervall. (Value starts at 1)
    :type period: int
    """
    max_range = 60000
    names = name_determination(sat_list) # create list of satellite names for plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]]) 
    if orientation == 'A':
        list_of_intervals = sat_list[sat_no_1].periods[sat_no_2]
    else:
        list_of_intervals = sat_list[sat_no_1].contacts[sat_no_2]
    if len(list_of_intervals) == 0:
        print('No line of sight available!')
        return
    else: 
        a = b = 6356.752  # Halbachse in x,y-Richtung
        c = 6378.137  # Halbachse in z-Richtung
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x0 = a * np.outer(np.cos(u), np.sin(v))
        y0 = b * np.outer(np.sin(u), np.sin(v))
        z0 = c * np.outer(np.ones_like(u), np.cos(v))

        start,end = list_of_intervals[period-1]

        x1 = sat_list[sat_no_1].x[start:end]
        y1 = sat_list[sat_no_1].y[start:end]
        z1 = sat_list[sat_no_1].z[start:end]
        x2 = sat_list[sat_no_2].x[start:end]
        y2 = sat_list[sat_no_2].y[start:end]
        z2 = sat_list[sat_no_2].z[start:end]
        fig.add_trace(go.Scatter3d(x=x0.flatten(), y=y0.flatten(), z=z0.flatten(), mode='markers', name='Earth', marker=dict(size=1, color='blue')))
        fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines',name=names[sat_no_1]), row=1, col=1) # create orbit
        fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines',name=names[sat_no_2]), row=1, col=1) # create orbit
        fig.update_layout(scene=dict(xaxis_title='X (km)', 
                                    yaxis_title='Y (km)',
                                    zaxis_title='Z (km)',
                                    aspectmode='cube',
                                    xaxis=dict(range=[-max_range, max_range]),  # arrange plot axis for identical scales 
                                    yaxis=dict(range=[-max_range, max_range]), 
                                    zaxis=dict(range=[-max_range, max_range]),
                                    ),
                        title='Specific Line of Sight Period')                                                  
    fig.show()


def intersatellite_vector(sat_list:list[Satellite]): 
    """
    This function builds a vector between two satellites and saves them in the attribute 'dirvec'.

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :return sat_list: list of satellite objects with attribute 'divec'
    :rtype: list
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)

    for sat in sat_list:
        sat.divec = [[] for _ in range(len_cal)]  # Initialize a list for each satellite at each time step
    
    for x in range(len_cal):
        for i in range(num_satellites):
            for j in range(num_satellites):
                if i == j:
                    sat_list[i].divec[x].append('self')
                else:
                    x_coord = sat_list[j].x[x] - sat_list[i].x[x]
                    y_coord = sat_list[j].y[x] - sat_list[i].y[x]
                    z_coord = sat_list[j].z[x] - sat_list[i].z[x]
                    direction_vector = (x_coord, y_coord, z_coord)
                    sat_list[i].divec[x].append(direction_vector)
    
    return sat_list

def angle_speed(sat_list:list[Satellite],time_step:int):
    """
    This function determines the medium angular speed per timestep of two vektors of 
    satellites from sat_list. Last value of list is always empty due to interstep 
    calculation.

    :param sat_list: list of objects of the Satellite class.
    :type sat_list: list[Satellite]
    :param time_step: timestep for 
    :type time_step: int
    :return sat_list: list of objects of the Satellite class with medium anglespeed
    :rtype: list
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)

    for sat in sat_list:
        sat.anglespeed = [[] for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
            for x in range(len_cal-1):
                if i == j:
                    speed = 'self'
                else:
                    A = sat_list[i].divec[x][j]
                    B = sat_list[i].divec[x+1][j]
                    dot_product = np.dot(A, B)
                    A = np.linalg.norm(A)
                    B = np.linalg.norm(B)
                    cos_theta= dot_product / (A * B)
                    angle = np.arccos(cos_theta)
                    angle = np.rad2deg(angle)
                    speed = angle/time_step
                sat_list[i].anglespeed[j].append(speed)
    
    return sat_list

def max_angular_speed_los(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This fuction determines the max medium angular speed of all line of sight periods.

    :param sat_list: list of objects of the Satellite class with medium anglespeed (.anglespeed)
    :param sat_no_1: Number of first satellite
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite
    :type sat_no_2: int
    :param period: specific intervall from list of intervall. (Value starts at 1)
    :type period: tuple
    :return max_speed: list of maximum medium angular speed per timestep, 
    :rtype: list
    """
    period_list = iteration_list_extraction_01(sat_list[sat_no_2].anglespeed[sat_no_1],periods)
    max_speed = []
    for speed in period_list:
        max_speed.append(max(speed))
    return max_speed

def distance_for_los(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function determines the relative distance between two satellites while the line of sight.

    :param sat_list: list of satellites objects
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of first satellite
    :type sat_no_1: int
    :param sat_no_2: Number of secound satellite
    :type sat_no_2: int
    :param periods: List of start and end point of los
    :type periods: list[tuple]
    :return dis_list: List of distance in line of sight
    :rtype: list
    """
    dis_list = []
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    for index0,los in enumerate(x0_los):
        dis = []
        for index1, x in enumerate(los): 
            pos0 = x0_los[index0][index1],y0_los[index0][index1],z0_los[index0][index1]
            pos1 = x1_los[index0][index1],y1_los[index0][index1],z1_los[index0][index1]
            distance = calculate_distance(pos0,pos1)
            dis.append(distance)
        dis_list.append(dis)
    return dis_list

def get_maximum_los_list(los_list):
    """
    This function returns the maximum value out for every period in a line of sight list. 

    :param los_list: 
    :return max_per_los:
    """
    max_per_los = []
    for period in los_list: 
        max_per_los.append(max(period))
    return max_per_los

def distance(sat_list:list[Satellite]):
    """
    This function calculates the distance between all satellites in every timestep

    :param sat_list: list of satellites objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)

    for sat in sat_list:
        sat.distance = [[] for _ in range(num_satellites)] 

    for i in range(num_satellites):
        for j in range(num_satellites):
            for x in range(len_cal):
                    pos0 = sat_list[i].x[x],sat_list[i].y[x],sat_list[i].z[x]
                    pos1 = sat_list[j].x[x],sat_list[j].y[x],sat_list[j].z[x]
                    distance = calculate_distance(pos0,pos1)
                    sat_list[i].distance[j].append(distance)
    return sat_list

def data_per_period(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function plotes the rotation speed of the satellites per period. 
    
    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one 
    :type sat_no_1: int
    :param sat_no_2: number of satellite two 
    :type sat_no_2: int
    :param periods: number of line of sight period
    :return anglespeed: list of anglespeed per period
    :rtype: list
    :return distance: list of distance per period 
    :rtype: list
    :return velo: list of velocity per period
    :rtype: list
    """
    anglespeed = iteration_list_extraction_01(sat_list[sat_no_1].anglespeed[sat_no_2],periods)
    distance = iteration_list_extraction_01(sat_list[sat_no_1].distance[sat_no_2],periods)
    velo = iteration_list_extraction_01(sat_list[sat_no_1].doppler[sat_no_2],periods)

    return anglespeed,distance,velo

def data_per_communication_period(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function plotes the rotation speed of the satellites per period. 
    
    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one 
    :type sat_no_1: int
    :param sat_no_2: number of satellite two 
    :type sat_no_2: int
    :param periods: number of line of sight period
    :type periods: int
    :return anglespeed: list of anglespeed per period
    :rtype: list
    :return distance: list of distance per period 
    :rtype: list
    :return velo: list of velocity per period
    :rtype: list
    """
    periods = sat_list[sat_no_1].contacts[sat_no_2]
    anglespeed = iteration_list_extraction_01(sat_list[sat_no_1].anglespeed[sat_no_2],periods)
    distance = iteration_list_extraction_01(sat_list[sat_no_1].distance[sat_no_2],periods)
    velo = iteration_list_extraction_01(sat_list[sat_no_1].doppler[sat_no_2],periods)

    return anglespeed,distance,velo

def plot_period(anglespeeds,distances,velos,thetas,time_step,period):
    """
    This funciton plottes important informations about a given contract period. 

    :param anglespeeds: list of anglespeed per period
    :type anglespeeds: list
    :param distances: list of distance per period 
    :type distnces: list 
    :param velos: list of relative speed per period
    :type velos: list
    :param thetas: Angle between borside and satellite
    :type thetas: float or int
    """
    length = range(0,len(distances[period]))
    length_angle = range(0,len(anglespeeds[period]))
    fig,axes = plt.subplots(2, 2)
    
    l = len(length)
    if thetas == []:
        thetas = np.zeros((l,l))

    axes[0, 0].plot(length_angle,anglespeeds[period])
    axes[0, 0].set_title('Rotation Speed')
    axes[0, 0].set_xlabel(f't in timestep [{time_step}s]')
    axes[0, 0].set_ylabel('deg/s')
    axes[0, 0].grid(True)

    axes[0, 1].plot(length,distances[period])
    axes[0, 1].set_title('Distance')
    axes[0, 1].set_xlabel(f't in timestep [{time_step}s]')
    axes[0, 1].set_ylabel('km')
    axes[0, 1].grid(True)

    axes[1, 0].plot(length,velos[period])
    axes[1, 0].set_title('Relative Speed')
    axes[1, 0].set_xlabel(f't in timestep [{time_step}s]')
    axes[1, 0].set_ylabel('km/s')
    axes[1, 0].grid(True)  

    axes[1, 1].plot(length,thetas[period])
    axes[1, 1].set_title('Theta')
    axes[1, 1].set_xlabel(f't in timestep [{time_step}s]')
    axes[1, 1].set_ylabel('deg')
    axes[1, 1].grid(True) 
    
    plt.tight_layout() 
    plt.savefig(f"parameters_of_period.pdf", format='pdf') # export for thesis
    plt.show()
 

def angle_between_vectors(vec0,vec1):
    """
    This function determines a angle between vectors.

    :param vec0: vector one
    :param vec1: vector two 
    :return theta: angle between vector
    """ 
    x0,y0,z0 = vec0
    x1,y1,z1 = vec1
    theta = np.arccos((x0*x1+y0*y1+z0*z1)/(np.sqrt(x0**2+y0**2+z0**2)*np.sqrt(x1**2+y1**2+z1**2)))   
    return theta

def nadir_theta(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between nadir pointing
    of satellite one and the vector between satellite one and two
    for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec0 = -x0_los[index0][index1],-y0_los[index0][index1],-z0_los[index0][index1]
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def opposite_nadir_theta(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between opposite nadir pointing 
    of satellite one and the vector between satellite one and two for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec0 = x0_los[index0][index1],y0_los[index0][index1],z0_los[index0][index1]
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def ahead_theta(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between ahead pointing (in respect to the flight direction)
    of satellite one and the vector between satellite one and two for every los period.  

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    xyz_los = iteration_list_extraction_01(sat_list[sat_no_1].flightvecs,periods)

    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec0 = xyz_los[index0][index1][0],xyz_los[index0][index1][1],xyz_los[index0][index1][2]
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def backwards_theta(sat_list,sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between backwards pointing (in respect to flight direction)
    of satellite one and the vector between satellite one and two for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    xyz_los = iteration_list_extraction_01(sat_list[sat_no_1].flightvecs,periods)

    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec0 = -xyz_los[index0][index1][0],-xyz_los[index0][index1][1],-xyz_los[index0][index1][2]
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def left_theta(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between satellite one and two
    for every los period.  

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    xyz_los = iteration_list_extraction_01(sat_list[sat_no_1].flightvecs,periods)

    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec01 = xyz_los[index0][index1][0],xyz_los[index0][index1][1],xyz_los[index0][index1][2]
            vec02 = -x0_los[index0][index1],-y0_los[index0][index1],-z0_los[index0][index1]
            vec0 = cross_product(vec01,vec02)
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def right_theta(sat_list:list[Satellite],sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between 90° right (in respect to flight direction) 
    orientated pointing of the antenna and the vector between satellite one and two. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    xyz_los = iteration_list_extraction_01(sat_list[sat_no_1].flightvecs,periods)

    theta_list = []
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec01 = xyz_los[index0][index1][0],xyz_los[index0][index1][1],xyz_los[index0][index1][2]
            vec02 = x0_los[index0][index1],y0_los[index0][index1],z0_los[index0][index1]
            vec0 = cross_product(vec01,vec02)
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def custom_vector_theta(sat_list:list[Satellite],pointing_vector,sat_no_1,sat_no_2,periods):
    """
    This function calculates the angle between a pointing vector of the to 
    examined satellite and the vector between the two satellites
    for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param pointing_vector: vector of antenna orientation
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :param periods: list of contact periods, start and end timestep
    :type periods: list[tuple]
    :return theta_list: list of angles for periods
    :rtype: list 
    """
    x0_los = iteration_list_extraction_01(sat_list[sat_no_1].x,periods)
    y0_los = iteration_list_extraction_01(sat_list[sat_no_1].y,periods)
    z0_los = iteration_list_extraction_01(sat_list[sat_no_1].z,periods)
    x1_los = iteration_list_extraction_01(sat_list[sat_no_2].x,periods)
    y1_los = iteration_list_extraction_01(sat_list[sat_no_2].y,periods)
    z1_los = iteration_list_extraction_01(sat_list[sat_no_2].z,periods)
    theta_list = []
    x,y,z = pointing_vector
    for index0,los in enumerate(x0_los):
        angles = []
        for index1, x in enumerate(los): 
            vec0 = x,y,z
            vec1 = x1_los[index0][index1]-x0_los[index0][index1],y1_los[index0][index1]-y0_los[index0][index1],z1_los[index0][index1]-z0_los[index0][index1]
            theta = np.rad2deg(angle_between_vectors(vec0,vec1))
            angles.append(theta)
        theta_list.append(angles)
    return theta_list

def nadir_theta_full(sat_list:list[Satellite],sat_no_1:int,sat_no_2:int):
    """
    This function calculates the angle between nadir pointing
    of satellite one and the vector between satellite one and two
    for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec0 = -sat_list[sat_no_1].x[i],-sat_list[sat_no_1].y[i],-sat_list[sat_no_1].z[i]
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def opposite_nadir_theta_full(sat_list,sat_no_1,sat_no_2):
    """
    This function calculates the angle between opposite nadir pointing 
    of satellite one and the vector between satellite one and two for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec0 = sat_list[sat_no_1].x[i],sat_list[sat_no_1].y[i],sat_list[sat_no_1].z[i]
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def ahead_theta_full(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function calculates the angle between ahead pointing (in respect to the flight direction)
    of satellite one and the vector between satellite one and two for every los period.  

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec0 = sat_list[sat_no_1].flightvecs[i][0],sat_list[sat_no_1].flightvecs[i][1],sat_list[sat_no_1].flightvecs[i][2]
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def backwards_theta_full(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function calculates the angle between backwards pointing (in respect to flight direction)
    of satellite one and the vector between satellite one and two for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec0 = -sat_list[sat_no_1].flightvecs[i][0],-sat_list[sat_no_1].flightvecs[i][1],-sat_list[sat_no_1].flightvecs[i][2]
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def left_theta_full(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between satellite one and two
    for every los period.  

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec01 = sat_list[sat_no_1].flightvecs[i][0],sat_list[sat_no_1].flightvecs[i][1],sat_list[sat_no_1].flightvecs[i][2]
        vec02 = -sat_list[sat_no_1].x[i],-sat_list[sat_no_1].y[i],-sat_list[sat_no_1].z[i]
        vec0 = cross_product(vec01,vec02)
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def right_theta_full(sat_list:list[Satellite],sat_no_1,sat_no_2):
    """
    This function calculates the angle between 90° right (in respect to flight direction) 
    orientated pointing of the antenna and the vector between satellite one and two. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_no_1: number of satellite one
    :type sat_no_1: int
    :param sat_no_2: number of satellite two
    :type sat_no_2: int
    :return theta_list: list of angles
    :rtype: list 
    """
    theta_list = []
    for i in range(len(sat_list[0].x)):
        vec01 = sat_list[sat_no_1].flightvecs[i][0],sat_list[sat_no_1].flightvecs[i][1],sat_list[sat_no_1].flightvecs[i][2]
        vec02 = sat_list[sat_no_1].x[i],sat_list[sat_no_1].y[i],sat_list[sat_no_1].z[i]
        vec0 = cross_product(vec01,vec02)
        vec1 = sat_list[sat_no_2].x[i]-sat_list[sat_no_1].x[i],sat_list[sat_no_2].y[i]-sat_list[sat_no_1].y[i],sat_list[sat_no_2].z[i]-sat_list[sat_no_1].z[i]
        theta = np.rad2deg(angle_between_vectors(vec0,vec1))
        if theta > sat_list[sat_no_1].theta_3dB:
            theta_list.append(False)
        else: 
            theta_list.append(True)
    return theta_list

def left_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = left_theta_full(sat_list,i,j)

def right_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° right (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every los period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = right_theta_full(sat_list,i,j)

def nadir_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = nadir_theta_full(sat_list,i,j)

def opposite_nadir_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = opposite_nadir_theta_full(sat_list,i,j)

def ahead_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = ahead_theta_full(sat_list,i,j)


def backwards_theta_loop_full(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = backwards_theta_full(sat_list,i,j)

def angle_check(theta_3dB,theta_list):
    """
    This function checks when theta get's smaller than theta 3dB and returnes a contact list. 

    :param theta_3dB: Theta 3 dB from the antenna
    :type theta_3dB: float
    :param theta_list: List of angles between antenna direction and satellite.
    :type theta_list: list[float]
    :return: Threshold whether connection is possible.
    :rtype: list[bool]
    """
    threshold = [[] for _ in range(len(theta_list))]
    for index0,los in enumerate(theta_list):
        for index1,x in enumerate(los):
            if theta_list[index0][index1] > theta_3dB:
                threshold[index0].append(False)
            else:
                threshold[index0].append(True)
    return threshold

def plot_antenna_contact(sat_list:list[Satellite],sat_no_1,sat_no_2,contact,orientation,period,sim_start,time_step):
    """
    This funciton plottes important informations about a given contract period. 

    :param anglespeed: List of anglespeed per period.
    :type anglespeed: list[float]
    :param distance: List of distance per period.
    :type distance: list[float]
    :param velo: List of velocity per period.
    :type velo: list[float]
    :param theta: Angle between boresight and satellite.
    :type theta: float
    """
    if len(contact) == 0:
        print('no special antenna contact for omnidirectional case')
    else:
        counter = 0
        for i in contact[period-1]:
            if i == 1:
                counter += 1
        length = range(0,len(contact[period-1]))

        if counter != 0:
            plt.bar(length,contact[period-1],align='edge', width=1)
            plt.title(f"Communication connection for case '{orientation}'")
            plt.xlabel('t in timestep')
            plt.ylabel('Boolean')
            plt.tight_layout() 
            plt.show()
            print(f'The possible conntection phase of {sat_list[sat_no_1].name} and {sat_list[sat_no_2].name}. \n For a more detailed solution set the timestep down.')
            periods = get_position_of_ones(contact[period-1])
            periods = find_continuous_sequences_01(periods)
            for k in range(len(periods)):
                start,end = periods[k]
                end = end+1
                if k == 0:
                    print(f' The connection goes from {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()} [{(end-start)*time_step} sek]')
                else:  
                    print(f'                 and from {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()} [{(end-start)*time_step} sek]')
        else:
            print(f' It is no possible antenna conntection phase of {sat_list[sat_no_1].name} and {sat_list[sat_no_2].name} \nin contract period no {period}.')

def contact_periods_between_all_sat(sat_list:list[Satellite],sat_no_1):
    """
    This function shows all possible line of sights between all satellites and one selected. 

    :param sat_list: A list of satellite objects.
    :type sat_list: list[Satellite]
    :param sat_no_1: The number of the satellite to be examined.
    :type sat_no_1: int
    """
    print('--------------------------------------------------------------------------------------------')
    print(f'For the given Epoch, following satellites have contact periods with {sat_list[sat_no_1].name}')
    print('--------------------------------------------------------------------------------------------')
    for index in range(len(sat_list[sat_no_1].los)):
        if index == sat_no_1:
            continue
        else:
            if any(sat.los[index] for sat in sat_list):
                print(f'Line of sight possible between {sat_list[sat_no_1].name} and {sat_list[index].name} (Sat No. {index})')
            else:
                print(f'No Line of sight is possible between {sat_list[sat_no_1].name} and {sat_list[index].name} (Sat No. {index})')
    return 

def pointing_direction_01(sat_list:list[Satellite],orientation):
    """
    This function determines which direction the satellite should be orientated for examination. 

    :param sat_list: A list of satellites with attribute periods.
    :type sat_list: list[Satellite]
    :param orientation: A string representing the orientation ('A', 'B', 'C', 'D', 'E', 'F', 'G').
    :type orientation: str
    """
    # Case A: Omnidirectional
    # Case B: flight direction
    # Case C: opposite flight direction
    # Case D: nadir pointing 
    # Case E: opposite nadir pointing
    # Case F: left in respect to flight direction 
    # Case G: right in respect to flight direction
    if orientation == 'A':
        pass
    elif orientation == 'B':
        ahead_theta_loop(sat_list)
    elif orientation == 'C':
        backwards_theta_loop(sat_list)
    elif orientation == 'D':
        nadir_theta_loop(sat_list)
    elif orientation == 'E':
        opposite_nadir_theta_loop(sat_list)
    elif orientation == 'F':
        left_theta_loop(sat_list)
    elif orientation == 'G':
        right_theta_loop(sat_list)
    return 

def pointing_direction_02(sat_list:list[Satellite],orientation,sat_no_1,sat_no_2,periods):
    """
    This function determines which satellite pointing mode should be assumed. 

    :param sat_list: A list of Satellite objects.
    :type sat_list: list[Satellite]
    :param orientation: A string representing the antenna attitude ('A', 'B', 'C', 'D', 'E', 'F', 'G').
    :type orientation: str
    :param sat_no_1: The number of the first satellite.
    :type sat_no_1: int
    :param sat_no_2: The number of the second satellite.
    :type sat_no_2: int
    :param periods: A list of line-of-sight (LOS) periods.
    :type periods: list
    :return: A list of theta values for every period stored in lists.
    :rtype: list[list]
    """
    if orientation == 'A':
        theta_list = []
    elif orientation == 'B':
        theta_list = ahead_theta(sat_list,sat_no_1,sat_no_2,periods)
    elif orientation == 'C':
        theta_list = backwards_theta(sat_list,sat_no_1,sat_no_2,periods)
    elif orientation == 'D':
        theta_list = nadir_theta(sat_list,sat_no_1,sat_no_2,periods)
    elif orientation == 'E':
        theta_list = opposite_nadir_theta(sat_list,sat_no_1,sat_no_2,periods)
    elif orientation == 'F':
        theta_list = left_theta(sat_list,sat_no_1,sat_no_2,periods)
    elif orientation == 'G':
        theta_list = right_theta(sat_list,sat_no_1,sat_no_2,periods)
    return theta_list

def pointing_direction_03(sat_list:list[Satellite],orientation):
    """
    This function determines which direction the satellite should be orientated for examination. 

    :param sat_list: A list of satellites with attribute periods.
    :type sat_list: list[Satellite]
    :param orientation: A string representing the antenna attitude (e.g., 'A', 'B', 'C', 'D', 'E', 'F', 'G').
    :type orientation: str
    """
    # Case A: Omnidirectional
    # Case B: flight direction
    # Case C: opposite flight direction
    # Case D: nadir pointing 
    # Case E: opposite nadir pointing
    # Case F: left in respect to flight direction 
    # Case G: right in respect to flight direction
    if orientation == 'A':
        pass
    elif orientation == 'B':
        ahead_theta_loop_full(sat_list)
    elif orientation == 'C':
        backwards_theta_loop_full(sat_list)
    elif orientation == 'D':
        nadir_theta_loop_full(sat_list)
    elif orientation == 'E':
        opposite_nadir_theta_loop_full(sat_list)
    elif orientation == 'F':
        left_theta_loop_full(sat_list)
    elif orientation == 'G':
        right_theta_loop_full(sat_list)
    return 

def flight_vector(sat_list:list[Satellite]):
    """
    This function calculates the unit vector of direction for 
    every satellite in every timestep.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    len_cal = len(sat_list[0].x)
    
    for sat in sat_list:
        sat.flightvecs = [] 

    for i in range(num_satellites):
        for x in range(len_cal):
                norm = np.sqrt(sat_list[i].velo[x][0]**2 + sat_list[i].velo[x][1]**2 + sat_list[i].velo[x][2]**2)
                a = 1/norm * sat_list[i].velo[x][0]
                b = 1/norm * sat_list[i].velo[x][1]
                c = 1/norm * sat_list[i].velo[x][2]
                l = [a,b,c]
                sat_list[i].flightvecs.append(l)

def cross_product(vec0,vec1):
    """
    This function calculates the cross product of two vectors.

    :param vec0: The first vector.
    :type vec0: list[float]
    :param vec1: The second vector.
    :type vec1: list[float]
    :return: The orthogonal vector.
    :rtype: list[float]
    """
    x0,y0,z0 = vec0
    x1,y1,z1 = vec1
    x = y0 * z1 - z0 - y1 
    y = z0 * x1 - x0 * z1
    z = x0 * y1 - y0 * x1
    vector = x,y,z
    return vector

def left_theta_loop(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every los period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = left_theta(sat_list,i,j,sat_list[i].periods[j])


def right_theta_loop(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° right (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every los period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = right_theta(sat_list,i,j,sat_list[i].periods[j])

def nadir_theta_loop(sat_list):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = nadir_theta(sat_list,i,j,sat_list[i].periods[j])

def opposite_nadir_theta_loop(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = opposite_nadir_theta(sat_list,i,j,sat_list[i].periods[j])

def ahead_theta_loop(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = ahead_theta(sat_list,i,j,sat_list[i].periods[j])

def backwards_theta_loop(sat_list:list[Satellite]):
    """
    This function calculates the angle between 90° left (in respect to flight direction) 
    orientated pointing of the antenna and the vector between every satellite for every period. 

    :param sat_list: list of satellites with attribute periods
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)

    for sat in sat_list:
        sat.theta = [[] for _ in range(num_satellites)]

    for i in range(num_satellites):
        for j in range(num_satellites):
            if i == j:
                continue
            else:
                sat_list[i].theta[j] = backwards_theta(sat_list,i,j,sat_list[i].periods[j])

def is_communication(sat_list:list[Satellite], theta_3dB, sat_number):
    """
    This function checks if communication is possible between all satellites
    from tle file and a satellite to investigate (sat_number)
    are able to communicate with a satellite under investigation.

    :param sat_list: list of satellite objects with attribute theta
    :param sat_number: number of satellite to investigate
    :param theta_3dB: theta_3dB / 2
    :return threshold: neasted list with 1 for contact and 0 for no contact
    """
    num_satellites = len(sat_list)
    threshold = [[] for _ in range(num_satellites)]

    for sat in range(num_satellites):
        if sat == sat_number:
            continue
        elif sat_list[sat_number].theta is not None:
            threshold[sat] = angle_check(theta_3dB,sat_list[sat_number].theta[sat])
        else:
            threshold[sat] = []
    return threshold

def list_counter(list:list):
    """
    This function counts every 1 in a list and returns the amound.

    :param list: list with or without 1
    :type list: list
    :return counter: amound of 1.
    :rtype: int
    """
    counter = 0
    for i in list:
        if i == True:
            counter += 1
    return counter 

def get_position_of_ones(list:list):
    """
    This function returnes the position of every 1, out of a list. 

    :param list: list of digits
    :type list: list
    :return positions: list of positions of 1
    :rtype: list
    """
    digit = 1
    positions = [index for index, value in enumerate(list) if value == digit]
    return positions

def threshold_filter(sat_list:list[Satellite],sat_number,sim_start,time_step,orientation,theta_3dB):
    """
    This function analyzes the contact possibility between the investigation satellite (sat_number) 
    and other satellites in the sat_list for various orientations. It prints the intervals of 
    potential contact based on the provided threshold periods.

    :param sat_list: List of Satellite objects.
    :type sat_list: list
    :param sat_number: Number of the investigation satellite.
    :type sat_number: int
    :param time_step: Time step in seconds.
    :type time_step: int
    :param orientation: Orientation case (A, B, C, D, E, F, G).
    :type orientation: str
    :param threshold: List of contract periods for special antenna orientation case.
    :type threshold: list of tuples
    """
    num_satellites = len(sat_list)

    if orientation == 'A':
        orien = 'omnidirectional'
        print('----------------------------------------------------------------------------------------')
        print(f' This programm calculates every cconnection interval of {sat_list[sat_number].name}')
        print(f' with every satellite which is defined in the tle file.')
        print(f' Assumptions:')
        print(f' - orientation of antenna {orien} ')
        print('----------------------------------------------------------------------------------------')
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                list_of_intervals = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].los[sat]))
                if len(list_of_intervals) == 0:
                    print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                    print('  No match')
                    continue
                else: 
                    print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                    for period in range(len(list_of_intervals)): 
                        start,end = list_of_intervals[period]
                        dt = end - start + 1
                        print(f'  Match in period No. {period+1} \t Duration of {dt*time_step} sek \t', end ='')
                        print(f' {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end+1).utc_strftime()}')
    else: 
        if orientation == 'B':
            orien = 'in flight direction'
        elif orientation == 'C':
            orien = 'in opposite flight direction'
        elif orientation == 'D':
            orien = 'as nadir pointing'
        elif orientation == 'E':
            orien = 'as starpointing'
        elif orientation == 'F':
            orien = 'to left hand side in flight direction'
        elif orientation == 'G':
            orien = 'to right hand side in flight direction'
        print('----------------------------------------------------------------------------------------')
        print(f' This programm calculates every connection interval of {sat_list[sat_number].name}')
        print(f' with every satellite which is defined in the tle file.')
        print(f' Assumptions:')
        print(f' - orientation of antenna {orien} ')
        print(f' - theta 3dB of {theta_3dB*2}°')
        print('----------------------------------------------------------------------------------------')
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] andd {sat_list[sat_number].name}')
                list_of_intervals_theta = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].theta[sat]))
                list_of_intervals_los = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].los[sat]))
                list_of_intervals = find_overlapping_intervals(list_of_intervals_theta,list_of_intervals_los)
                checker = len(list_of_intervals)
                if checker > 0:
                    for i, period in enumerate(list_of_intervals):
                        start,end = period
                        end = end+1
                        print(f'  Match in period No. {i+1} \t Duration of {(end-start)*time_step} sek \t', end ='')
                        print(f'   {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()} ')
                        
                else: 
                    print('  No match')
 
def modulate_theta_3dB(theta_3dB):
    """
    This function halves the value of theta 3dB.

    :param theta_3dB: Theta 3 dB.
    :type theta_3dB: float
    :return: Halved theta.
    :rtype: float
    """
    theta_3dB /= 2
    return theta_3dB

def modulate_sat_number(sat_number):
    """
    This function subtracts 1 from the satellite number.

    :param sat_number: Satellite number.
    :type sat_number: int
    :return: Modulated number.
    :rtype: int
    """
    sat_number -= 1
    return sat_number

def point_to_sphere_distances(sat_list:list[Satellite],sat_no_1,sat_no_2,periods,period):
    """
    This function calculates the distance of x points of a vector with the nearest point on the 
    surface of an ellipsoid.

    :param sat_list: List of objects of satellites with points of vector.
    :type sat_list: list[Satellite]
    :param sat_no_1: Number of target satellite.
    :type sat_no_1: int
    :param sat_no_2: Number of malicious satellite.
    :type sat_no_2: int
    :param periods: List of contact periods.
    :type periods: list[tuple]
    :param period: Number of specific period (starts from 1).
    :type period: int
    :return mesh: Matrix of heights.
    :rtype: list[list[float]]
    """
    earth_radius = 6370   # Halbachse in x-Richtung
    num_points_on_line = 10000 # actually one more, defined in loop
    start, end = periods[period]
    for sat in sat_list:
        sat.height = []
    mesh = []
    for x in range(start,end+1):
        points_on_line = []
        for k in range(1+num_points_on_line):
            x_coord = sat_list[sat_no_1].x[x] + (sat_list[sat_no_2].x[x] - sat_list[sat_no_1].x[x]) * k / num_points_on_line
            y_coord = sat_list[sat_no_1].y[x] + (sat_list[sat_no_2].y[x] - sat_list[sat_no_1].y[x]) * k / num_points_on_line 
            z_coord = sat_list[sat_no_1].z[x] + (sat_list[sat_no_2].z[x] - sat_list[sat_no_1].z[x]) * k / num_points_on_line      
            length = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
            height = length - earth_radius
            points_on_line.append(height)
        mesh.append(points_on_line)

    return mesh

def plot_communication(heights):
    """
    This function plotes the height of the line of sight over a contactperiod. 
    Works with point to sphere distance.

    :param heights: matrix of heights for every timestep in period
    :type heights: numpy.ndarray
    """
    data = heights
    data = np.array(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    Z = data
    surface = ax.plot_surface(X, Y, Z, cmap=plt.cm.Spectral, alpha=0.9)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Timesteps')
    ax.set_zlabel('Height over 6370km')

    ax.set_xticks([0, len(data[0])], ['0', '1'])
    ax.xaxis.labelpad = -5

    ax.contourf(X, Y, Z, zdir='x', offset=-cols/5,cmap='autumn')
    ax.contourf(X, Y, Z, zdir='y', offset=rows+rows/5.5,cmap='autumn')

    colorbar = fig.colorbar(surface, ax=ax, pad=0.1) 
    plt.show()

def plot_communication_interactive(heights,period):
    """
    This function plotes the height of the line of sight over a contactperiod. 
    Works with point to sphere distance.

    :param heights: Matrix of heights for every timestep in the period.
    :type heights: numpy.ndarray
    :param period: The number of the specific period.
    :type period: int
    """
    data = heights
    data = np.array(data)

    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    Z = data
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Spectral')])
    x_ticks = {'tickvals': [0,len(data[0])], 'ticktext': ['0', '1']}
    fig.update_layout(
        scene=dict(
            xaxis_title='Distance [1]',
            yaxis_title='Timesteps [s]',
            zaxis_title='Height over 6370 [km]',
            xaxis=x_ticks
        ),title = f'Heights of Line of Sight for Period No. {period}'
    )
    fig.show()
    
def plot_communication_detailed(mesh,step,time_step,distance,sim_start,f):
    """
    This function plotes the height of the los of a specific time step. 
    :param mesh: Matrix representing distance over a specific period.
    :type mesh: numpy.ndarray
    :param step: The time step to examine.
    :type step: int
    :param time_step: The time step duration.
    :type time_step: int
    :param distance: List of distances over a specific period.
    :type distance: list[float]
    :param sim_start: The datetime of the simulation start point.
    :type sim_start: datetime.datetime
    :param f: The frequency of communication.
    :type f: float
    """
    d = round(distance[0][step])
    height = mesh[step]
    x = range(len(height))
    min_y = np.min(height)
    min_x = x[np.argmin(height)]
    dt = timedelta(seconds=time_step*step)
    t = dt + sim_start
    plt.plot(x,height)
    fspl = calculate_fspl(distance[0][step],f)
    
    plt.title(f'Height of LOS for {t}')
    plt.xlabel("Distance [km]")
    plt.ylabel("Height over 6370 [km]")
    plt.xticks([0, len(height)], ['0', d])
    plt.grid(True)
    plt.plot(min_x, min_y, 'r.', markersize=10, )
    plt.annotate(f'{min_y:.0f}', xy=(min_x, min_y), xytext=(min_x-5, min_y+20))
    #plt.savefig(f"LOS_height.pdf", format='pdf')
    plt.show
    print(f'Free space loss for this link is {fspl:.2f} dB.')

def calculate_fspl(distance, f):
    """
    This function calculates the free space loss for a frequency and given distance.

    :param d: distance [m]
    :type d: float
    :param f: frequency [Hz]
    :type f: float
    :return fspl: free space loss [dB]
    :rtype: float
    """
    c = 299792458 #m/s
    lambd = c/f
    fspl = 20*np.log10(((4*np.pi*distance)/lambd))
    return fspl

def fetch_and_save_tle(url, filename):
    """
    This function downloads a txt file from an url and saves it with a custum name. 
    At https://celestrak.org one could find lists of TLEs. 

    :param url: url 
    :param filename: name of how the file should be saved 
    """
    response = requests.get(url)
    response.raise_for_status()

    with open(filename, 'w') as file:
        file.write(response.text)

    print(f"TLE-data is fetched and saved in {filename}.")

def ocean_ground_tracker(sat_list:list[Satellite]):
    """
    This function determines whether a satellite is located over the ocean. The information is stored 
    in the .ocean attribute with either True or False. GEO satellites just been calculated for the 
    first timestep, because of API restrictions as duplicate location requests are blocked.

    :param sat_list: list of satellites objects
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.ocean = [] 
    geolocator = Nominatim(user_agent="satellites")
    for sat in range(num_satellites): 
        dis = np.sqrt(sat_list[sat].x[0]**2+sat_list[sat].y[0]**2+sat_list[sat].z[0]**2)
        if dis > 40000:                                                                 # make sure GEO sat only one request, doubble requests gets blocked from API 
            coordinates = f"{sat_list[sat].lat[0]},{sat_list[sat].lon[0]}"
            location = geolocator.reverse(coordinates)
            if location == None:
                for i in range(len(sat_list[0].lat)):
                    sat_list[sat].ocean.append(True)
            else:
                for i in range(len(sat_list[0].lat)):
                    sat_list[sat].ocean.append(False)
        else: 
            for step in range(len(sat_list[0].lat)):
                time.sleep(1)                                                        # dont violate the API restiction rules 
                coordinates = f"{sat_list[sat].lat[step]},{sat_list[sat].lon[step]}"
                location = geolocator.reverse(coordinates)
                if location == None:
                    sat_list[sat].ocean.append(True)
                else:
                    sat_list[sat].ocean.append(False)    

def are_inside_area(coordinates, threat_areas):
    """
    Checks if a list of earth coordinates (longitude, latitude) are inside defined threat areas.

    :param coordinates: List of (longitude, latitude) tuples
    :param threat_areas: Geopandas GeoDataFrame of threat areas
    :return: List of booleans indicating whether each coordinate is inside a threat area
    """
    points = gpd.GeoSeries([Point(lon, lat) for lon, lat in coordinates])
    return points.apply(lambda point: threat_areas.contains(point).any())

def is_inside_area(coordinate, threat_areas):
    """
    Checks if an earth coordinate (longitude, latitude) is inside defined threat areas.

    :param coordinate: A tuple of (longitude, latitude)
    :param threat_areas: Geopandas GeoDataFrame of threat areas
    :return: Boolean indicating whether the coordinate is inside a threat area
    """
    if coordinate == None:
        return True
    lon, lat = coordinate
    point = Point(lon, lat)
    return threat_areas.contains(point).any()

def threatarea_checker(sat_list:list[Satellite],threatarea_file):
    """
    This function determines for every position of every satellie 
    if the groundtrack is inside a threatarea. 

    :param sat_list: list of satellites objects
    :param threatarea_file: name of .shp file
    """
    for sat in sat_list:
        sat.area = []

    threat_areas = gpd.read_file(threatarea_file)
    
    for sat in sat_list:
        sat.area = are_inside_area(list(zip(sat.lon,sat.lat)),threat_areas)

def calculate_total_difference(period_data):
    """
    This function calculates the total difference of overlapping intervalls, made to process
    calculate_overlaps_for_each_interval return list
    """
    period_num, ranges = period_data
    total_diff = sum(end - start for start, end in ranges)
    return total_diff

def find_matching_true_periods(list_a:list, list_b:list, intervals:list[tuple]):
    """
    This function determines if both lists a and b have the same indexes for 
    True values for all periods given in the list intervals.

    :param list_a: list A
    :type list_a: list
    :param list_b: list B
    :type list_b: list 
    :param intervals: los periods 
    :type intervals: list
    :return matching_periods: list of tuples of matching periods with their index in the intervals list and amount of time steps
    """
    matching_periods = []

    for index, (start, end) in enumerate(intervals):
        counter = 0
        for i in range(start, end):
            if list_a[i] == True and list_b[i] == True:
                counter += 1

        if counter > 0:
            matching_periods.append((index, counter))

    return matching_periods

def calculate_overlaps_for_each_interval(intervals_a, intervals_b):
    """
    This function calculates the if the b lists of tuple have an overalps with the a list tuples. 

    :param intervals_a: list of tuple with (start,end) of interval 
    :param intervals_b: list of tuple with (start,end) of interval 
    :return all_overlaps: list of [(periodnumber, (overlapstart,overlapend),...),...]
    """
    all_overlaps = []

    for index, (start_a, end_a) in enumerate(intervals_a):
        interval_overlaps = []
        for start_b, end_b in intervals_b:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                interval_overlaps.append((overlap_start, overlap_end))
        all_overlaps.append((index, interval_overlaps))

    return all_overlaps

def threshold_checker_threatarea(sat_list:list[Satellite],threshold:list[tuple],sat_number:int,time_step:int,orientation:str,theta_3dB,target_only=False):
    """
    This function checks if a contact between satellites is possible based on the given threshold areas.

    :param sat_list: List of satellite objects.
    :type sat_list: List[Satellite]
    :param threshold: List of contact periods for special antenna orientation cases.
    :type threshold: List[tuple]
    :param sat_number: Number of the investigating satellite.
    :type sat_number: int
    :param time_step: Time step.
    :type time_step: int
    :param theta_3dB: Full Width at Half Maximum (FWHM) of the antenna.
    :type theta_3dB: float
    :param orientation: Case of orientation (A, B, C, D, E, F, G) possible.
    :type orientation: str
    :param target_only: True if both satellites have to be inside the threshold area, False if only the target.
    :type target_only: bool

    """
    num_satellites = len(sat_list)

    if orientation == 'A':
        orien = 'omnidirectional'
        print('--------------------------------------------------------------------')
        print(f'Contact according to attack probability of {sat_list[sat_number].name}')
        print(f'Assumptions:')
        print(f'- orientation of antenna {orien} ')
        print(f'- special threatarea ')
        print('--------------------------------------------------------------------')
        
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                list_of_intervals = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].los[sat]))
                if len(list_of_intervals) == 0:
                    print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                    print('  No line of sight')
                    continue
                else: 
                    if target_only == False:
                        matching = find_matching_true_periods(sat_list[sat_number].area,sat_list[sat].area,list_of_intervals)

                        matching_sorted = sorted(matching, key=lambda x: x[1], reverse=True)
                        
                        if matching_sorted == []:
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            print('  No threat thru area')
                        else:  
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            for index, (period_num,count) in enumerate(matching_sorted): 
                                
                                print(f'    {index+1}st. Rank: period no. {period_num+1} [lenght:{count*time_step}sek]')
                    else: 
                        matching = calculate_overlaps_for_each_interval(list_of_intervals, find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].area)))

                        if matching == []:
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            print('  No threat thru area')
                        else:  
                            matching_sorted = sorted(matching, key=calculate_total_difference, reverse=True)
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            for index, (period_num, ranges) in enumerate(matching_sorted):
                                for start, end in ranges:
                                    print(f'    {index + 1}st. Rank: period no. {period_num+1} [lenght:{(end-start+1)*time_step}sek]')

    else: 
        if orientation == 'B':
            orien = 'in flight direction'
        elif orientation == 'C':
            orien = 'in opposite flight direction'
        elif orientation == 'D':
            orien = 'as nadir pointing'
        elif orientation == 'E':
            orien = 'as starpointing'
        elif orientation == 'F':
            orien = 'to left hand side in flight direction'
        elif orientation == 'G':
            orien = 'to right hand side in flight direction'
        print('--------------------------------------------------------------------')
        print(f'Contact according to attack probability of {sat_list[sat_number].name}')
        print(f'Assumptions:')
        print(f'- orientation of antenna {orien} ')
        print(f'- theta 3dB of {theta_3dB*2}°')
        print(f'- special threatarea ')
        print('--------------------------------------------------------------------')
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                list_of_intervals = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].los[sat]))
                if target_only == False:
                    if list_of_intervals == []:
                        print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                        print('  No line of sight')
                        continue
                    else:
                        area_1 = sat_list[sat_number].area
                        area_2 = sat_list[sat].area
                        matching = []
                        for index,(start, end) in enumerate(list_of_intervals):
                            counter = 0
                            for indey, step in enumerate(range(start, end + 1)): 
                                if area_1[step] == True and area_2[step] == True and threshold[sat][index][indey] == True: 
                                    counter += 1
                            matching.append((index,counter))
                        matching_sorted = sorted(matching, key=lambda x: x[1], reverse=True)
                        final_list = []
                        for item in matching_sorted:
                            if item[1] > 0: 
                                final_list.append(item)

                        if final_list == []:
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            print('    No threat in specific area')
                        else:  
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            for index, (period_num,count) in enumerate(final_list): 
                                liste = []
                                for access in sat_list[0].multipleaccess[1]:
                                    a,b,c,d = access
                                    if a == period_num + 1:
                                        liste.append((b,c,d))
                                print(f'    {index+1}st. Rank: period no. {period_num+1} lenght:[{count*time_step}sek] Also connection:{liste}')

                else:
                    if list_of_intervals == []:
                        print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                        print('    No line of sight')
                        continue
                    else:
                        area_1 = sat_list[sat_number].area
                        matching = []
                        for index,(start, end) in enumerate(list_of_intervals):
                            counter = 0
                            for indey, step in enumerate(range(start, end + 1)): 
                                if area_1[step] == True and threshold[sat][index][indey] == True: 
                                    counter += 1
                            matching.append((index,counter))
                        matching_sorted = sorted(matching, key=lambda x: x[1], reverse=True)
                        final_list = []
                        for item in matching_sorted:
                            if item[1] > 0: 
                                final_list.append(item)

                        if final_list == []:
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            print('    No threat in specific area')
                        else:  
                            print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                            for index, (period_num,count) in enumerate(final_list): 
                                liste = []
                                for access in sat_list[sat_number].multipleaccess[sat]:
                                    a,b,c,d = access
                                    if a == period_num + 1:
                                        liste.append((b,c,d))
                                print(f'    {index+1}st. Rank: period no. {period_num+1} lenght:[{count*time_step}sek] Also connection:{liste}')

def ultimate_threshold_future(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: satellite is inside the antennacone
    list C: is satellite antennacone inside the threatarea 

    :param sat_list: list of satellites with attribute .los .theta .antenna_areas
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat.theta[j]
                c = sat_list[j].antenna_areas[i]
                if len(a) != len(b) != len(c):
                    raise ValueError(f"Problem in list lenght of los theta or area.")

                comparison = [(x, y, z) for x, y, z in zip(a, b, c)]
                sat.threshold.append(comparison)

def ultimate_threshold_future_maxdistance(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: satellite is inside the antennacone
    list C: is satellite antennacone inside the threatarea 
    list D: max distance

    :param sat_list: list of satellites with attribute .los .theta .antenna_areas .max_distance  
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat.theta[j]
                c = sat_list[j].antenna_areas[i]
                d = sat.max_distance[j]
                if len(a) != len(b) != len(c) != len(d):
                    raise ValueError(f"Problem in list lenght of los theta or area.")

                comparison = [(w, x, y, z) for w, x, y, z in zip(a, b, c, d)]
                sat.threshold.append(comparison)

def ultimate_threshold_past(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: satellite is inside the antennacone

    :param sat_list: list of satellites with attribute .los .theta .max_distance 
    :type sat_list: list[Satellite] 
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat.theta[j]
                if len(a) != len(b):
                    raise ValueError(f"Problem in list lenght of los theta or area.")

                comparison = [(x, y) for x, y in zip(a, b)]
                sat.threshold.append(comparison)

def ultimate_threshold_past_maxdistance(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: satellite is inside the antennacone
    list C: max distance

    :param sat_list: list of satellites with attribute .los .theta .max_distance 
    :type sat_list: list[Satellite] 
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat.theta[j]
                c = sat.max_distance[j]
                if len(a) != len(b):
                    raise ValueError(f"Problem in list lenght of .los .theta or .max_distance")

                comparison = [(x, y, z) for x, y, z in zip(a, b, c)]
                sat.threshold.append(comparison)

def ultimate_threshold_past_spherical(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight

    :param sat_list: list of satellites with attribute .los
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                sat.threshold.append(a)

def ultimate_threshold_past_spherical_maxdistance(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: max distance

    :param sat_list: list of satellites with attribute .los .max_distance
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat.max_distance[j]
                if len(a) != len(b):
                    raise ValueError(f"Problem in list lenght of .los or .max_distance.")

                comparison = [(x, y) for x, y in zip(a, b)]
                sat.threshold.append(comparison)

def ultimate_threshold_future_spherical(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: is satellite antenna cone inside the threatarea 

    :param sat_list: list of satellites with attribute .los .antenna_areas
    :type sat_list: list[Satellite]
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat_list[j].antenna_areas[i]
                if len(a) != len(b):
                    raise ValueError(f"Problem in list lenght of .los or .antenna_areas.")

                comparison = [(x, y) for x, y in zip(a, b)]
                sat.threshold.append(comparison)

def ultimate_threshold_future_spherical_maxdistance(sat_list:list[Satellite]):
    """
    This function merges lists of thresholds together to detect specific 
    contact periods. 
    
    list A: line of sight
    list B: is satellite antenna cone inside the threatarea 
    list C: max distance

    :param sat_list: list of satellites with attribute .los .theta .area 
    :type sat_list: list[Satellite] 
    """
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.threshold = []

    for i, sat in enumerate(sat_list):
        for j in range(num_satellites):
            if i == j:
                sat.threshold.append([])
                continue
            else:
                a = sat.los[j]
                b = sat_list[j].antenna_areas[i]
                c = sat.max_distance[j]
                if len(a) != len(b):
                    raise ValueError(f"Problem in list lenght of los theta or area.")

                comparison = [(x, y, z) for x, y, z in zip(a, b, c)]
                sat.threshold.append(comparison)

def set_threshold_future(sat_list:list[Satellite],max_dis,max_d_toggle):
    """
    This functions enables and disables the maximum distance calculation and treshold 
    for contact periods of directional antennas for future attack calculation.
    
    :param sat_list: list of satellite objects 
    :param max_dis: maximum allowed distance of satellites for contact in [km]
    :type max_dis: int
    :param max_d_toggle: enable or disable maximum distance for contacts
    """
    if max_d_toggle == True: 
        max_distance_checker(sat_list,max_dis)
        ultimate_threshold_future_maxdistance(sat_list)
        contacts_for_thresholds_abcd(sat_list)
    if max_d_toggle == False: 
        ultimate_threshold_future(sat_list)
        contacts_for_thresholds_abc(sat_list)

def set_threshold_future_spherical(sat_list:list[Satellite],max_dis,max_d_toggle):
    """
    This functions enables and disables the maximum distance calculation and treshold 
    for contact periods of spherical antennas for future attack calculation. 
    
    :param sat_list: list of satellite objects 
    :param max_dis: maximum allowed distance of satellites for contact in [km]
    :type max_dis: int
    :param max_d_toggle: enable or disable maximum distance for contacts
    """
    if max_d_toggle == True: 
        max_distance_checker(sat_list,max_dis)
        ultimate_threshold_future_spherical_maxdistance(sat_list)
        contacts_for_thresholds_abc(sat_list)
    if max_d_toggle == False: 
        ultimate_threshold_future_spherical(sat_list)
        contacts_for_thresholds_ab(sat_list)

def set_threshold_past(sat_list:list[Satellite],max_dis,max_d_toggle):
    """
    This functions enables and disables the maximum distance calculation and treshold 
    for contact periods of directional antennas for past attack calculation.
    
    :param sat_list: list of satellite objects 
    :param max_dis: maximum allowed distance of satellites for contact in [km]
    :type max_dis: int
    :param max_d_toggle: enable or disable maximum distance for contacts
    """
    if max_d_toggle == True: 
        max_distance_checker(sat_list,max_dis)
        ultimate_threshold_past_maxdistance(sat_list)
        contacts_for_thresholds_abc(sat_list)
    if max_d_toggle == False: 
        ultimate_threshold_past(sat_list)
        contacts_for_thresholds_ab(sat_list)

def set_threshold_past_spherical(sat_list:list[Satellite],max_dis,max_d_toggle):
    """
    This function enables or disables the maximum distance calculation and threshold 
    for contact periods of spherical antennas for past attack calculation. 
    
    :param sat_list: A list of Satellite objects.
    :type sat_list: list[Satellite]
    :param max_dis: The maximum allowed distance of satellites for contact in kilometers.
    :type max_dis: int
    :param max_d_toggle: A boolean indicating whether to enable or disable the maximum distance for contacts.
    :type max_d_toggle: bool
    """
    if max_d_toggle == True: 
        max_distance_checker(sat_list,max_dis)
        ultimate_threshold_past_spherical_maxdistance(sat_list)
        contacts_for_thresholds_ab(sat_list)
    if max_d_toggle == False: 
        ultimate_threshold_past_spherical(sat_list)
        contacts_for_thresholds_a(sat_list)

def contacts_for_thresholds_abcd(sat_list:list[Satellite]):
    """
    This function defines if a contact is available with all four thresholds in .thresholds

    :param sat_list: list of satellites with attribute threshold 
    :type sat_list: list[Satellite]
    """
    def contact(x):
        a,b,c,d = x
        if a == b == c == d == True:
            e = True
        else:
            e = False 
        return e 
    
    for i in range(len(sat_list)):
        sat_list[i].contacts = []
        for j in range(len(sat_list)):
            if i == j:
                sat_list[i].contacts.append([])
                continue 
            else: 
                info_list = sat_list[i].threshold[j]
                los_iterator = map(contact,info_list)
                contact_list = list(los_iterator)
                info_list = find_continuous_sequences_01(get_position_of_true(contact_list))
                sat_list[i].contacts.append(info_list)
    return

def contacts_for_thresholds_abc(sat_list:list[Satellite]):
    """
    This function defines if a contact is available with all three thresholds in .thresholds

    :param sat_list: list of satellites with attribute threshold 
    :type sat_list: list[Satellite]
    """
    def contact(x):
        a,b,c = x
        if a == b == c == True:
            d = True
        else:
            d = False 
        return d 
    
    for i in range(len(sat_list)):
        sat_list[i].contacts = []
        for j in range(len(sat_list)):
            if i == j:
                sat_list[i].contacts.append([])
                continue 
            else: 
                info_list = sat_list[i].threshold[j]

                los_iterator = map(contact,info_list)
                contact_list = list(los_iterator)
                info_list = find_continuous_sequences_01(get_position_of_true(contact_list))
                sat_list[i].contacts.append(info_list)
    return

def contacts_for_thresholds_ab(sat_list:list[Satellite]):
    """
    This function defines if a contact is available with all two thresholds in .thresholds

    :param sat_list: list of satellites with attribute threshold 
    :type sat_list: list[Satellite]
    """
    def contact(x):
        a,b = x
        if a == b == True:
            d = True
        else:
            d = False 
        return d 
    
    for i in range(len(sat_list)):
        sat_list[i].contacts = []
        for j in range(len(sat_list)):
            if i == j:
                sat_list[i].contacts.append([])
                continue 
            else: 
                info_list = sat_list[i].threshold[j]
                los_iterator = map(contact,info_list)
                contact_list = list(los_iterator)
                info_list = find_continuous_sequences_01(get_position_of_true(contact_list))
                sat_list[i].contacts.append(info_list)
    return

def contacts_for_thresholds_a(sat_list:list[Satellite]):
    """
    This function defines if a contact is available with one thresholds in .thresholds

    :param sat_list: list of satellites with attribute threshold 
    :type sat_list: list[Satellite]
    """
    
    for i in range(len(sat_list)):
        sat_list[i].contacts = []
        for j in range(len(sat_list)):
            if i == j:
                sat_list[i].contacts.append([])
                continue 
            else: 
                contact_list = sat_list[i].threshold[j]
                info_list = find_continuous_sequences_01(get_position_of_true(contact_list))
                sat_list[i].contacts.append(info_list)


def overlap_interval(interval_1:tuple, interval_2:tuple):
    """
    Checks if two intervals overlap and returns the overlapping interval.

    :param interval_1: Tuple of the first interval (start, end)
    :type interval_1: tuple 
    :param interval_2: Tuple of the second interval (start, end)
    :type interval_2: tuple 
    :return: Tuple of the overlapping interval (start, end) or None if no overlap
    """
    start1, end1 = interval_1
    start2, end2 = interval_2

    if start1 <= end2 and start2 <= end1:
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return (overlap_start, overlap_end)
    else:
        return None

def find_multiple_access_contacts(sat_list:list[Satellite]):
    """
    This function detects if a satellite has multiple contact posibilities while a contact period with 
    a special Satellite. The information is stored inside .multipleacces with (number of contactperiod,
    number of satellite, startpoint, endpoint)

    :param sat_list: list of satellite objects
    """
    for i,sat in enumerate(sat_list):
        sat.multiaccess_l_1 = [[] for _ in sat_list] 
        for j,periods in enumerate(sat.contacts):
            if i != j:
                for index, contact1 in enumerate(periods):
                        for l in range(len(sat_list)):
                            if l != i and l != j:
                                for contact2 in sat_list[i].contacts[l]:
                                    overlap = overlap_interval(contact1, contact2)
                                    if overlap:
                                        a, b = overlap
                                        sat_list[i].multiaccess_l_1[j].append((index, l, a, b))
    return

def find_multiple_access_periods(sat_list:list[Satellite]):
    """
    This function detects if a satellite has multiple contact posibilities while a contact period with 
    a special Satellite. The information is stored inside .multipleacces with (number of contactperiod,
    number of satellite, startpoint, endpoint)

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite] 
    """
    for i,sat in enumerate(sat_list):
        sat.multiaccess_l_2 = [[] for _ in sat_list] 
        for j,periods in enumerate(sat.periods):
            if i != j:
                for index, contact1 in enumerate(periods):
                        for l in range(len(sat_list)):
                            if l != i and l != j:
                                for contact2 in sat_list[i].periods[l]:
                                    overlap = overlap_interval(contact1, contact2)
                                    if overlap:
                                        a, b = overlap
                                        sat_list[i].multiaccess_l_2[j].append((index, l, a, b))
    return

def find_multiple_access_mix(sat_list:list[Satellite]):
    """
    This function detects if a satellite has multiple contact posibilities while a contact period with 
    a special Satellite. The information is stored inside .multipleacces with (number of contactperiod,
    number of satellite, startpoint, endpoint)

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite] 
    """
    for i,sat in enumerate(sat_list):
        sat.multiaccess_l_3 = [[] for _ in sat_list] 
        for j,periods in enumerate(sat.contacts):
            if i != j:
                for index, contact1 in enumerate(periods):
                        for l in range(len(sat_list)):
                            if l != i and l != j:
                                for contact2 in sat_list[i].periods[l]:
                                    overlap = overlap_interval(contact1, contact2)
                                    if overlap:
                                        a, b = overlap
                                        sat_list[i].multiaccess_l_3[j].append((index, l, a, b))
    return

def create_cone_with_vector(X, theta_degree, angle_res):
    """
    This function creates a cone with the vector X as the main axis and adds additional 
    vectors that form two smaller cones within the cone.  
    
    :param X: The main axis vector of the cone.
    :type X: numpy.ndarray
    :param theta_degree: The angle of the cone in degrees.
    :type theta_degree: float
    :param angle_res: The angle resolution for adding additional vectors.
    :type angle_res: float
    :return: A list of vectors representing the cone and the inner cones.
    :rtype: list[numpy.ndarray]
    """
    theta = np.deg2rad(theta_degree)
    X = X / np.linalg.norm(X)

    if np.allclose(X, np.array([1, 0, 0])):
        basis1 = np.array([0, 1, 0])
        basis2 = np.array([0, 0, 1])
    else:
        basis1 = np.cross(X, np.array([1, 0, 0]))
        basis2 = np.cross(X, basis1)

    vectors = []
    # antennacone 
    for i in range(0, 360, angle_res):
        phi = np.deg2rad(i)
        rotated_basis1 = np.cos(phi) * basis1 + np.sin(phi) * basis2
        cone_vector = np.cos(theta) * X + np.sin(theta) * rotated_basis1
        inner_cone_vector_1 =  np.cos(theta*1/3) * X + np.sin(theta*1/3) * rotated_basis1
        inner_cone_vector_2 =  np.cos(theta*2/3) * X + np.sin(theta*2/3) * rotated_basis1
        vectors.append(cone_vector)
        vectors.append(inner_cone_vector_1)
        vectors.append(inner_cone_vector_2)
    vectors.append(X)
    return vectors

def create_cone_with_vector_for_huge_theta(X, theta_degree, angle_res):
    """
    This function creates a cone with the vector X as the main axis and adds additional 
    vectors that form four smaller cones within the cone.

    
    :param X: Hauptachse Vektor
    :type X: np.ndarray
    :param theta_degree: theta in [Grad]
    :type theta_degree: float
    :param angle_res: angle resolution
    :type angle_res: float or int
    :return: list of vectors forming the cone
    :rtype: list[np.ndarray]
    """
    theta = np.deg2rad(theta_degree)
    X = X / np.linalg.norm(X)

    # Basisvektoren bestimmen
    if np.allclose(X, np.array([1, 0, 0])):
        basis1 = np.array([0, 1, 0])
        basis2 = np.array([0, 0, 1])
    else:
        basis1 = np.cross(X, np.array([1, 0, 0]))
        basis2 = np.cross(X, basis1)

    vectors = []
    # Erstellen der Kegeloberfläche
    for i in range(0, 360, angle_res):
        phi = np.deg2rad(i)
        rotated_basis1 = np.cos(phi) * basis1 + np.sin(phi) * basis2
        cone_vector = np.cos(theta) * X + np.sin(theta) * rotated_basis1
        inner_cone_vector_1 =  np.cos(theta*1/9) * X + np.sin(theta*1/9) * rotated_basis1
        inner_cone_vector_2 =  np.cos(theta*1/7) * X + np.sin(theta*1/7) * rotated_basis1
        inner_cone_vector_3 =  np.cos(theta*1/4) * X + np.sin(theta*1/4) * rotated_basis1
        inner_cone_vector_4 =  np.cos(theta*1/3) * X + np.sin(theta*1/3) * rotated_basis1
        inner_cone_vector_5 =  np.cos(theta*1/2) * X + np.sin(theta*1/2) * rotated_basis1
        inner_cone_vector_6 =  np.cos(theta*2/3) * X + np.sin(theta*2/3) * rotated_basis1
        inner_cone_vector_7 =  np.cos(theta*3/4) * X + np.sin(theta*3/4) * rotated_basis1
        inner_cone_vector_8 =  np.cos(theta*6/7) * X + np.sin(theta*6/7) * rotated_basis1
        inner_cone_vector_9 =  np.cos(theta*8/9) * X + np.sin(theta*8/9) * rotated_basis1
        vectors.append(cone_vector)
        vectors.append(inner_cone_vector_1)
        vectors.append(inner_cone_vector_2)
        vectors.append(inner_cone_vector_3)
        vectors.append(inner_cone_vector_4)
        vectors.append(inner_cone_vector_5)
        vectors.append(inner_cone_vector_6)
        vectors.append(inner_cone_vector_7)
        vectors.append(inner_cone_vector_8)
        vectors.append(inner_cone_vector_9)
    vectors.append(X)
    return vectors

def find_intersection_with_wgs84(vector_start,vector_direction,transformer=False):
    """
    This function calculates the intersection betweenn a vector and the wgs84 earth ellipsoid
    and determines the latitude and longitude of the intersection. In case of use CRS
    EPSG:4326 than transformer=False, if CRS ESRI:53042 used than transformer=True.

    :param vector_start: The starting vector of the vector
    :type vector_start: list
    :param vector_direction: The direction vector of the vector
    :type vector_direction: list
    :param transformer: A toggle whether a coordinate transformation is needed, defaults to False
    :type transformer: bool
    :return: A tuple containing the latitude and longitude of the intersection point
    :rtype: tuple(float, float)
    """
    a = 6378.137 
    b = 6356.752 
    vector_direction = vector_direction / np.linalg.norm(vector_direction)
    if np.dot(vector_start, vector_direction) >= 0: 
        return None # check if vector points towards earth
    A = (vector_direction[0] ** 2) / a ** 2 + \
        (vector_direction[1] ** 2) / a ** 2 + \
        (vector_direction[2] ** 2) / b ** 2
    B = 2 * (vector_start[0] * vector_direction[0]) / a ** 2 + \
        2 * (vector_start[1] * vector_direction[1]) / a ** 2 + \
        2 * (vector_start[2] * vector_direction[2]) / b ** 2
    C = (vector_start[0] ** 2) / a ** 2 + \
        (vector_start[1] ** 2) / a ** 2 + \
        (vector_start[2] ** 2) / b ** 2 - 1
    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return None  # no intersection
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2)
    intersection_point = vector_start + t * vector_direction
    lat = np.degrees(np.arctan2(intersection_point[2], np.sqrt(intersection_point[0] ** 2 + intersection_point[1] ** 2)))
    lon = np.degrees(np.arctan2(intersection_point[1], intersection_point[0]))
    if transformer == True: 
        proj_esri_53042 = Proj('esri:53042')
        proj_epsg_4326 = Proj('epsg:4326')
        transform = Transformer.from_proj(proj_epsg_4326,proj_esri_53042)
        x_esri, y_esri  = transform.transform(lat, lon)
        return x_esri, y_esri 
    return lat, lon

def create_antenna_cone(sat_list:list[Satellite],huge_theta=False):
    """
    This function creates a vektor cone for the attacking satellites antenna for every timestep in
    direction of every satellite.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite] 
    """
    for i,sat in enumerate(sat_list):
        sat.cone_vectors = [[] for _ in sat_list]
        for j in range(len(sat_list)):
            if i != j:
                for k in range(len(sat_list[0].divec)):
                    if huge_theta == True: 
                        cone_vectors = create_cone_with_vector_for_huge_theta(sat.divec[k][j],sat.theta_attack,2)
                    else:
                        cone_vectors = create_cone_with_vector(sat.divec[k][j],sat.theta_attack,2)
                    sat.cone_vectors[j].append(cone_vectors)

def create_lat_lon_from_intersection(sat_list:list[Satellite],start_time:datetime,time_step,transformer=False):
    """
    This function determies the latitude and longitude of vectors intersection with earth. 
    In case of use CRS EPSG:4326 than transformer=False, if CRS ESRI:53042 used than transformer=True.

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param start_time: datetime object sim start 
    :type start_time: datetime
    :param time_step: time step of calculation
    :type time_step: int
    :param transformer: toggle for crs
    :type transformer: boolaen
    """
    for i,sat in enumerate(sat_list):
        sat.intersections = [[] for _ in sat_list]
        for j in range(len(sat_list)):
            if i != j:
                for k in range(len(sat_list[0].divec)):
                    time = start_time + timedelta(seconds=time_step*k)
                    gst = calculate_gmst_with_skyfield(time) 
                    position_vec = sat.x[k],sat.y[k],sat.z[k]
                    x,y,z = position_vec
                    position_vec_ecef = eci_to_ecef((x,y,z),gst)
                    intersections = []
                    for vector in sat.cone_vectors[j][k]:
                        dir_vec_ecef =  eci_to_ecef(vector,gst)
                        intersection = find_intersection_with_wgs84(position_vec_ecef,dir_vec_ecef,transformer)
                        intersections.append(intersection)
                    sat.intersections[j].append(intersections)

def antenna_cone_is_inside_threatarea(sat_list:list[Satellite],threatarea_file:str):
    """
    This function determines if the signal traveled from satellites antenna is reaching the 
    pre defined threatarea or not. 

    :param sat_list: list of satellite objects
    :param threatarea_file: name of .shp file 
    """
    threat_areas = gpd.read_file(threatarea_file)

    for sat in sat_list:
        sat.antenna_areas = [[] for _ in sat_list] 
    for i,sat in enumerate(sat_list): 
        for j in range(len(sat_list)):
            if i != j:
                for k in range(len(sat.x)):
                    is_inside = []
                    for p in sat.intersections[j][k]:
                        if p is not None:
                            if is_inside_area(p, threat_areas) == True:
                                is_inside.append(True)
                            else:
                                is_inside.append(False)
                        else: 
                            is_inside.append(True)
                    if all(is_inside):
                        sat.antenna_areas[j].append(True)
                    else:
                        sat.antenna_areas[j].append(False)

def calculate_rotation_matrix(gst):
    """
    This function calulates a rotation matrix from eci to ecef for a specific time.

    :param gst: GST in Stunden
    :type gst: float or int
    :return: 3x3 NumPy-Array
    :rtype: numpy.ndarray
    """
    rotation_angle = gst * 15.0 #15.041084443619143
    rotation_angle_rad = math.radians(rotation_angle)

    cos_angle = np.cos(rotation_angle_rad)
    sin_angle = np.sin(rotation_angle_rad)
    rotation_matrix = np.array([
        [cos_angle, sin_angle, 0],
        [-sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def eci_to_ecef(eci_coords:list, gst):
    """
    Converts ECI coordinates into ECEF coordinates.

    :param eci_coords: ECI-coordinates (x, y, z)
    :type eci_coords: list
    :param gst: Greenwich Sidereal Time in hours
    :type gst: float
    :return ecef_coords: ECEF-coordinates (x, y, z)
    :rtype: list
    """
    rotation_matrix = calculate_rotation_matrix(gst)
    ecef_coords = np.dot(rotation_matrix, eci_coords)
    return ecef_coords

def gcrs_to_ecef(gcrs_coords:list, time_utc):
    """
    Converts GCRS coordinates to ECEF coordinates.

    :param gcrs_coords: GCRS coordinates (x, y, z)
    :type gcrs_coords: list
    :param time_utc: Time in UTC
    :type time_utc: Timescale
    :return: ECEF coordinates (x, y, z)
    :rtype: list
    """
    # calcualtion of gmst
    gst = calculate_gmst_with_skyfield(time_utc)

    # change from GST in Degrees 
    rotation_angle = gst * 15.0  # 1 h = 15 deg
    rotation_angle_rad = math.radians(rotation_angle)

    # Determination of rotion matrix
    cos_angle = np.cos(rotation_angle_rad)
    sin_angle = np.sin(rotation_angle_rad)
    rotation_matrix = np.array([
        [cos_angle, sin_angle, 0],
        [-sin_angle, cos_angle, 0],
        [0, 0, 1]
        ])
    ecef_coords = np.dot(rotation_matrix, gcrs_coords)
    return ecef_coords

def calculate_gmst_with_skyfield(time_utc):
    """
    Calculates the Greenwich Mean Sidereal Time using Skyfield for a given UTC time.

    :param time_utc: Time in UTC as a datetime object.
    :type time_utc: Timescale
    :return: GMST in hours.
    :rtype: gmst object
    """
    ts = load.timescale()
    t = ts.utc(time_utc.year, time_utc.month, time_utc.day, time_utc.hour, time_utc.minute, time_utc.second)
    gmst = t.gmst 
    return gmst

def ecef_to_geodetic(x:float, y:float, z:float):
    """
    Converts ECEF coordinates to geodetic coordinates (latitude, longitude).

    :param x: X coordinate in ECEF (km)
    :type x: float
    :param y: Y coordinate in ECEF (km)
    :type y: float 
    :param z: Z coordinate in ECEF (km)
    :type z: float
    :return: latitude, longitude in degrees
    :rtype: tuple
    """
    # WGS84 ellipsoid constants
    a = 6378.137  # equatorial radius in km
    b = 6356.752  # polar radius in km
    e2 = 1 - (b**2 / a**2)

    # calculations
    r = np.sqrt(x**2 + y**2)
    E2 = a**2 - b**2
    F = 54 * b**2 * z**2
    G = r**2 + (1 - e2) * z**2 - e2 * E2
    c = (e2**2 * F * r**2) / (G**3)
    s = ( 1 + c + np.sqrt(c**2 + 2*c) )**(1/3)
    P = F / (3 * (s + 1/s + 1)**2 * G**2)
    Q = np.sqrt(1 + 2 * e2**2 * P)
    ro = -(P * e2 * r) / (1 + Q) + np.sqrt((a**2 / 2) * (1 + 1/Q) - (P * (1 - e2) * z**2) / (Q * (1 + Q)) - P * r**2 / 2)
    tmp = (r - e2 * ro)**2
    U = np.sqrt( tmp + z**2 )
    V = np.sqrt( tmp + (1 - e2) * z**2 )
    zo = (b**2 * z) / (a * V)

    # Calculate latitude and longitude
    latitude = np.arctan((z + e2 * zo) / r)
    longitude = np.arctan2(y, x)

    return np.degrees(latitude), np.degrees(longitude)

def plot_points_on_folium_map(points:list[tuple]):
    """
    Plot points on a map using Folium.

    :param points: List of (latitude, longitude) 
    :type points: list[tuple]
    """
    def reinige_liste(meine_liste):
        # Erstellt eine neue Liste, die nur Tuples aus der ursprünglichen Liste enthält
        gereinigte_liste = [element for element in meine_liste if element is not None]
        return gereinigte_liste
    points = reinige_liste(points)
    # Create a map centered around the average coordinates
    avg_lat = sum(p[0] for p in points) / len(points)
    avg_lon = sum(p[1] for p in points) / len(points)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    # Add markers for each point
    for lat, lon in points:
        folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)

    # Display the map
    return m

def plot_fspl(sat_list:list[Satellite],sat_one:int,sat_two:int,time_step:int,frequency:float,period:int):
    """
    This function plottes the free space loss for a specific communication.

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_one: number of first satellite
    :type sat_one: int
    :param sat_two: number of secound satellite 
    :type sat_two: int
    :param time_step: value of timestep
    :type time_step: int
    :param frequency: frequency 
    :type frequency: float
    :param period: number of specific period 
    :type period: int
    """
    list_of_intervals = sat_list[sat_one].contacts[sat_two]
    contact = list_of_intervals[period]
    start,end = contact
    fspl = []
    for i in range(start,end+1):
        d = sat_list[sat_one].distance[sat_two][i]
        d = d * 1000
        fspl.append(calculate_fspl(d,frequency))
    x = np.linspace(start,end,len(fspl))
    plt.plot(x,fspl)
    plt.title(f'FSPL for Communication Period No.{period}')
    plt.xlabel(f't in timestep [{time_step}s]')
    plt.ylabel("FSPL [dB]")
    plt.xticks([start, end], [start, end])
    plt.grid(True)
    #plt.savefig(f"fspl_period_{period}.pdf" , format='pdf')
    plt.show

def show_antenna_groundtrack(sat_list:list[Satellite],time_step_no,sat_no_1:int,sat_no_2:int):
    """
    This function shows the groundtrack of the satellite antenna for a specific time period 
    and a specific target pointed satellite.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_no_1: number of attacking satellite 
    :type sat_no_1: int
    :param sat_no_2: number of target satellite 
    :type sat_no_2: int
    :param time_step_no: number of specific timestep
    :type time_step_no: int
    """
    points = sat_list[sat_no_2].intersections[sat_no_1][time_step_no]
    points.append((sat_list[sat_no_1].lat[time_step_no],sat_list[sat_no_1].lon[time_step_no]))
    points.append((sat_list[sat_no_2].lat[time_step_no],sat_list[sat_no_2].lon[time_step_no]))
    m = plot_points_on_folium_map(points)
    return m

def threshold_checker(sat_list:list[Satellite],threshold,sat_number:int,time_step:int,orientation:str,theta_3dB:float):
    """
    This function checks if a contact between satellites is possible.

    :param sat_list: List of satellite objects.
    :type sat_list: list[Satellite]
    :param threshold: List of contract periods for special antenna orientation case.
    :type threshold: list
    :param sat_number: Number of the investigation satellite.
    :type sat_number: int
    :param time_step: Time step.
    :type time_step: int
    :param orientation: Case of orientation (A, B, C, D, E, F, G) possible.
    :type orientation: str
    :param theta_3dB: Full Width at Half Maximum (FWHM) of the antenna.
    :type theta_3dB: float
    """
    num_satellites = len(sat_list)

    if orientation == 'A':
        orien = 'omnidirectional'
        print('--------------------------------------------------------------------')
        print(f'This programm calculates every contact periods of {sat_list[sat_number].name}')
        print(f'with every satellite which is defined in the tle file.')
        print(f'Assumptions:')
        print(f'- orientation of antenna {orien} ')
        print(f'- theta 3dB of {theta_3dB*2}°')
        print('--------------------------------------------------------------------')
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                list_of_intervals = find_continuous_sequences_01(get_position_of_true(sat_list[sat_number].los[sat]))
                if len(list_of_intervals) == 0:
                    continue
                else: 
                    print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                    for period in range(len(list_of_intervals)): 
                        start,end = list_of_intervals[period]
                        dt = end - start
                        print(f'  Match in period No. {period+1} \t Duration of {dt*time_step} sek')
    else: 
        if orientation == 'B':
            orien = 'in flight direction'
        elif orientation == 'C':
            orien = 'in opposite flight direction'
        elif orientation == 'D':
            orien = 'as nadir pointing'
        elif orientation == 'E':
            orien = 'as starpointing'
        elif orientation == 'F':
            orien = 'to left hand side in flight direction'
        elif orientation == 'G':
            orien = 'to right hand side in flight direction'
        print('--------------------------------------------------------------------')
        print(f'This programm calculates every contact periods of {sat_list[sat_number].name}')
        print(f'with every satellite which is defined in the tle file.')
        print(f'Assumptions:')
        print(f'- orientation of antenna {orien} ')
        print(f'- theta 3dB of {theta_3dB*2}°')
        print('--------------------------------------------------------------------')
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                counter = 0
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name}')
                for period in range(len(threshold[sat])): 
                    checker = list_counter(threshold[sat][period])
                    if checker > 0:
                        print(f'  Match in period No. {period+1} \t Duration of {(checker)*time_step} sek')
                        counter += 1
                if counter == 0:
                    print('  No match')
                    
def find_overlapping_intervals(list1, list2):
    """ 
    This function takes two lists of tuple and returnes a list of 
    overlapping intervalls of tuple. 

    :param list1: list of tuple 
    :type list1: list[tuple]
    :param list2: list of tuple 
    :type list2: list[tuple]
    :return find_overlapping_intervalls: list of tuple 
    :rtype: list[tuple] 
    """
    find_overlapping_intervals = []
    for interval1 in list1:
        for interval2 in list2:
            start = max(interval1[0], interval2[0])
            end = min(interval1[1], interval2[1])
            if start < end:
                find_overlapping_intervals.append((start, end))
    return find_overlapping_intervals

def contacts_for_only_communication(sat_list:list[Satellite]):
    """
    This function takes a list of satellites and merges the intervals from attribute 
    of .theta and .los together and saves them into .communication.

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    """
    for sat in sat_list:
        sat.contacts = []
        for j in range(len(sat_list)):
                list_of_intervals_theta = find_continuous_sequences_01(get_position_of_true(sat.theta[j]))
                list_of_intervals_los = find_continuous_sequences_01(get_position_of_true(sat.los[j]))
                list_of_intervals = find_overlapping_intervals(list_of_intervals_theta,list_of_intervals_los)
                sat.contacts.append(list_of_intervals)


def threshold_filter_future_multi(sat_list:list[Satellite],sat_number:int,sim_start,sim_end,time_step:int,orientation:str):
    """
    This function prints a list of possible contact periods between satellite called by 
    index number and every other given satellite, taking into accound a special 
    threatarea. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite] 
    :param sat_number: index of investigation satellite
    :type sat_number: int 
    :param sim_start: start time of simulation
    :type sim_start: Timesclae object 
    :param time_step: time step
    :type time_step: int
    :param orientation: case of attitude A,B,C,D,E,F,G possible 
    :type sim_start: Timesclae object 
    """
    num_satellites = len(sat_list)
    if orientation == 'A':
        orien = 'omnidirectional'
        print('--------------------------------------------------------------------')
        print(f'Communication contact periods with {sat_list[sat_number].name}')
        print(f'Between: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,0).utc_strftime()} - {get_time(sim_end.year,sim_end.month,sim_end.day,sim_end.hour,sim_end.minute,sim_end.second,time_step,0).utc_strftime()}')
        print(f'Assumptions:')
        print(f'- attitude of attacked antenna omnidirectional ')
        print(f'- beamwidth of attacking satellite {sat_list[sat_number].theta_attack*2}°')
        print(f'- special threatarea')
        print('--------------------------------------------------------------------')
        sat_list[sat_number].threatlevel = [[] for _ in sat_list]
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                find_multiple_access_contacts(sat_list) # l_2
                list_of_intervals = sat_list[sat_number].contacts[sat]
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name} [Sat No.{sat_number}]')
                if len(list_of_intervals) == 0:
                    print('    No match')
                    continue
                else: 
                    for index,period in enumerate(list_of_intervals): 
                        start,end = list_of_intervals[index]
                        dt = end - start + 1
                        print(f'\n    Match in period No. {index} \t Duration of {dt*time_step} sek \t', end = ' ')       
                        print(f'from: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}\t [{start,end}]', end = ' ')
                        
                        print(f'Multiaccess with:', end = ' ')
                        
                        access_counter = 0
                        sat_counter = 0
                        lastperiod = None
                        lastsat = None
                        for i in range(len(sat_list[sat_number].multiaccess_l_1[sat])):
                        
                            if sat_list[sat_number].multiaccess_l_1[sat] == []:                            
                                level = 'M0'
                                sat_list[sat_number].threatlevel[sat].append((period,level))
                                continue
                            else: 
                                match = sat_list[sat_number].multiaccess_l_1[sat][i] 
                                a,b,c,d = match 
                                if a == index:
                                    print(f'Sat No. {b}{c,d}', end = ' ')
                                    if lastperiod == a:
                                        if lastsat == b:
                                            lastperiod = a
                                            lastsat = b
                                        else: 
                                            lastperiod = a
                                            access_counter += 1
                                            lastsat = b
                                            sat_counter += 1
                                    else:
                                        lastsat = b
                                        access_counter += 1
                                        lastperiod = a
                                        if sat_counter > 1: 
                                            level = f'M{sat_counter}'
                                            sat_list[sat_number].threatlevel[sat].append((period,level))
                                            print(f'[{level}]', end = '')
                                            continue

                        level = f'M{access_counter}'
                        sat_list[sat_number].threatlevel[sat].append((index,level))
                        print(f'[{level}]',end = '')
                            
    else: 
        if orientation == 'B':
            orien = 'in flight direction'
        elif orientation == 'C':
            orien = 'in opposite flight direction'
        elif orientation == 'D':
            orien = 'as nadir pointing'
        elif orientation == 'E':
            orien = 'as starpointing'
        elif orientation == 'F':
            orien = 'to left hand side in flight direction'
        elif orientation == 'G':
            orien = 'to right hand side in flight direction'
        print('--------------------------------------------------------------------')
        print(f'Communication contact periods with {sat_list[sat_number].name}')
        print(f'Between: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,0).utc_strftime()} - {get_time(sim_end.year,sim_end.month,sim_end.day,sim_end.hour,sim_end.minute,sim_end.second,time_step,0).utc_strftime()}')
        print(f'Assumptions:')
        print(f'- attitude of attacked antenna {orien}')
        print(f'- beamwidth antenna {sat_list[sat_number].theta_3dB}°')
        print(f'- beamwidth attacking antenna {sat_list[sat_number].theta_attack*2}°')
        print(f'- special threatarea ')
        print('--------------------------------------------------------------------')
        sat_list[sat_number].threatlevel = [[] for _ in sat_list]
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                find_multiple_access_contacts(sat_list)
                list_of_intervals = sat_list[sat_number].contacts[sat]
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name} [Sat No.{sat_number}]\n')
                if len(list_of_intervals) == 0:
                        print('    No match')
                        continue
                else:
                    for index, period in enumerate(list_of_intervals):
                        start, end = period
                        dt = end - start + 1

                        print(f'    Match in period No. {index} \t Duration of {dt*time_step} sek \t', end = ' ')       
                        print(f'from: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}\t [{start,end}]', end = ' ')
                        
                        print(f'Multiaccess with:', end = ' ')

                        access_counter = 0
                        sat_counter = 0
                        lastperiod = None
                        lastsat = None
                        for i in range(len(sat_list[sat_number].multiaccess_l_1[sat])):
                        
                            if sat_list[sat_number].multiaccess_l_1[sat] == []:                            
                                level = 'M0'
                                sat_list[sat_number].threatlevel[sat].append((period,level))
                                continue
                            else: 
                                match = sat_list[sat_number].multiaccess_l_1[sat][i] 
                                a,b,c,d = match 
                                if a == index:
                                    print(f'Sat No. {b}{c,d}', end = ' ')
                                    if lastperiod == a:
                                        if lastsat == b:
                                            lastperiod = a
                                            lastsat = b
                                        else: 
                                            lastperiod = a
                                            access_counter += 1
                                            lastsat = b
                                            sat_counter += 1
                                    else:
                                        lastsat = b
                                        access_counter += 1
                                        lastperiod = a
                                        if sat_counter > 1: 
                                            level = f'M{sat_counter}'
                                            sat_list[sat_number].threatlevel[sat].append((period,level))
                                            print(f'[{level}]', end = '')
                                            continue

                        level = f'M{access_counter}'
                        sat_list[sat_number].threatlevel[sat].append((index,level))
                        print(f'[{level}]',end = '')
                        print(f' ')   

def threshold_filter_past_multi(sat_list:list[Satellite],sat_number:int,sim_start,sim_end,time_step:int,orientation:str):
    """
    This function prints a list of possible contact periods between satellite with 
    number sat_number and every other given satellite. 

    :param sat_list: list of satellite objects 
    :type sat_list: list[Satellite]
    :param sat_number: index of investigation satellite 
    :type sat_number: int
    :param sim_start: start datetime of simulation
    :type sim_start: Timesclae object 
    :param sim_end: end datetime of simulation
    :type sim_end: Timesclae object 
    :param time_step: time step
    :type time_step: int
    :param orientation: case of attitude A,B,C,D,E,F,G possible 
    :type orientation: str
    """
    num_satellites = len(sat_list)
    if orientation == 'A':
        orien = 'omnidirectional'
        print('--------------------------------------------------------------------')
        print(f'Communication contact periods with  {sat_list[sat_number].name}')
        print(f'Between: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,0).utc_strftime()} - {get_time(sim_end.year,sim_end.month,sim_end.day,sim_end.hour,sim_end.minute,sim_end.second,time_step,0).utc_strftime()}')
        print(f'Assumptions:')
        print(f'- attitude of attacked antenna omnidirectional ')
        print('--------------------------------------------------------------------')
        sat_list[sat_number].threatlevel = [[] for _ in sat_list]
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                find_multiple_access_contacts(sat_list) # l_1
                list_of_intervals = sat_list[sat_number].contacts[sat]
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name} [Sat No.{sat_number}]')
                if len(list_of_intervals) == 0:
                    print('    No match')
                    continue
                else: 
                    for index,period in enumerate(list_of_intervals): 
                        start,end = list_of_intervals[index]
                        dt = end - start + 1
                        print(f'\n    Match in period No. {index} \t Duration of {dt*time_step} sek \t', end = ' ')       
                        print(f'from: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}\t [{start,end}]', end = ' ')
                        
                        print(f'Multiaccess with:', end = ' ')
                        
                        access_counter = 0
                        sat_counter = 0
                        lastperiod = None
                        lastsat = None
                        for i in range(len(sat_list[sat_number].multiaccess_l_1[sat])):
                        
                            if sat_list[sat_number].multiaccess_l_1[sat] == []:                            
                                level = 'M0'
                                sat_list[sat_number].threatlevel[sat].append((period,level))
                                continue
                            else: 
                                match = sat_list[sat_number].multiaccess_l_1[sat][i] 
                                a,b,c,d = match 
                                if a == index:
                                    print(f'Sat No. {b}{c,d}', end = ' ')
                                    if lastperiod == a:
                                        if lastsat == b:
                                            lastperiod = a
                                            lastsat = b
                                        else: 
                                            lastperiod = a
                                            access_counter += 1
                                            lastsat = b
                                            sat_counter += 1
                                    else:
                                        lastsat = b
                                        access_counter += 1
                                        lastperiod = a
                                        if sat_counter > 1: 
                                            level = f'M{sat_counter}'
                                            sat_list[sat_number].threatlevel[sat].append((period,level))
                                            print(f'[{level}]', end = '')
                                            continue

                        level = f'M{access_counter}'
                        sat_list[sat_number].threatlevel[sat].append((index,level))
                        print(f'[{level}]',end = '')
                            
    else: 
        if orientation == 'B':
            orien = 'in flight direction'
        elif orientation == 'C':
            orien = 'in opposite flight direction'
        elif orientation == 'D':
            orien = 'as nadir pointing'
        elif orientation == 'E':
            orien = 'as starpointing'
        elif orientation == 'F':
            orien = 'to left hand side in flight direction'
        elif orientation == 'G':
            orien = 'to right hand side in flight direction'
        print('--------------------------------------------------------------------')
        print(f'Communication contact periods with {sat_list[sat_number].name}')
        print(f'Between: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,0).utc_strftime()} - {get_time(sim_end.year,sim_end.month,sim_end.day,sim_end.hour,sim_end.minute,sim_end.second,time_step,0).utc_strftime()}')
        print(f'Assumptions:')
        print(f'- attitude of attacked antenna {orien}')
        print(f'- beamwidth antenna {sat_list[sat_number].theta_3dB}°')
        print('--------------------------------------------------------------------')
        sat_list[sat_number].threatlevel = [[] for _ in sat_list]
        for sat in range(num_satellites):
            if sat == sat_number:
                continue
            else:
                find_multiple_access_mix(sat_list)
                list_of_intervals = sat_list[sat_number].contacts[sat]
                print(f'\nCheck {sat_list[sat].name} [Sat No.{sat}] and {sat_list[sat_number].name} [Sat No.{sat_number}]\n')
                if len(list_of_intervals) == 0:
                        print('    No match')
                        continue
                else:
                    for index, period in enumerate(list_of_intervals):
                        start, end = period
                        dt = end - start + 1



                        print(f'    Match in period No. {index} \t Duration of {dt*time_step} sek \t', end = ' ')       
                        print(f'from: {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,start).utc_strftime()} - {get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,end).utc_strftime()}\t [{start,end}]', end = ' ')
                        
                        print(f'Multiaccess with:', end = ' ')

                        access_counter = 0
                        sat_counter = 0
                        lastperiod = None
                        lastsat = None
                        for i in range(len(sat_list[sat_number].multiaccess_l_3[sat])):
                        
                            if sat_list[sat_number].multiaccess_l_3[sat] == []:                            
                                level = 'M0'
                                sat_list[sat_number].threatlevel[sat].append((period,level))
                                continue
                            else: 
                                match = sat_list[sat_number].multiaccess_l_3[sat][i] 
                                a,b,c,d = match 
                                if a == index:
                                    print(f'Sat No. {b}{c,d}', end = ' ')
                                    if lastperiod == a:
                                        if lastsat == b:
                                            lastperiod = a
                                            lastsat = b
                                        else: 
                                            lastperiod = a
                                            access_counter += 1
                                            lastsat = b
                                            sat_counter += 1
                                    else:
                                        lastsat = b
                                        access_counter += 1
                                        lastperiod = a
                                        if sat_counter > 1: 
                                            level = f'M{sat_counter}'
                                            sat_list[sat_number].threatlevel[sat].append((period,level))
                                            print(f'[{level}]', end = '')
                                            continue

                        level = f'M{access_counter}'
                        sat_list[sat_number].threatlevel[sat].append((index,level))
                        print(f'[{level}]',end = '')
                        print(f' ')              

def sort_threatlevel(sat_list:list[Satellite],sat_number):
    """
    This function sortes the multiaccess treat level contact periods after highest multiaccess value. 

    :param sat_list: list of satellite objects
    :type sat_list: list[Satellite]
    :param sat_number: number of satellites to examinate 
    :type sat_number: int
    :return sorted_threatlevel: sorted threat level 
    :type sortet_threatlevel: list
    """
    def sort_of_mvalue(element):
        index, wert = element  # Element ist ein Tupel (Index, 'M*')
        return int(wert[1:])  # Extrahiert den numerischen Teil nach 'M' und konvertiert ihn in eine Integer
    sorted_threatlevel = []
    for sat in sat_list[sat_number].threatlevel:
        sorted_level = sorted(sat, key=sort_of_mvalue, reverse=True)
        sorted_threatlevel.append(sorted_level)

    return sorted_threatlevel
 

def show_threatlevel_multiaccess(sat_list:list[Satellite],sat_number):
    """
    This function shows the threat level of all contact periods on basis of multiaccess 
    of satellites with sat_number and sorts them by level.

    :param sat_list: list of satellites: 
    :type sat_list: list[Satellite]
    :sat_number: number of satellite to examinate 
    :type sat_number: int
    """
    sort = sort_threatlevel(sat_list,sat_number)
    print('----------------------------------------------------------------------------------')
    print(f'   Multiaccess while specific contact periods with {sat_list[sat_number].name}')
    print('----------------------------------------------------------------------------------\n')
    
    
    for index,sat in enumerate(sort):
        if index == sat_number:
            continue
        if sat == []:
            print(f'   Sat No.{index} {sat_list[index].name}')
            print('         -')
        else:
            print(f'   Sat No.{index} {sat_list[index].name}')
            print('        ', end = ' ')
            for level in sat:
                index,threat_level = level 
                print(f'Period No.{index}: [{threat_level}]', end = ' ')
            print(' ')

def plot_shapefile(threatarea_file,switch):
    """
    This function generates a plot of a shapefile. 

    :param threatarea_file: filename of shapefile
    :type threatarea_file: str
    :param switch: toggel to switch between coordinate systems 
    :type switch: boolean 
    """
    area = gpd.read_file(threatarea_file)
    fig, ax = plt.subplots()
    area.plot(ax=ax)
    if switch == True:
        ax.set_xlabel('X-axis [1]')
        ax.set_ylabel('Y-axis [1]')
    else:
        ax.set_xlabel('longitude [°]')
        ax.set_ylabel('latitude [°]')
    ax.set_title('Worldmap of Threat Areas')
    plt.show()

def is_boolean(value):
    """
    This function checks if a value is boolean or not. 

    :param value: value to check
    :type value: boolean
    :return: True if value is Boolean, False if not
    :rtype: boolean
    """
    if isinstance(value, bool):
        return True
    else:
        return False

def is_value_in_range(value,range):
    """
    Checks if a given value is within the range of 0 to 180.

    Issues a warning if the value is outside the range or if the data type is invalid.

    :param value: The value to be checked.
    :type value: int or float
    :raise TypeError: if value type wrong
    :return: Returns True if the value is within the range, otherwise False.
    :rtype: bool
    """
    a,b = range
    if not isinstance(value, (int, float)):
        warnings.warn("Value has to be integer or float.", RuntimeWarning)
        return False
    if a <= value <= b:
        return True
    else:
        return False

def is_orientation_right(orientation:str):
    """
    This function checks if orientation is correct declared. 

    :param orientation: orientation
    :type orientation: str
    """
    if orientation == 'A' or orientation == 'B' or orientation == 'C' or orientation == 'D'or orientation == 'E' or orientation == 'F' or orientation == 'G':
        return
    else: 
        warnings.warn("The value {orientation} is wrong! Only A,B,C,D,E,F,G", RuntimeWarning)
        return 

import warnings

def check_if_integer(input_value:bool):
    """
    This function checks if a value is a integer or not. 
    
    :param input_value: value to check
    :type imput_value: bool, int, float, str
    :return: True if value is integer, False if not
    :rtype: bool
    """
    if not isinstance(input_value, int):
        return False
    else:
        return True

def input_checker_future(theta_target,time_step,huge_theta,theta_attack,sat_number,atmosphere,transformer,orientation):
    """
    This function checks if the input values are in the correct range. 

    :param theta_target: beamwidth target antenna
    :raise TypeError: if theta_target outside of 0-180 
    :param time_step: time step of caclulation
    :raise TypeError: if time_step is no integer 
    :param huge_theta: True if attack_theta is huge
    :raise TypeError: if huge_theta is no boolean
    :param transformer: True if ESRI:53042 is used
    :raise TypeError: if huge_theta is no boolean
    :param theta_attack:  beamwidth attack antenna
    :raise TypeError: if theta_attack outside of 0-180
    :param sat_number: number of satellite to investigate
    :raise TypeError: if sat_number no integer
    :param atmosphere: height of atmosphere
    :raise TypeError: if atmosphere outside of 0-500
    """
    if not is_boolean(huge_theta):     
        warnings.warn("'huge_theta' is no boolean!", RuntimeWarning)
    if not is_boolean(transformer):     
        warnings.warn("'transformer' is no boolean!", RuntimeWarning)
    if not check_if_integer(sat_number):
        warnings.warn("'sat_number' is no integer!", RuntimeWarning)
    if not check_if_integer(time_step):
        warnings.warn("'time_step' is no integer!", RuntimeWarning)
    if not is_value_in_range(theta_target,(0,180)):
        warnings.warn("'theta_3dB' is out of range!", RuntimeWarning)
    if not is_value_in_range(theta_attack,(0,180)):
        warnings.warn("'theta_attack' is out of range!", RuntimeWarning)
    if not is_value_in_range(atmosphere,(0,500)):
        warnings.warn("'atmosphere is out of range!", RuntimeWarning)
    is_orientation_right(orientation) 

def input_checker_past(theta_target,time_step,sat_number,atmosphere,orientation):
    """
    This function checks if the input values are in the correct range. 

    :param theta_target: beamwidth target antenna
    :raise TypeError: if theta_target outside of 0-180 
    :param time_step: time step of caclulation
    :raise TypeError: if time_step is no integer 
    :param sat_number: number of satellite to investigate
    :raise TypeError: if sat_number no integer
    :param atmosphere: height of atmosphere
    :raise TypeError: if atmosphere outside of 0-500
    """
    if not check_if_integer(sat_number):
        warnings.warn("'sat_number' is no integer!", RuntimeWarning)
    if not check_if_integer(time_step):
        warnings.warn("'time_step' is no integer!", RuntimeWarning)
    if not is_value_in_range(theta_target,(0,180)):
        warnings.warn("'theta_3dB' is out of range!", RuntimeWarning)
    if not is_value_in_range(atmosphere,(0,500)):
        warnings.warn("'atmosphere is out of range!", RuntimeWarning)
    is_orientation_right(orientation) 

def timestep_2_datetime(sim_start:datetime,time_step:int,specific_time_step:int):
    """
    This function determines the datetime of a specific timestep.

    :param sim_start: dateteime of simulation start
    :type sim_start: datetime object
    :param time_step: timestep to inversigate
    :type time_step: int
    :param specific_time_step: time steo to investigate
    :type specific_time_step: int
    """
    print(f'This is the datetime for the timestep: [{specific_time_step}]')
    time = get_time(sim_start.year,sim_start.month,sim_start.day,sim_start.hour,sim_start.minute,sim_start.second,time_step,specific_time_step).utc_strftime()
    print('\n')
    print('\t', end=' ')
    print(f'-- {time} --')
    return

def max_distance_checker(sat_list,max_distance):
    """
    This funciton determined intervals in which the distance betweenn satellites
    is lower than the maximum defined distance. 
    
    :param sat_list: list of satellite objects
    :type sat_list: list[Satellites]
    :param max_distance: maximum distance of satellites [km]
    :type max_distance: int
    """
    calc_len = len(sat_list[0].x)
    num_satellites = len(sat_list)
    for sat in sat_list:
        sat.max_distance = []
    
    for i,sat in enumerate(sat_list):
        for k in range(num_satellites):
            max_d_list = []
            if k==i:
                sat.max_distance.append([False] * calc_len)
                continue
            for d in range(calc_len):
                if sat.distance[k][d] > max_distance:
                    max_dis = False
                else: 
                    max_dis = True
                max_d_list.append(max_dis)
            sat.max_distance.append(max_d_list)