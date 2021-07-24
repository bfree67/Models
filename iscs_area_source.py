# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:22:30 2021

@author: Brian
"""

import numpy as np
from numpy.linalg import lstsq
import scipy.special as sp
import scipy.integrate as spi
from scipy.spatial.distance import cdist
import math
import PG_Turner as pg  # local library to compute PG coefficients
from haversine import haversine  # https://pypi.org/project/haversine/

def fit_line(pts):
    #pts = [(30, 220),(1385, 1050)]
    x_coords, y_coords = zip(*pts)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    return m, c

def closest_pt(pts):
    
    m, c = fit_line(pts)
    
    x1 = pts[0][0]; y1 = pts[0][1]
    x2 = pts[1][0]; y2 = pts[1][1]
    
    x_line = np.linspace(x1,x2,100)
    f_line = lambda x_line: m*x_line + c
    
    ## make a special case if angle is 0 and line is straight
    if orient_ang == 0.:
        y_line = np.linspace(y1, y2, len(x_line))
    else:
        y_line = f_line(x_line)
        
    coords = list(zip(x_line, y_line))
    
    # find coordinates closes to the drone 
    return coords[cdist(coords, drone).argmin()], round(cdist(coords, drone).min(),1)

def bearing(coord1, coord2):
    ### calculates bearing between coord1 and coord2
    ### coord must be in tuples (lat,long)
    
    lat1 = coord1[0]; long1 = coord1[1]
    lat2 = coord2[0]; long2 = coord2[1]
    
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

    return round(brng,1)

def haversine_r(coord_s, coord_r):
    # return distance between coordinates
    return round(haversine(coord_s, coord_r),3)

def pg_sigma(a,b,c,x):
    ## calculate the sigma based on input coefficients and distance (x)
    return a*(x**(b+c*math.log(x)))

def PQ_sigma_y(A):
    ### stability class coefficients for sigma_y
    if A == 'A':
        PQ_sy_class = np.asarray([209.6, 0.8804, -0.006902])  ## Class A
    if A == 'B':
        PQ_sy_class = np.asarray([154.7, 0.8932, -0.006271])  ## Class B
    if A == 'C':
        PQ_sy_class = np.asarray([103.3, 0.9112, -0.004845])  ## Class C
    if A == 'D':
        PQ_sy_class = np.asarray([68.28, 0.9112, -0.004845]) ## Class D
    if A == 'E':
        PQ_sy_class = np.asarray([51.05, 0.9112, -0.004845]) ## Class E
    if A == 'F':
        PQ_sy_class = np.asarray([33.96, 0.91121, -0.004845]) ## Class F

    ### assign coefficients
    ay = PQ_sy_class[0]
    by = PQ_sy_class[1]
    cy = PQ_sy_class[2]
    
    return ay, by, cy

def PQ_sigma_z(A):
    ### stability class coefficients for sigma_z
    if A == 'A':
        PQ_sz_class = np.asarray([417.9, 2.058, 0.2499])  ## Class A
    if A == 'B':
        PQ_sz_class = np.asarray([109.8, 1.064, 0.01163])  ## Class B
    if A == 'C':
        PQ_sz_class = np.asarray([61.14, 0.9147, 0.])  ## Class C
    if A == 'D':
        PQ_sz_class = np.asarray([30.38, 0.7306, -0.032]) ## Class D
    if A == 'E':
        PQ_sz_class = np.asarray([21.14, 0.6802, -0.04522]) ## Class E
    if A == 'F':
        PQ_sz_class = np.asarray([13.72, 0.6584, -0.05367]) ## Class F

    ### assign coefficients
    az = PQ_sz_class[0]
    bz = PQ_sz_class[1]
    cz = PQ_sz_class[2]
    
    return az, bz, cz
    
#########################
### set weather variables
ws = 3 # mwx["WS"] # wind speed in [m/s]
wd = 270 # mwx["WD"] # wind direction in [deg] from the source
wd_d = wd + 180 # convert wind direction to the source

if wd_d > 180:
    wd_compare = wd_d - 180.0
else:
    wd_compare = wd_d
    
wd_compare_u = wd_compare - 90 # convert compass to unit circle deg [gr deg]
    
temp_a = 273.1 + 25 # mwx["TEMPERATURE"] # ambient temperature in [deg C] converted to [deg K]
release_temp = temp_a
cloud_cover = 5 # mwx["CLOUDS"] # cloud coverage in a scale from 1 - 10

# cloud ceiling in [ft]
cloud_ceiling = 7000

### DEFINE AREA SOURCE
#### SW Corner of area source
Lat_s = 60.172680 #source['LATITUDE_S']
Long_s = 24.692392 #source['LONGITUDE_S']
coord_s = (Lat_s, Long_s)
### dimensions
x_dist = 100 #[m]
y_dist = 200 #[m]
orient_ang_c = -5. # deg from North
orient_ang_u = orient_ang_c - 90 # convert to unit circle deg [UC deg]

##############################################
### set request variables from drone
Lat_r = 60.172658 #prf['LATITUDE_U']
Long_r = 24.697688 #prf['LONGITUDE_U']
coord_r = (Lat_r, Long_r)
# collection altitude in [m]
dz = 18 # prf['ALTITUDE']
# assume initial release height in [m] for surface at ground level
release_h = 1

# absolute difference between wind direction and source orientation
src_wind_ang = abs(orient_ang_c - wd_compare) # [gr deg]
or_ang_r = math.radians(orient_ang_c) # convert to radians

# find near pts of source side nearest drone
x1 = round(x_dist * math.cos(or_ang_r), 1)
y1 = round(y_dist * math.sin(or_ang_r), 1)
near = (x1, y1)

# find far pts of source side nearest drone
hyp_far = round(math.hypot(x_dist, y_dist), 1)
theta_far = math.atan(y_dist/x_dist) - or_ang_r #[gr rad]
x2 = round(hyp_far * math.cos(theta_far), 1)
y2 = round(hyp_far * math.sin(theta_far), 1)
far = (x2, y2)

line_pts = [near, far]

####################################
## calc drone parameters
hyp_drone = haversine_r(coord_s, coord_r) * 1000  # haversine distance of drone from SW corner source in [m]
bearing_drone_d = bearing(coord_s, coord_r) - 90 # bearing from pt to drone. convert to [gr deg]
bearing_drone_r = math.radians(bearing_drone_d) # convert to radians

## calc component distances 
x_drone = round(hyp_drone * math.cos(bearing_drone_r), 1)
y_drone = round(hyp_drone * math.sin(bearing_drone_r), 1)

#make coordinate matrix
drone = np.asmatrix((x_drone, y_drone))  # drone coordinates relative to area source refernce base.

################################################
## find closest unit point on area source perimeter and distance to drone
min_location, d_hyp = closest_pt(line_pts)

dx = round(d_hyp * math.cos(bearing_drone_r), 1)
dy = round(d_hyp * math.sin(bearing_drone_r), 1)

###### define weather
## Generate Pasquill-Gifford (P-G) Stability Category bassesd on MWX inputs
pg_coef = pg.PG_cat(ws, Lat_s, Long_s, cloud_cover, cloud_ceiling)
## get atmospheric stability category constants 
ay, by, cy = PQ_sigma_y(pg_coef)
az, bz, cz = PQ_sigma_z(pg_coef)

sig_y = ay*(dx**(by+cy*math.log(dx))) ## coefficient for the cross-wind distribution (y-axis) output in [m]
sig_z = az*(dx**(bz+cz*math.log(dx))) ## coefficient for the vertical distribution (z-axis) output in [m]

# Define constant in area soruce equation
D = 1e-6 / (ws * math.pi * ay * az) #converts ug/s to g/s

# Define trapezoid method increment between near source side and drone in x plane
x_diff = dx/100

###### solve area source equation using trapezoid method
x_input = np.linspace(x_diff, dx, 101)
# area source equation
f  = lambda x: (x**-(by+cy*np.log(x))) * (x**-(bz+cz*np.log(x))) * sp.erfc(dy*x**-(by+cy*np.log(x))/ay)
y_input = f(x_input)
# solve definite integral for beta
beta = D * np.trapz(y_input, x = None, dx = x_diff)




