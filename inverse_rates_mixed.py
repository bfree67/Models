# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:41:16 2021

back calcualtes emission rates for mulitple sources
assumes each source has the same emission rate

Q = C / (Beta1 + Beta2 + Beta3 + ....)

https://pypi.org/project/haversine/

ver 2 - added plume generation using the same model that did the back-calc
ver 3 - update wd to account for wind blowing "from" a direction

@author: Brian
"""

import numpy as np
import math
import PG_Turner as pg  # local library to compute PG coefficients
from haversine import haversine  # https://pypi.org/project/haversine/

############## Functions ##################

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

def ws_power(ws,release_h, pg):
    #assumes rural only conditions and measurments at 10 [m]
    #returns EPA power law modified wind speed in [m/s]
    
    if pg == 'A':
        p = 0.07  ## Class A
    if pg == 'B':
        p = 0.07  ## Class B
    if pg == 'C':
        p = 0.1  ## Class C
    if pg == 'D':
        p = 0.15 ## Class D
    if pg == 'E':
        p = 0.35 ## Class E
    if pg == 'F':
        p = 0.55 ## Class F
        
    return ws * (release_h/10)**p

def plume_rise(release_temp, release_d, release_v, temp_a, ws_p):
    
    # calculate buoyancy flux F [m4/s3]
    g = 9.861 # gravitational acceleration [m/s2]
    F = (g*release_v*release_d**2)*(release_temp - temp_a)/(4*release_temp)
    
    if F <= 55 :
        delta_h = (21.425*F**0.75)/ws_p
    
    if F > 55 :
        delta_h = (38.71*F**0.6)/ws_p
        
    return delta_h
    

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
    return round(haversine(coord_s, coord_r),3)

def pg_sigma(a,b,c,x):
    ## calculate the sigma based on input coefficients and distance (x)
    return a*(x**(b+c*math.log(x)))

def beta_calc_pt(source, mwx, prf):

    #####################################
    ##### calculates beta for a single point source
    #####################################
    
    ### set source variables
    Lat_s = source['LATITUDE_S']
    Long_s = source['LONGITUDE_S']
    coord_s = (Lat_s, Long_s)
    
    # release height in [m]
    if source['RELEASE_HGT'] == 0:
        release_h = 10
    else:
        release_h = source['RELEASE_HGT']
    
    # release inner diameter in [m]
    if source['RELEASE_DIAM'] == 0:
        release_d = 1.0
    else:
        release_d = source['RELEASE_DIAM']
       
    # release velocity in [m/s]
    if source['RELEASE_VEL'] == 0:
        release_v = 5.0
    else:
        release_v = source['RELEASE_VEL']
    
    # release temp in [deg K]
    if source['RELEASE_TEMP'] == 0:
        release_temp = 373.0
    else:
        release_temp = source['RELEASE_TEMP']
    
    #########################
    ### set weather variables
    ws = mwx["WS"] # wind speed in [m/s]
    wd = mwx["WD"] # wind direction in [deg] from the source
    wd_d = wd + 180 # convert wind direction to the source
    temp_a = mwx["TEMPERATURE"] + 273.1# ambient temperature in [deg C] converted to [deg K]
    cloud_cover = mwx["CLOUDS"] # cloud coverage in a scale from 1 - 10
    
    # cloud ceiling in [ft]
    if mwx["CEILING_FT"] == 0:
        cloud_ceiling = 7000
    else:
        cloud_ceiling = mwx["CEILING_FT"]
    
    ###########################
    ### set request variables
    Lat_r = prf['LATITUDE_U']
    Long_r = prf['LONGITUDE_U']
    coord_r = (Lat_r, Long_r)
    
    # collection altitude in [m]
    dz = prf['ALTITUDE']
    
    ##########################################
    ## start calculations
    
    h = haversine_r(coord_s, coord_r)  # haversine distance of drone from source in [km]
    bearing_drone_d = bearing(coord_s, coord_r) # bearing of drone from source in [deg]
    theta_d = abs(wd_d - bearing_drone_d) # absolute difference of wind direction from drone bearing
    theta_r = math.radians(theta_d) # convert to radians
    dx = round(h * math.cos(theta_r), 3) # x component distance relative to wind direction in [km]
    dy = round(h * math.sin(theta_r), 3) * 1000 # y component distance relative to wind direction in [m]
    
    ## Generate Pasquill-Gifford (P-G) Stability Category bassesd on MWX inputs
    pg_coef = pg.PG_cat(ws, Lat_s, Long_s, cloud_cover, cloud_ceiling)
    
    ## update wind speed using power law
    ws_p = ws_power(ws, release_h, pg_coef)
    
    ## get effective release height
    h_eff = release_h + plume_rise(release_temp, release_d, release_v, temp_a, ws_p)
    
    ## get atmospheric stability category constants 
    ay, by, cy = PQ_sigma_y(pg_coef)
    az, bz, cz = PQ_sigma_z(pg_coef)
        
    sig_y = ay*(dx**(by+cy*math.log(dx))) ## coefficient for the cross-wind distribution (y-axis) output in [m]
    sig_z = az*(dx**(bz+cz*math.log(dx))) ## coefficient for the vertical distribution (z-axis) output in [m]
    
    ## calculate beta where C = Q * beta where beta is in [m3/s]
    beta = ((1/(2*math.pi*ws_p*sig_y*sig_z))*
            math.exp((-dy**2)/(2*sig_y**2))*
             (math.exp(-(dz-h_eff)**2/(2*sig_z**2)) + math.exp(-(dz+h_eff)**2/(2*sig_z**2))))
    
    return beta

def beta_calc_area(source, mwx, prf):
    #####################################
    ##### for single source, area source
    #####################################
    
    ### set source variables
    ### for area source, lat/long should be the SW corner of the rectangle
    Lat_s = source['LATITUDE_S']
    Long_s = source['LONGITUDE_S']
    coord_s = (Lat_s, Long_s)
    
    # assume initisal release height in [m] for surface at ground level
    release_h = 1
    
    x_dist = source['X_DIST'] / 1000 # convert x dist to [km]
    y_dist = source['Y_DIST']  # in [m]
    orient_ang = source['ORIENTATION']
    
    #########################
    ### set weather variables
    ws = mwx["WS"] # wind speed in [m/s]
    wd = mwx["WD"] # wind direction in [deg] from the source
    wd_d = wd + 180 # convert wind direction to the source
    
    if wd_d > 180:
        wd_compare = wd_d - 180.0
    else:
        wd_compare = wd_d
        
    # absolute difference between wind direction and source orientation
    src_wind_ang = abs(orient_ang - wd_compare)
    
    temp_a = mwx["TEMPERATURE"] + 273.1# ambient temperature in [deg C] converted to [deg K]
    release_temp = temp_a
    cloud_cover = mwx["CLOUDS"] # cloud coverage in a scale from 1 - 10
    
    # cloud ceiling in [ft]
    if mwx["CEILING_FT"] == 0:
        cloud_ceiling = 7000
    else:
        cloud_ceiling = mwx["CEILING_FT"]
    
    ###########################
    ### set request variables
    Lat_r = prf['LATITUDE_U']
    Long_r = prf['LONGITUDE_U']
    coord_r = (Lat_r, Long_r)
    
    # collection altitude in [m]
    dz = prf['ALTITUDE']
    haps = prf['POLLUTANTS']
    hap_list = list(haps.keys()) ### list of pollutant names
    
    ##########################################
    ## start calculations
    
    # if the difference between wind and orientation is > 45 deg, switch the x, y dimensions
    if src_wind_ang > 45.0:
        temp = y_dist
        y_dist = x_dist * 1000 # in [m]
        x_dist = temp/1000 # in [km]
    
    # update distances to account for wind direction    
    src_wind_ang_r = math.radians(src_wind_ang)
    x_dist_new = round(x_dist * math.cos(src_wind_ang_r), 3)
    y_dist_new = round(y_dist * math.sin(src_wind_ang_r), 3)
        
    area = (x_dist*1000) * y_dist # in [m^2]
    
    h = haversine_r(coord_s, coord_r)  # haversine distance of drone from source in [km]
    bearing_drone_d = bearing(coord_s, coord_r) # bearing of drone from source in [deg]
    theta_d = abs(wd_d - bearing_drone_d) # absolute difference of wind direction from drone bearing
    theta_r = math.radians(theta_d) # convert to radians
    dx = round(h * math.cos(theta_r), 3) # x component distance relative to wind direction in [km]
    dx_r = dx - (x_dist_new/2)
    
    dy = round(h * math.sin(theta_r), 3) * 1000 # y component distance relative to wind direction in [m]
    dy_r = dy - (y_dist_new/2)
    
    ## Generate Pasquill-Gifford (P-G) Stability Category bassesd on MWX inputs
    pg_coef = pg.PG_cat(ws, Lat_s, Long_s, cloud_cover, cloud_ceiling)
    #pg = 'C' # forced value for testing only
    
    ## get atmospheric stability category constants 
    ay, by, cy = PQ_sigma_y(pg_coef)
    az, bz, cz = PQ_sigma_z(pg_coef)
    
    ##### find virtual Xo distance of virtual source
    sigma_y_o = (x_dist/4.2)*1000 # get initial sigma based on x length and convert t0 [m]
    
    # put coefficients into quadratic formula format
    c = math.log(sigma_y_o/ay)
    b = by
    a = cy
    
    ## find distance in [km]
    # in case the 1st term is 0...
    if a == 0:
        r1 = math.exp(-c/b)
    
    # find roots
    else:
        r1 = round(math.exp((-b + (b**2 - (4*a*c))**0.5)/(2*a)),3)
        r2 = round(math.exp((-b + (b**2 + (4*a*c))**0.5)/(2*a)),3)
        
        # find error between sigma generated roots - lowest error wins
        ds1 = abs(sigma_y_o - pg_sigma(ay,by,cy,r1))
        ds2 = abs(sigma_y_o - pg_sigma(ay,by,cy,r2))
        
        if ds1 > ds2:
            r1 = r2
    
    # calculate virtual distance and effective distance
    x_virtual = r1 # in [km]
    dx_eff = dx_r + x_virtual
    
    ## get atmospheric stability category sigmas based on effective distance 
    sig_y = pg_sigma(ay,by,cy,dx_eff) ## coefficient for the cross-wind distribution (y-axis) output in [m]
    sig_z = pg_sigma(az,bz,cz,dx_eff) ## coefficient for the vertical distribution (z-axis) output in [m]
    
    ## wind speed does not need to be updated
    ws_p = ws
    
    ## get effective release height
    h_eff = release_h 
    
    ## calculate beta where C = Q * beta where beta is in [m3/s]
    beta = ((1/(2*math.pi*ws_p*sig_y*sig_z))*
            math.exp((-dy**2)/(2*sig_y**2))*
             (math.exp(-(dz-h_eff)**2/(2*sig_z**2)) + math.exp(-(dz+h_eff)**2/(2*sig_z**2))))
    
    return beta

def make_plume(source, mwx, prf, pollutant, emission_rate):
    #############################################################
    ##### calculate plume for a single point source and pollutant
    #############################################################
    
    ### set source variables
    Lat_s = source['LATITUDE_S']
    Long_s = source['LONGITUDE_S']
    coord_s = (Lat_s, Long_s)
    
    # release height in [m]
    if source['RELEASE_HGT'] == 0:
        release_h = 10
    else:
        release_h = source['RELEASE_HGT']
    
    # release inner diameter in [m]
    if source['RELEASE_DIAM'] == 0:
        release_d = 1.0
    else:
        release_d = source['RELEASE_DIAM']
       
    # release velocity in [m/s]
    if source['RELEASE_VEL'] == 0:
        release_v = 5.0
    else:
        release_v = source['RELEASE_VEL']
    
    # release temp in [deg K]
    if source['RELEASE_TEMP'] == 0:
        release_temp = 373.0
    else:
        release_temp = source['RELEASE_TEMP']
    
    #########################
    ### set weather variables
    ws = mwx["WS"] # wind speed in [m/s]
    wd = mwx["WD"] # wind direction in [deg] from the source
    wd_d = wd + 180 # convert wind direction to the source
    temp_a = mwx["TEMPERATURE"] + 273.1# ambient temperature in [deg C] converted to [deg K]
    cloud_cover = mwx["CLOUDS"] # cloud coverage in a scale from 1 - 10
    
    # cloud ceiling in [ft]
    if mwx["CEILING_FT"] == 0:
        cloud_ceiling = 7000
    else:
        cloud_ceiling = mwx["CEILING_FT"]
        
    #########################
    ### set request variables
    step = prf["outputStep"]   
    maxhgt = prf["maxHeight"]
    receptor_spc = prf["spacing"]
     
    ##########################################
    ## start calculations
      
    ## Generate Pasquill-Gifford (P-G) Stability Category bassesd on MWX inputs
    pg_coef = pg.PG_cat(ws, Lat_s, Long_s, cloud_cover, cloud_ceiling)
    
    ## update wind speed using power law
    ws_p = ws_power(ws, release_h, pg_coef)
    
    ## get effective release height
    h_eff = release_h + plume_rise(release_temp, release_d, release_v, temp_a, ws_p)
    
    ## get atmospheric stability category constants 
    ay, by, cy = PQ_sigma_y(pg_coef)
    az, bz, cz = PQ_sigma_z(pg_coef)
    
    plume = {}
    
    for dz in range(0,int(maxhgt),int(step)):
        # make height of layer
         
         pt = 0 # pt counter in layer
         plume_temp = {}
         
         for x in range (100, 10000, int(receptor_spc)):
            ## minimum distance to source is 100 [m]
            dx = x/1000 ### convert to km
            
            sig_y = ay*(dx**(by+cy*math.log(dx))) ## coefficient for the cross-wind distribution (y-axis) output in [m]
            sig_z = az*(dx**(bz+cz*math.log(dx))) ## coefficient for the vertical distribution (z-axis) output in [m]
            
            temp_c = {}
            for dy in range(0, 1000, int(receptor_spc)):
                
                ## computes half of the plume from the center line ( dy = 0)
    
                ## calculate beta where C = Q * beta where beta is in [m3/s]
                conc = emission_rate * ((1/(2*math.pi*ws_p*sig_y*sig_z))*
                        math.exp((-dy**2)/(2*sig_y**2))*
                         (math.exp(-(dz-h_eff)**2/(2*sig_z**2)) + math.exp(-(dz+h_eff)**2/(2*sig_z**2))))
                
                ## set dictionary parameters
                coords = {"geometry": {"type": "Point","coordinates":[x, dy, dz]}}
                properties = {"properties":{"pollutants": {"name":pollutant, "conc": conc, "units":"ug/m3"}}}
                temp_c["features"] = [coords, properties]
                
                #make plume layer for specific altitude
                plume_temp[str(pt)]= temp_c
                
                #update pt counter
                pt += 1
                
         ### add layer to plume
         plume['altitude_'+str(dz)] = plume_temp
         
    return plume

    
########################################################################
######## MAIN PROGRAM ##################################################
########################################################################

pmc = {"PKEY": 1,
"TIMESTAMP": "2021-06-09  5:49:47 PM",
"AREA_NAME":"Test Location",
"MISSION_PROFILE": "Test123",
"NUMBER_SOURCES": 1,
"source1": {
	"DATUM": "WGS84",
	"SOURCE_NAME": "Source 1",
	"LATITUDE_S": 60.1724,
	"LONGITUDE_S": 24.695,
	"RELEASE_HGT": 15,
	"SOURCE_TYPE": "Point",
	"X_DIST": 0,
	"Y_DIST": 0,
	"ORIENTATION": 0,
	"EMISSION_RATE": 0,
	"RELEASE_VEL": 0,
	"RELEASE_TEMP": 0,
	"RELEASE_DIAM": 1.2},
"source2": {
	"DATUM": "WGS84",
	"SOURCE_NAME": "Source 2",
	"LATITUDE_S":  60.171739,
	"LONGITUDE_S":  24.694603,
	"RELEASE_HGT": 12,
	"SOURCE_TYPE": "Point",
	"X_DIST": 0,
	"Y_DIST": 0,
	"ORIENTATION": 0,
	"EMISSION_RATE": 0,
	"RELEASE_VEL": 0,
	"RELEASE_TEMP": 0,
	"RELEASE_DIAM": 0.8},
"source3": {
	"DATUM": "WGS84",
	"SOURCE_NAME": "Source 3",
	"LATITUDE_S":  60.172680,
	"LONGITUDE_S": 24.692392,
	"RELEASE_HGT": 18,
	"SOURCE_TYPE": "Area",
	"X_DIST": 20,
	"Y_DIST": 20,
	"ORIENTATION": 15,
	"EMISSION_RATE": 0,
	"RELEASE_VEL": 0,
	"RELEASE_TEMP": 0,
	"RELEASE_DIAM": 1.3}
}

mwx = {
"PKEY": 1,
"TIMESTAMP": "2021-06-09  5:55:47 PM",
"MISSION_PROFILE": "Test123",
"TYPE": "Current",
"FORECAST_START": "",
"FORECAST_END": "",
"WS": 2.5,
"WD": 270,
"TEMPERATURE": 22,
"CLOUDS": 3,
"CEILING_FT": 7000
}

prf = {
"PKEY": 1,
"TIMESTAMP": "2021-06-09  7:49:47 PM",
"MISSION_PROFILE": "Test123",
"LATITUDE_U":  60.172658,
"LONGITUDE_U": 24.697688,
"ALTITUDE": 18,
"outputStep": 5,
"maxHeight": 20,
"spacing": 100,
"POLLUTANTS": {
	"VOC": 200,
	"CO": 3500,
	"NO2": 35
	}
}

#########################################################################
########### main program #######################

## find number of sources
rep = pmc["NUMBER_SOURCES"]

## sum betas
beta = 0.
for i in range(1,rep+1):
    source = pmc["source"+str(i)]
    
    if source["SOURCE_TYPE"] == 'Point':
        beta += beta_calc_pt(source, mwx, prf)
        
    if source["SOURCE_TYPE"] == 'Area':
        beta += beta_calc_area(source, mwx, prf)


# get list of pollutants    
haps = prf['POLLUTANTS']
hap_list = list(haps.keys()) ### list of pollutant names

# calculate emission rate assuming all sources have the same emission rate
print('Number of sources: ' + str(rep))

all_plumes={}  # define dictionary for all pollutants

for pollutant in hap_list:
    
    # get drone measured concentration from list
    conc = haps[pollutant]
    
    # calculate emission rate
    emission_rate = round((conc/beta)/1e6, 3) # in [g/s]

    print(pollutant, emission_rate, 'g/s')
    
    all_plumes["pollutant"] = make_plume(source, mwx, prf, pollutant, emission_rate)


    

        

    

    
    