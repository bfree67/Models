# -*- coding: utf-8 -*-
import base64
import pprint as pp
import numpy as np
from numpy.linalg import lstsq
import math
from pysolar.solar import get_altitude
from haversine import haversine  # https://pypi.org/project/haversine/
import sys
import warnings
import datetime
import string
import sys
import json
import os
import pytz
warnings.filterwarnings("ignore")

############## THIS is the main entrypoint for gcloud functions

def entrypoint_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """


    # print(request)
    # print(request.get_data())
    # print(request.get_json(force=True))

    final_string = ""
    #request_data = json.loads(request.get_json(force=True))
    request_data = request # only for dev
    
    wx_data = request_data['wx']
    site_data = request_data['pmc']
    obs_data = request_data['prf']

    wx_mission_profile = request_data['wx']['MISSION_PROFILE']
    site_mission_profile = request_data['pmc']['MISSION_PROFILE']  
    obs_mission_profile = request_data['prf']['MISSION_PROFILE']

    if wx_mission_profile == site_mission_profile and wx_mission_profile == obs_mission_profile:
        
        pass

    else:
        return f'Critical Failure: Mission profiles are not the same!'

    ##################################################################
    ### SET WEATHER VARIABLES
    ### assumes drone is directly downwind of source regardless of wind direction

    ws = wx_data['WS'] # mwx["WS"] # wind speed in [m/s]

    # look for temperature but use 25 C if not available
    try:
        temp_c = wx_data['TEMPERATURE'] # mwx ['TEMPERATURE'] ambient temp [deg C]
    except:
        temp_c = 25 # assume STP (25 C)

    ambient_temp_k = 273.1 + temp_c # mwx["TEMPERATURE"] # ambient temperature in [deg C] converted to [deg K]
    
    cloud_cover = wx_data['CLOUDS'] # mwx["CLOUDS"] # cloud coverage in a scale from 1 - 10, 1 - no clouds, 10 - overcast

    # look for atmospheric pressure but us STP (1 atm) if not available
    try:
        atp = wx_data['PRESSURE']
    except:
        atp = 1     #if atm pressure is not available assume 1 atm
        
    if atp > 2:
        return f'Critical Failure: Pressure is too high! Probably not in atm.'

    # cloud ceiling in [ft]
    try:
        cloud_ceiling = wx_data['CEILING_FT']
    except:
        cloud_ceiling = 7000 # assume 7000 ft if not available
       
    try:
        wx_hght = wx_data['WX_HGT'] # in meters [m]
    except:
        wx_hght = 10 # default value

    ################################################################
    ### DEFINE SOURCE

    source_type = site_data['source1']['SOURCE_TYPE']
    
    # location type: Rural or Urban
    try:
        location_type = site_data['LOCATION_TYPE']
    except:
        location_type = 'Rural'
    
    #### Center of area source
    Lat_s = site_data['source1']['LATITUDE_S'] #source['LATITUDE_S']
    Long_s = site_data['source1']['LONGITUDE_S'] #source['LONGITUDE_S']
    coord_s = (Lat_s, Long_s)
    
    # Area Surce
    if source_type == 'Area':
       
        ### area source dimensions
        Sx_dist = site_data['source1']['X_DIST'] #[m]
        Sy_dist = site_data['source1']['Y_DIST'] #[m]
        
        ## calculate site area based on circular or rectangular conditions
        try:
            source_shape = site_data['source1']['SOURCE_SHAPE']
        except:
            source_shape = 'Circle'
        
        if source_shape == 'Circle':
        
            S_area = round(math.pi*(Sy_dist/2)**2,1) #[square meters] round source
        
        else:
            S_area = round(Sx_dist * Sy_dist,1) # rectangular source
        
        # assume source orientation that drone is on x-axis and y-axis is dispersion away
        sigma_y0 = Sy_dist/4.2 # calculate initial sigma_yo of virtual source
        
        release_temp_k = ambient_temp_k  # use ambient air temperature for release temp
        
    else:
        # for flare and point sources
        sigma_y0 = 0 
        release_temp_k = site_data['source1']['RELEASE_TEMP'] + 273.1
        
        if release_temp_k < ambient_temp_k:
            release_temp_k = ambient_temp_k    
        
        release_vel = site_data['source1']['RELEASE_VEL']
        
        if release_vel == 0 and source_type != 'Area':
            return f'Critical Failure: Release Velocity cannot be 0.'
            
        S_area = 1 # dummy variable not used for pt/flare calcs

    # release height is surface for area, top of stack for pt and effective release for flare
    release_h = site_data['source1']['RELEASE_HGT']
    
    if release_h == 0 and source_type != 'Area':
        return f'Critical Failure: Release Height cannot be 0.'
    
    # interior diameter is release diameter of flame for flare
    interior_d = site_data['source1']['RELEASE_DIAM']
    
    if interior_d == 0 and source_type != 'Area':
        return f'Critical Failure: Interior Diameter cannot be 0.'

    ##############################################
    ### set drone observation data

    Lat_r = obs_data['LATITUDE_U'] #prf['LATITUDE_U']
    Long_r = obs_data['LONGITUDE_U'] #prf['LONGITUDE_U']
    coord_r = (Lat_r, Long_r)

    # observation datetime
    obs_date_string = obs_data['TIMESTAMP'] #prf['TIMESTAMP'] must be in 24 hr/UTC format

    # collection altitude in [m]
    dz = obs_data['ALTITUDE'] # prf['ALTITUDE']

    pollutant_dict = obs_data['POLLUTANTS']

    ####################################
    ## calc drone parameters
    hyp_drone = haversine_r(coord_s, coord_r) * 1000  # haversine distance of drone from source center in [m]

    ###### define weather sigmas

    ## prepare observation date string
    date_format = '%Y-%m-%d %H:%M:%S'
    obs_date = datetime.datetime.strptime(obs_date_string, date_format)

    ## Generate Pasquill-Gifford (P-G) Stability Category based on MWX inputs
    pg_coef = PG_cat(ws, Lat_s, Long_s, cloud_cover, cloud_ceiling, obs_date)

    ## get atmospheric stability category constants 
    cy, dy = PQ_sigma_y(pg_coef)
    az, bz = PQ_sigma_z(pg_coef)

    ## make initial text output
    final_string += '\nMission Profile: ' + wx_mission_profile + ' '
    final_string += '\nObservation taken {} m from source center at altitude of {} m AGL'.format(hyp_drone, dz)

    if source_type == 'Area':
        final_string += '\nArea source with X dimensions of {} m and Y dimensions {} m'.format(Sx_dist, Sy_dist)
        final_string += '\nTotal area is {}'.format(S_area)
        
    elif source_type == 'Point':
        final_string += '\nPoint source release height is {} m and interior diameter is {} m'.format(release_h, interior_d)
        final_string += '\nPoint source release temp is {} C and release velocity is {} m/s'.format(release_temp_k - 273.1, release_vel)
     
    elif source_type == 'Flare':
        final_string += '\nFlare effective release height is {} m and interior diameter is {} m'.format(release_h, interior_d)
        
    # loop through pollutants
    for pollutant in list(pollutant_dict.keys()):
        
        # collect drone measured concentrations for each pollutant
        pollutant_units = pollutant_dict[pollutant]['units']
        pollutant_value = pollutant_dict[pollutant]['value'] #[ppm]
        
        # if there is no molecular weight assigned, take the pollutant
        try:
            mw_value = pollutant_dict[pollutant]['mw'] #[pollutant molecular weight in g/mole]                                       
            
            # check to see if MW is numeric
            if type(mw_value) == int or type(mw_value) == float:
                pass
            
            else:
                mw_value = pollutant
        
        # on error, assign value to pollutant name and use MW dictionary
        except:
            mw_value = pollutant
        
        # convert ppm to ug/m3
        if pollutant_units == 'ppm':
            pollutant_ug_m3 = ppm_ug_m3(pollutant_value, mw_value, release_temp_k, atp)
        
        else:
            pollutant_ug_m3 = pollutant_value
            
            
        ## use numerical iteration to find x_0 for area source only
        if source_type == 'Area':
            for x in range(1, 100*Sx_dist):
                
                x_km = x/1000 # convert x to [km]
                
                # find initial sigma value constants 465.11628 & 0.017453293 are from EPA ISC model
                delta_sigma = abs(sigma_y0 - 465.11628 * x_km * math.tan(0.017453293 * (cy - dy * math.log(x_km))))
                
                if delta_sigma < 1:
                    x_0 = x # assign x_0 in [m]
                    break
        
        else:
            x_0 = 0
            
        # set virtual source location based on drone distance and X_0
        dx = hyp_drone + x_0 # in [m]
        dx_km = dx/1000 # convert to [km]
        
        th = 0.017453293 * (cy - dy * math.log(dx_km))
        sig_y =  465.11628 * dx_km * math.tan(th) ## coefficient for the cross-wind distribution (y-axis) output in [m]
        
        sig_z = az*dx_km ** bz ## coefficient for the vertical distribution (z-axis) output in [m]
        
        '''### calculate Q (emission rate)'''
        
        # correct wind speed based on height: set default location = Rural and 10 m collection hgt
        ws_hgt = wind_hgt_correct(ws, release_h, pg_coef, location_type, wx_hght)
        
        # account for plume rise in point sources only (flares should already be account for)
        if source_type == 'Point':
            h_eff = plume_rise(release_h, release_vel, interior_d, ws_hgt)
            
        else:
            h_eff = release_h
        
        ## calculate beta where C = Q * beta where beta is in [m3/s]
        beta = ((1/(2*math.pi*ws_hgt*sig_y*sig_z))*
                math.exp((-dy**2)/(2*sig_y**2))*
                (math.exp(-(dz-h_eff)**2/(2*sig_z**2)) + math.exp(-(dz+h_eff)**2/(2*sig_z**2))))
        
        pt_source_rate = pollutant_ug_m3 / beta
        
        area_rate = pt_source_rate / S_area
        
        final_string += '\nObserved {} concentration was {} {}'.format(pollutant, pollutant_value, pollutant_units)
        
        if source_type == 'Area':
            final_string += '\n{} area source emission rate is {:0.2e} ug/s/m^2'.format(pollutant, area_rate)

        else:
            final_string += '\n{} source emission rate is {:0.4e} ug/s'.format(pollutant, pt_source_rate)

            
    return json.dumps(final_string)

################################ METHODS

def haversine_r(coord_s, coord_r):
    # return distance between coordinates
    return round(haversine(coord_s, coord_r),3)

def PQ_sigma_y(A):
    ### stability class coefficients for sigma_y using EPA ISC method
    if A == 'A':
        PQ_sy_class = np.asarray([24.167, 2.5334])  ## Class A
    if A == 'B':
        PQ_sy_class = np.asarray([18.333, 1.8096])  ## Class B
    if A == 'C':
        PQ_sy_class = np.asarray([12.5, 1.0896])  ## Class C
    if A == 'D':
        PQ_sy_class = np.asarray([8.333, 0.72382]) ## Class D
    if A == 'E':
        PQ_sy_class = np.asarray([6.25, 0.54287]) ## Class E
    if A == 'F':
        PQ_sy_class = np.asarray([4.1667, 0.36191]) ## Class F

    ### assign coefficients
    cy = PQ_sy_class[0]
    dy = PQ_sy_class[1]

    return cy, dy

def PQ_sigma_z(A):
    ### stability class coefficients for sigma_z
    if A == 'A':
        PQ_sz_class = np.asarray([122.8, 0.94470])  ## Class A (<0.1 km only)
    if A == 'B':
        PQ_sz_class = np.asarray([90.673, 0.93198])  ## Class B (<0.2 km only)
    if A == 'C':
        PQ_sz_class = np.asarray([61.141, 0.9147])  ## Class C (all)
    if A == 'D':
        PQ_sz_class = np.asarray([34.459, 0.86974]) ## Class D (<0.3 km only)
    if A == 'E':
        PQ_sz_class = np.asarray([24.26, 0.8366]) ## Class E (<0.1 km only)
    if A == 'F':
        PQ_sz_class = np.asarray([15.209, 0.81558]) ## Class F (<0.2 km only)

    ### assign coefficients
    az = PQ_sz_class[0]
    bz = PQ_sz_class[1]

    return az, bz


####################################### ANOTHER FILE


def solar_altitude(lat = 42.206, long = -71.382, 
                   date = datetime.datetime.now()):
    
    '''
    
    calculates solar altitude based on lat/long and time of observer
    has default lat/long as uses time of request

    Changes made to solar_altitude() function to account for date default
    Assumes date is supplied as 24 hr datetime object - should be referenced to UTC

    
    '''

    date_tz = date.astimezone()  # assume its UTC
    altitude = round(get_altitude(lat, long, date_tz),1) # get_altitude from pysolar
    
    print(date_tz, altitude)
    
    if altitude <= 0:
        altitude = 0
        
    return altitude

def insolation_class(solar_altitude):
    
    '''
    
    select insolation class number based on solar altitude
    calculated with function solar_attitude()
    
    '''
    if solar_altitude > 60:
        insol_class = 4
    
    if solar_altitude <= 60 and solar_altitude > 35:
        insol_class = 3
        
    if solar_altitude <= 35 and solar_altitude > 15:
        insol_class = 2
        
    if solar_altitude <= 15:
        insol_class = 1
        
    return insol_class

def nri(sol_alt, cloud_cover,cloud_ceiling):
    
    '''
    
    calculate Net Radiation Index (NRI) using cloud coverage (1 - 10) and cloud ceiling in ft
    calls fucntion insolation_class() to get base NRI
    
    '''
    
    nr_index = insolation_class(sol_alt)
    nr_mod = 0
    
    if cloud_cover == 10:  # 10 = total overcast
        nr_index = 0
        
    if cloud_cover > 5:
        if cloud_ceiling < 7000 and cloud_cover != 10:  ## units in [ft] (aviation)
            nr_mod = -2
        if cloud_ceiling >=7000 and cloud_ceiling < 16000:
            nr_mod = -1
                        
    return nr_index + nr_mod

def wind_class(ws):
    
    '''
    
    create wind speed index based on wind speed in [m/s]
    
    '''
    
    if ws < 0.8:
        wc = 0
    
    if ws >= 0.8 and ws < 1.9:
        wc = 1

    if ws >= 1.9 and ws < 2.9:
        wc = 2

    if ws >= 2.9 and ws < 3.4:
        wc = 3
        
    if ws >= 3.4 and ws < 3.9:
        wc = 4
        
    if ws >= 3.9 and ws < 4.9:
        wc = 5
    
    if ws >= 4.9 and ws < 5.5:
        wc = 6
        
    if ws >= 5.5 and ws < 5.9:
        wc = 7
        
    if ws >= 5.9:
        wc = 6
        
    return wc

def PG_cat(ws, lat, long, cloud_cover, cloud_ceiling, obs_date):
    
    '''
    
    PG classes into a dictionary where list rank goes NRI 4, 3, 2, 1, 0, -1, -2
    
    '''
    pg_dict = {0: [1,1,2,3,4,6,7],
               1: [1,2,2,3,4,6,7],
               2: [1,2,3,4,4,5,6],
               3: [2,2,3,4,4,5,6],
               4: [2,2,3,4,4,4,5],
               5: [2,3,3,4,4,4,5],
               6: [3,3,4,4,4,4,5],
               7: [3,3,4,4,4,4,4],
               8: [3,4,4,4,4,4,4]}
    
    # make list of upper case characters A, B, C, D...
    char_list = list(string.ascii_uppercase)
    
    sol_alt = solar_altitude(lat, long, obs_date)  # added obs_date (must be in 24 hr/UTC datetime object)
    
    nri_new = nri(sol_alt, cloud_cover, cloud_ceiling)
    
    pg = pg_dict[wind_class(ws)][4-nri_new]  ## Extract PG class
    pg_char = char_list[pg-1]    ## PG alphabet class
    
    return pg_char

def wind_hgt_correct(ws, release_h, pg_coef, location = 'Rural', wind_h = 10):
    
    '''
    corrects wind speed based on release height using ISC model (eq 1-6)
    
    Inputs:
        ws is measured wind speed
        release_h is effective release height
        location is 'Rural' or 'Urban'. Default it Rural
        wind_h is height wind measurement is taken. Default is 10 m
        pg_coeff is the P-G coefficient (A - F)
    
    '''
    
    p_coef_dict = {"Urban":{"A":0.15, "B":0.15, "C":0.20, "D":0.25, "E":0.3, "F":0.3},
    "Rural":{"A":.07, "B":.07, "C":0.1, "D":0.15, "E":0.35, "F":0.55} }
    
    p_coeff = p_coef_dict[location][pg_coef]
    
    if release_h <= wind_h:
        return ws
    
    else:
        return round(ws * (release_h/wind_h)**p_coeff, 3)
    
def plume_rise(release_h, release_vel, interior_d, ws):
    
    '''
    Estimates plume rise for point source only using ISC model (eq 1-7)
    
    Inputs:
        release_h is given pt source stack height
        release_vel is release gas velocity
        interior_d is release diameter
        ws is measured wind speed
        
    Output:
        Effective release height
    
    '''
    
    if release_vel >= 1.5*ws:
        
        release_h += release_h
        
    else:
    
        release_h += 2 * interior_d * ((release_vel/ws) - 1.5)
    
    return release_h

 
def ppm_ug_m3(C_ppm, MW_value, T_c = 25, P_atm = 1):
    
    '''
    
    convert concentration in ppm to ug/m3
    default Temperature and pressure are STP (1 atm and 25 deg C)
    added 21 Sep 2022
    
    '''

    if type(MW_value) == int or type(MW_value) == float:
        MW = MW_value
    
    else:
        # MW_value is a string and therefore make a dictionay of MW values
        # make dictionary of pollutant MW in g/m3
        # ref: https://teesing.com/en/library/tools/ppm-mg3-converter
        
        print('Y')
        
        MW_dict = { 
                  'H2S': 34.08,
                  'CO': 28.01,
                  'DMS': 62.13,
                  'SO2': 64.06,
                  'NO2': 46.0,
                  'NO': 30,
                  'NH3': 17.03,
                  'VOC':78.95,
                  'CH4': 16.04
                  }
        
        #if MW pollutant is not in the list, quit
        if MW_value in MW_dict and type(MW_value) == str:
            MW = MW_dict[MW_value]
            
        else:
            print('{} not in the pollutant dictionay'.format(MW_value))
            sys.quit()
            
    T_k = T_c + 273.1 # convert temperature [C] to Kelvin
    R = 8.205736*(10**-5) # ideal gas constant in [m3⋅atm⋅K−1⋅mol−1]
    
    return round(C_ppm * MW * P_atm / (R * T_k), 1)

    
############### read json
def read_json(json_file):
    
    '''
    read json file and convert to local dictionary
    
    input: json_file is path + filename.json
    
    output: dictionary json_data
    
    added 27 Sep 2022
    '''
    
    with open(json_file, 'r') as js:
        json_data = json.load(js)
        
    return json_data

json_file = r'C:\TARS\AAAActive\Aeromon\aeromon_cloud\example3_1.json'
request_data = read_json(json_file)

dump = entrypoint_http(request_data)
print(dump)
