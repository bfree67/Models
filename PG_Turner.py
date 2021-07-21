# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:27:54 2021

Calculate Pasquill-Gifford (P-G) Stability Category based on
-latitude/longitude 
-time
-Cloud coverage and ceiling
-Wind speed

Time is set to UTC

Uses Turner method to determine P-G categories
https://www.epa.gov/sites/production/files/2020-10/documents/mmgrma_0.pdf
http://www.webmet.com/met_monitoring/641.html

@author: Brian
"""

from pysolar.solar import get_altitude
import datetime
import string

def solar_altitude(lat = 42.206, long = -71.382, 
                   date = datetime.datetime.now(datetime.timezone.utc)):
    
    ## calculate solar altitude based on lat/long and time of observer
    
    altitude = round(get_altitude(lat, long, date),1)
    
    if altitude <= 0:
        altitude = 0
        
    return altitude

def insolation_class(solar_altitude):
    
    # select insolation class number based on solar altitude
    
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
    
    # calculate Net Radiation Index (NRI) using cloud coverage (1 - 10) and cloud ceiling in ft
    # calls insolation_class() to get base NRI
    
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
    
    ## create wind speed index based on wind speed in [m/s]
    
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

def PG_cat(ws, lat, long, cloud_cover, cloud_ceiling):
    
    ## PG classes into a dictionary where list rank goes NRI 4, 3, 2, 1, 0, -1, -2
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
    
    sol_alt = solar_altitude(lat, long)
    
    nri_new = nri(sol_alt, cloud_cover, cloud_ceiling)
    
    pg = pg_dict[wind_class(ws)][4-nri_new]  ## Extract PG class
    pg_char = char_list[pg-1]    ## PG alphabet class
    
    return pg_char
 
## Global variables

lat = 61.2      # latitude of observer
long = -23.3    # longitude of observer
cloud_cover = 6
cloud_ceiling = 8000
ws = 5

###### start program

#print(PG_cat(ws,lat, long, cloud_cover, cloud_ceiling))