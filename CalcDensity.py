#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import yaml
from Cluster_functions import read_file, get_indices_sparse, unnest, great_circle_np
from datetime import datetime as dt, timedelta as td
import time
from scipy.sparse import dok_matrix
from sparse import DOK
from numpy import loadtxt

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)
    
calcDensity = True
distchar = "250km"
dist_thresh = 250 

#########################
# Functions
#########################
def great_circle(lat1, long1, lat2, long2):

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
        
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
        
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
        
    # Compute spherical distance from spherical coordinates.
        
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    if(cos > 1):
        cos = 1
    arc = math.acos( cos )

    # Calculate distance in kilometers
    dist = arc*6378.16

    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return dist
def calc_Rossby_radius(lat=45,N=1.3e-2,H=10):
	return N*H/(2*7.29*10**-5*np.sin(lat*np.pi/180))

#Define grid
lats = np.arange(90,-90.1,-1.5)
lons = np.arange(-180,180,1.5)

#########################
# Load storm tracks 
#########################

#Storm tracks file
st_file = Options["st_file"]
nrskip = Options["nrskip"]

str_id, str_nr, str_dt, str_lat, str_lon = read_file(st_file)
str_pres   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[5],dtype=float)
str_lapl   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[9],dtype=float)

#Convert to an array
str_dt          = np.array(str_dt)
str_month = np.array([x.month for x in str_dt])
str_year = np.array([x.year for x in str_dt])
#str_id = str_id - np.nanmin(str_id) + 1

nrstorms = len(np.unique(str_id))

#Construct array with datetimes
dt_array = []
for yidx in range(1979,2017):
	# To get year (integer input) from the user
	# year = int(input("Enter a year: "))
	if (yidx % 4) == 0:
		leapyear = True
		nr_times = 366*4 #(whole year) 364 (just winter)
	else:
		leapyear = False
		nr_times = 365*4 #(whole year) 360 (just winter)

#	start = dt(yidx, 12, 1, 0) (just winter)
	start = dt(yidx, 1, 1, 0) 

	dt_array_temp = np.array([start + td(hours=i*6) for i in range(nr_times)])
	dt_array.extend(dt_array_temp)
    
#Convert to an array
str_reltime     = np.zeros(len(str_lon))
str_reltimestep = np.zeros(len(str_lon))
str_reltimestepAtl = np.zeros(len(str_lon))
str_isclustered = np.zeros(len(str_lon))
str_rot         = np.zeros(len(str_lon))
    
#########################
# Get indices of storms 
# so that ids_storms[id] gives the ids in the arrays
# str_id, str_lon,.. belonging to that specific storm
#########################
str_id = str_id - np.nanmin(str_id) + 1

uniq_ids = np.unique(str_id)
ids_storms = get_indices_sparse(str_id)
nrstorms = len(uniq_ids)

#####################################################
# 1. Determine storm density
# i.e. first time a storm hits a particular region
#####################################################
start = time.time()

if(calcDensity):

    #Define arrays
    if(Options["frameworkSparse"] == True):
        storms = DOK((len(dt_array),len(lats),len(lons))) #Cyclone centre density
        tracks = DOK((len(dt_array),len(lats),len(lons))) #Track density
        lysis  = DOK((len(dt_array),len(lats),len(lons))) #Track density
        genesis = DOK((len(dt_array),len(lats),len(lons))) #Track density
    else:
        storms = np.zeros((12,len(lats),len(lons))) #Cyclone centre density
        tracks = np.zeros((12,len(lats),len(lons))) #Track density
        lysis  = np.zeros((12,len(lats),len(lons))) #Track density
        genesis  = np.zeros((12,len(lats),len(lons))) #Track density        


    #Pressure stats
    storms_minpres = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density where min. pressure occurs
    storms_mindpdt = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density where min dpdt occurs

    #Loop over storm_tracks
    nr_storms = np.max(str_id)

    #Arrays for saving storm stats
    minpres_storms = np.zeros(nr_storms)
    maxlapl_storms = np.zeros(nr_storms)

    if(distchar != "Rossby"):
        dist_temp = dist_thresh

    #Loop over storms
    for strid in range(nr_storms): #clust_idxs: #range(1,nr_storms+1): nr_storms
        print("Storm: " + str(strid))
        temp_lat = str_lat[ids_storms[uniq_ids[strid]]]
        temp_lon = str_lon[ids_storms[uniq_ids[strid]]]
        temp_dt  = str_dt[ids_storms[uniq_ids[strid]]]
        temp_pres = str_pres[ids_storms[uniq_ids[strid]]]
        temp_lapl = str_lapl[ids_storms[uniq_ids[strid]]]
        temp_minpres = np.nanmin(temp_pres)
        temp_maxlapl = np.nanmax(str_lapl[ids_storms[uniq_ids[strid]]])
        temp_meanlapl = np.nanmean(str_lapl[ids_storms[uniq_ids[strid]]]) #str_lapl[str_id == stridx]
        temp_dpdt = np.zeros(len(temp_pres))
        temp_dpdt[1:-1] = temp_pres[2:] - temp_pres[:-2]
        #minpresidx = np.nanargmin(temp_pres)
        mindpdtidx = np.nanargmin(temp_dpdt)
        minpresidx = np.nanargmin(temp_pres)
        str_reltime[ids_storms[uniq_ids[strid]]] =  np.arange(0,len(temp_lat))*0.25 - minpresidx *0.25
        str_reltimestep[ids_storms[uniq_ids[strid]]] =  np.arange(0,len(temp_lat)) - minpresidx
        temp_rot = np.zeros(len(temp_lat))
        #temp_connected = str_connected[str_id == stridx]

        #Switch to prevent double counting	
        bool_tracks   = np.full((len(lats),len(lons)),False)

        #Loop over times
        for tridx in range(len(temp_dt)):
            #print("Idx: " + str(tridx))

            #Find time index for current time of storm track in result array
            if (temp_dt[tridx] in dt_array):
                tidx = dt_array.index(temp_dt[tridx])
                midx = temp_dt[tridx].month - 1
                #Loop over lons and lats
                for latidx in np.where(np.abs(temp_lat[tridx] - lats) <= dist_temp/111)[0]: 
                    lattemp = np.abs(lats[latidx])
                    if(distchar == "Rossby"):
                        if(lattemp > 20):
                            dist_temp = np.abs(calc_Rossby_radius(lat=lattemp))
                        else:
                            dist_temp = calc_Rossby_radius(lat=20.0)
                    dists = great_circle_np(temp_lat[tridx],temp_lon[tridx], lats[latidx],lons) < dist_temp
                    for lonidx in np.where(dists)[0]:
                        #Calculate distance to grid point
                        #If distance is < 500 km increase nr. of storms
                        storms[midx,latidx,lonidx] += 1
                        if(bool_tracks[latidx,lonidx] == False):						
                            tracks[midx,latidx,lonidx] += 1
                        if(tridx == 0):						
                            genesis[midx,latidx,lonidx] += 1
                        if(tridx == len(temp_dt) - 1):	
                            lysis[midx,latidx,lonidx] += 1
                        if(tridx == minpresidx):
                            storms_minpres[midx,latidx,lonidx] += 1
                        if(tridx == mindpdtidx):
                            storms_mindpdt[midx,latidx,lonidx] += 1 									
                        bool_tracks[latidx,lonidx] = True

print(time.time() - start)