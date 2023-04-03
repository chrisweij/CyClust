#!/usr/bin/env python
# -*- encoding: utf-8

#from dynlib.shorthands import dt, td, get_instantaneous, metsave, fig, np
from dynlib.settings import proj
from dynlib.context.erainterim import conf
import dynlib.context.derived
from scipy.ndimage.filters import gaussian_filter
#from geopy.distance import great_circle as great_circle_old

from datetime import datetime as dt, timedelta as td
from dynlib.metio import metopen, metsave, get_instantaneous
import matplotlib.dates as mdates 

import dynlib.diag
from collections import namedtuple
#import xarray as xr
import scipy.interpolate as ip

import copy
import matplotlib 
matplotlib.use("agg")
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
from numpy import loadtxt
#import dynlib.figures as fig
import yaml
#execfile("projection.py")
from Cluster_functions import read_file, get_indices_sparse, unnest

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)

#Add directories to path
conf.datapath.insert(0,"/Data/gfi/spengler/cwe022/EI/slope_avg/")
conf.datapath.insert(0,"//Data/gfi/spengler/cwe022/averages/")
conf.datapath.insert(2,"/Data/gfi/share/Reanalysis/ERA_INTERIM/TOPO_LANDMASK/")
conf.datapath.insert(0,"/Data/gfi/share/Reanalysis/ERA_INTERIM/6HOURLY/TENDENCY2/")

#Options
datachar = "EI" #"LeonidasAll"
#distchar = "Rossby"
distchar = "250km"
dist_thresh = 250 

#Options related to histograms
slope_bins = np.arange(0,18,0.25)

#Plotting path
#conf.plotpath = "/home/cwe022/dynlib/examples/pinto_plots/"
conf.plotpath = '/home/WUR/weije043/scripts/python/CyClust/Plots'

#Switches
calcDensity = True
calcDensity2 = False
plotDensities = True

Rearth = 6366.0e3

import math
conf.register_variable([dynlib.context.derived.ff, ], ["300", ])

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



# (c) Northern polar centered map, focussing on extratropics
def n_extratropics():
	''' Stereographic map, centered on the north pole, covering most of the northern hemisphere
	Returns
	-------
	Basemap
		map projection instance
	'''

	return Basemap(projection='npstere',boundinglat=35,lon_0=-50,resolution='c', area_thresh=10000)

n_extratropics.aspect = 1.0

# (f) Southern polar centered map, focussing on extratropics
def s_extratropics():
	''' Stereographic map, centered on the north pole, covering most of the northern hemisphere
	Returns
	-------
	Basemap
		map projection instance
	'''

	return Basemap(projection='spstere',boundinglat=-35,lon_0=0,resolution='c', area_thresh=10000)


s_extratropics.aspect = 1.0

#Construct grid
ncmask, landmask, maskgrid = dynlib.metio.metopen("ei.ans.land-sea",q="lsm")
grid = copy.copy(maskgrid)
grid.x = maskgrid.x[::3,::3] #[0:60,::]
grid.y = maskgrid.y[::3,::3] #[0:60,::]
grid.dx = maskgrid.dx[::3,::3] #[0:60,::]
grid.dy = maskgrid.dy[::3,::3] #[0:60,::]
grid.oro = maskgrid.oro[::3,::3] #[0:60,::]
grid.nx = 240
grid.ny = 121 #60
conf.gridsize = (121,240) #(60,240)

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
uniq_ids = np.unique(str_id)
ids_storms = get_indices_sparse(str_id)
nrstorms = len(uniq_ids)

#####################################################
# 1. Determine storm density
# i.e. first time a storm hits a particular region
#####################################################
if(calcDensity):

	#Define arrays
	storms = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density
	tracks = np.zeros((len(dt_array),len(lats),len(lons))) #Track density
	lysis  = np.zeros((len(dt_array),len(lats),len(lons))) #Track density
	genesis  = np.zeros((len(dt_array),len(lats),len(lons))) #Track density

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
	for strid in range(nr_storms): #clust_idxs: #range(1,nr_storms+1):
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
				#Loop over lons and lats
				for latidx in range(len(lats)):
					lattemp = np.abs(lats[latidx])
					if(distchar == "Rossby"):
						if(lattemp > 20):
							dist_temp = np.abs(calc_Rossby_radius(lat=lattemp))
						else:
							dist_temp = calc_Rossby_radius(lat=20.0)
					if(np.abs(temp_lat[tridx] - lats[latidx]) <= dist_temp/111): 
						for lonidx in range(len(lons)):
								#Calculate distance to grid point
								dist = great_circle(temp_lat[tridx],temp_lon[tridx], lats[latidx],lons[lonidx])

								#If distance is < 500 km increase nr. of storms
#								if ((dist < 500.0)): 
#								if ((dist < 700.0)): 
								if ((dist < dist_temp)): 
									storms[tidx,latidx,lonidx] += 1
									if(bool_tracks[latidx,lonidx] == False):						
										tracks[tidx,latidx,lonidx] += 1
									if(tridx == 0):						
										genesis[tidx,latidx,lonidx] += 1
									if(tridx == len(temp_dt) - 1):	
										lysis[tidx,latidx,lonidx] += 1
									if(tridx == minpresidx):
										storms_minpres[tidx,latidx,lonidx] += 1
									if(tridx == mindpdtidx):
										storms_mindpdt[tidx,latidx,lonidx] += 1 									
									bool_tracks[latidx,lonidx] = True
									

#Another method
if(calcDensity2):
	ncmask, landmask, maskgrid = dynlib.metio.metopen("ei.ans.land-sea",q="lsm")
	#Construct grid
	import copy
	grid = copy.copy(maskgrid)
	grid.x = maskgrid.x[::3,::3] + 180.0#[0:60,::]
	grid.y = maskgrid.y[::3,::3] #[0:60,::]
	grid.dx = maskgrid.dx[::3,::3] #[0:60,::]
	grid.dy = maskgrid.dy[::3,::3] #[0:60,::]
	grid.oro = maskgrid.oro[::3,::3] #[0:60,::]
	grid.nx = 240
	grid.ny = 121 #60
	conf.gridsize = (121,240) #(60,240)

	#Define arrays
	lats = np.arange(90,-90.1,-1.5)
	lons = np.arange(0,360,1.5)
	mean_storms = np.zeros((len(lats),len(lons))) #Cyclone centre density
	mean_tracks = np.zeros((len(lats),len(lons))) #Track density
	mean_lysis  = np.zeros((len(lats),len(lons))) #Track density
	mean_genesis  = np.zeros((len(lats),len(lons))) #Track density

	for latidx in range(len(lats)):
		print(" Lat: " + str(lats[latidx]))
		for lonidx in range(len(lons)):
			min_lat = np.nanmax([-90.0,lats[latidx] - 6.0])
			max_lat = np.nanmin([90.0,lats[latidx] + 6.0])
			min_lon = lons[lonidx] - 6.0
			max_lon = lons[lonidx] + 6.0
			if((min_lon >= 0.0) & (max_lon <= 360.0)):
				temp_id = str_id[(str_lat >= min_lat) & (str_lat <= max_lat) & (str_lon >= min_lon) & (str_lon <= max_lon)]
				temp_lon = str_lon[(str_lat >= min_lat) & (str_lat <= max_lat) & (str_lon >= min_lon) & (str_lon <= max_lon)]
				temp_lat = str_lat[(str_lat >= min_lat) & (str_lat <= max_lat) & (str_lon >= min_lon) & (str_lon <= max_lon)]
				areaidxs = (grid.y >=  min_lat) & (grid.y <= max_lat) & (grid.x  >= min_lon) & (grid.x <= max_lon)
			elif(min_lon < 0.0):
				temp_id = str_id[(str_lat >= min_lat) & (str_lat <= max_lat) &  (str_lon <= max_lon) | (str_lat >= min_lat) & (str_lat <= max_lat) &  (str_lon >= min_lon  + 360.0)]
				areaidxs = (grid.y >=  min_lat) & (grid.y <= max_lat)  & (grid.x  <= max_lon) | (grid.y >=  min_lat) & (grid.y <= max_lat) & (grid.x >= min_lon + 360.0) 
			elif(max_lon > 360.0):
				temp_id = str_id[(str_lat >= min_lat) & (str_lat <= max_lat) & (str_lon >= min_lon) | (str_lat >= min_lat) & (str_lat <= max_lat) & (str_lon <= max_lon - 360.0)]
				areaidxs = (grid.y >=  min_lat) & (grid.y <= max_lat)  & (grid.x  >= min_lon) | (grid.y >=  min_lat) & (grid.y <= max_lat) & (grid.x <= max_lon - 360.0)
			else:
				print("I should not be here")
			diff_lon = 12.0
			area = np.abs((max_lat - min_lat)*diff_lon)*np.cos((min_lat+max_lat)/2.*np.pi/180.)*111111**2 #(np.nansum(np.abs(grid.dy[areaidxs]*grid.dx[areaidxs]))

			mean_storms[latidx,lonidx] = len(temp_id)/(area*len(dt_array))*10**12
			mean_tracks[latidx,lonidx] = len(np.unique(temp_id))/(area*len(dt_array))*10**12

	plt.figure() 
	overlays = [fig.map_overlay_contour(mean_storms[::,::]*100,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)] 
	fig.map(mean_storms[::,::]*100,grid,m=worldShift,overlays=overlays,scale=scale_density,title="Storm Density",cmap="RdBu_r",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1) 
	plt.savefig("StormDensity_IMILAST_" + datachar + ".pdf")                                                                                                                                                                      
