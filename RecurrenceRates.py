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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import loadtxt
import dynlib.figures as fig
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
conf.plotpath = "/home/cwe022/dynlib/examples/pinto_plots/"

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
								#difflon = np.nanmin([np.abs(temp_lon[tridx] -lons[lonidx]),np.abs((temp_lon[tridx] -lons[lonidx])%360.0)]) 
								#if(difflon*np.cos(lats[latidx]*np.pi/180.0) < 5.0):
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

if(calcDensity):
	mean_storms = np.nanmean(storms,axis=0)
	mean_tracks = np.nanmean(tracks,axis=0)
	mean_genesis = np.nanmean(genesis,axis=0)
	mean_lysis = np.nanmean(lysis,axis=0)
	mean_mindpdt = np.nanmean(storms_mindpdt,axis=0)
	mean_minpres = np.nanmean(storms_minpres,axis=0)
	outfile ="/Data/gfi/spengler/cwe022/Density_" + datachar + "_" + distchar + ".npz"
	#np.savez(outfile, storms=storms, tracks=tracks,mean_storms=mean_storms, #genesis= genesis,lysis=lysis,
	#mean_tracks=mean_tracks,mean_genesis=mean_genesis,mean_lysis=mean_lysis, mean_mindpdt= mean_mindpdt, mean_minpres=mean_minpres)
	np.savez(outfile, mean_storms=mean_storms, mean_tracks=mean_tracks,mean_genesis=mean_genesis,mean_lysis=mean_lysis)
else:
	#outfile ="/Data/gfi/spengler/cwe022/Densisty.npz"
	outfile ="/Data/gfi/spengler/cwe022/Density_" + datachar + "_" + distchar + ".npz"
	#outfile ="/Data/gfi/spengler/cwe022/Density_" + datachar + ".npz"
	Results = np.load(outfile)
	mean_storms = Results['mean_storms']
	mean_tracks = Results['mean_tracks']
	mean_genesis = Results['mean_genesis']
	mean_lysis = Results['mean_lysis']
	storms = Results['storms']
	tracks = Results['tracks']

scale_clust = np.arange(1,3.5,0.4)
scale_nonclust = np.arange(1,6.5,0.5)
scale_density = np.arange(3,25.1,2)
scale_tracks  = np.arange(2,12.1,1)
scale_genesis = np.arange(0.1,2.2,0.2)

from mpl_toolkits.basemap import Basemap
def worldShift():
	''' World map, using the Robin projection

	Returns
	-------
	Basemap
		map projection instance
	'''

	return Basemap(projection='robin',lon_0=70,resolution='c', area_thresh=50000)

worldShift.aspect = 2.0

#Construct grid
ncmask, landmask, maskgrid = dynlib.metio.metopen("ei.ans.land-sea",q="lsm")
import copy
grid = copy.copy(maskgrid)
grid.x = maskgrid.x[::3,::3] #+ 180.0#[0:60,::]
grid.y = maskgrid.y[::3,::3] #[0:60,::]
grid.dx = maskgrid.dx[::3,::3] #[0:60,::]
grid.dy = maskgrid.dy[::3,::3] #[0:60,::]
grid.oro = maskgrid.oro[::3,::3] #[0:60,::]
grid.nx = 240
grid.ny = 121 #60
conf.gridsize = (121,240) #(60,240)

if(plotDensities):
	if(distchar != "Rossby"):
		mul_fac = (500/dist_thresh)**2.0*4.0/np.pi*100.0

	#Loop over lons and lats
	if(distchar =="Rossby"):
		mul_fac = np.zeros(conf.gridsize)
		for latidx in range(len(lats)):
			lattemp = np.abs(lats[latidx])
			Rossby_temp = calc_Rossby_radius(lat=lattemp)
			mul_fac[latidx,:] = (500/Rossby_temp)**2.0*4.0/np.pi*100.0
			

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_tracks*4*30,  grid,title=None,cb_label='The nr. of storms per month', scale=[10], linewidths=0.75)]
	fig.map(mean_tracks*4*30,grid,overlays=overlays,scale=scale_tracks,cmap="PuBuGn",extend="both") #np.arange(2,12.1,1)
	plt.savefig("Mean_monthlystorms_" + datachar + "_" + distchar + ".pdf") #[::-1,::]

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_storms*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)]
	fig.map(mean_storms*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_density,title="Storm Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("StormDensity_IMILAST_" + datachar + "_" + distchar + ".pdf")

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_tracks*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
	fig.map(mean_tracks*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_tracks,title="Track Density", cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("TracDensity_IMILAST_" + datachar + "_" + distchar + ".pdf")

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
	fig.map(mean_genesis*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("Genesis_IMILAST_" + datachar + "_" + distchar + ".pdf")

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_lysis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
	fig.map(mean_lysis*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Lysis Density", cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("Lysis_IMILAST_" + datachar + "_" + distchar + ".pdf")

	if(calcDensity):
		plt.figure()
		overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
		fig.map(mean_mindpdt*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
		plt.savefig("MinDpdt_IMILAST_" + datachar + "_" + distchar + ".pdf")

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
		fig.map(mean_minpres*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
		plt.savefig("MinPres_IMILAST_" + datachar + "_" + distchar +".pdf")

	## seasonal differences ##
	seasons = ["DJF","MAM","JJA","SON"]

	for season in seasons:
		months = np.array([x.month for x in dt_array])
		if(season == "DJF"):
			selidxs = (months < 3) | (months >= 12)
		elif(season == "MAM"):
			selidxs = (months < 6) | (months >= 3)
		elif(season == "JJA"):
			selidxs = (months < 9) | (months >= 6)
		elif(season == "SON"):
			selidxs = (months < 12) | (months >= 8)

		mean_storms = np.nanmean(storms[selidxs,::],axis=0)
		mean_tracks = np.nanmean(tracks[selidxs,::],axis=0)
		if(calcDensity):
			mean_genesis = np.nanmean(genesis[selidxs,::],axis=0)
			mean_lysis = np.nanmean(lysis[selidxs,::],axis=0)
			mean_mindpdt = np.nanmean(storms_mindpdt,axis=0)
			mean_minpres = np.nanmean(storms_minpres,axis=0)

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_tracks*mul_fac,  grid,title=None,cb_label='The nr. of storms per month', scale=[10], linewidths=0.75)]
		fig.map(mean_tracks*mul_fac,grid,overlays=overlays,scale=scale_tracks,cmap="PuBuGn",extend="both") #np.arange(2,12.1,1)
		plt.savefig("Mean_monthlystorms_" + datachar + "_" + season + "_" + distchar +".pdf")

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_storms*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
		fig.map(mean_storms*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_density,title="Storm Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
		plt.savefig("StormDensity_IMILAST_" + datachar + "_" + season + "_" + distchar +".pdf")

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_storms*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
		fig.map(mean_storms*mul_fac,grid,overlays=overlays,scale=[2,5,10,15,20,25,35,50,70,100],cb_tickspacing="uniform",title="Storm Density",cmap="terrain",cb_label='% per 1000 km$^2$',extend="max", m=n_extratropics) #np.arange(2,12.1,1)
		plt.savefig("StormDensity_IMILAST_" + datachar + "_" + season + "_" + distchar +"_NH.pdf")

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_storms*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
		fig.map(mean_storms*mul_fac,grid,overlays=overlays,scale=[2,5,10,15,20,25,35,50,70,100],cb_tickspacing="uniform",title="Storm Density",cmap="terrain",cb_label='% per 1000 km$^2$',extend="max", m=s_extratropics) #np.arange(2,12.1,1)
		plt.savefig("StormDensity_IMILAST_" + datachar + "_" + season + "_" + distchar +"_SH.pdf")

		plt.figure()
		overlays = [fig.map_overlay_contour(mean_tracks*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
		fig.map(mean_tracks*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_tracks,title="Track Density", cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
		plt.savefig("TracDensity_IMILAST_" + datachar + "_" + season + "_" + distchar +".pdf")
		if(calcDensity):
			plt.figure()
			overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
			fig.map(mean_genesis*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
			plt.savefig("Genesis_IMILAST_" + datachar + "_" + season + "_" + distchar + ".pdf")

			plt.figure()
			overlays = [fig.map_overlay_contour(mean_lysis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
			fig.map(mean_lysis*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Lysis Density", cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
			plt.savefig("Lysis_IMILAST_" + datachar + "_" + season + "_" + distchar + ".pdf")

			plt.figure()
			overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
			fig.map(mean_mindpdt*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
			plt.savefig("MinDpdt_IMILAST_" + datachar +  "_" + season + "_" + distchar + ".pdf")

			plt.figure()
			overlays = [fig.map_overlay_contour(mean_genesis*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[.5], linewidths=0.75)]
			fig.map(mean_minpres*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_genesis,title="Genesis Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
			plt.savefig("MinPres_IMILAST_" + datachar + "_" + season + "_" + distchar + ".pdf")



########################################################
# 2. Determine clustering storm density
# according to Priestley et. al. 2017 and Maillier 2006
########################################################
def running_mean(x,w):
	x_ = np.insert(x, 0, 0)
	sliding_average = x_[:w].sum() / w + np.cumsum(x_[w:] - x_[:-w]) / w
	return sliding_average


#Run
runningmean_temp = np.apply_along_axis(running_mean,0,tracks,w=4*7 + 1)*(4*7+1)
runstats_7days = np.zeros([7,grid.x.shape[0],grid.x.shape[1]])
runstats_7days_DJF = np.zeros([7,grid.x.shape[0],grid.x.shape[1]])
tidxs = (months[14:-14] <= 2) | (months[14:-14] == 12)
for nrstrm in range(7):
	runstats_7days[nrstrm,::] = np.nanmean((runningmean_temp>=nrstrm),axis=0)
	runstats_7days_DJF[nrstrm,::] = np.nanmean((runningmean_temp[tidxs,::]>=nrstrm),axis=0)
dispersion_7days = np.nanvar(runningmean_temp,axis=0)/np.nanmean(runningmean_temp,axis=0) - 1


plt.figure()
fig.map(runstats_7days[4,::]*100.0,grid,cmap="PuBuGn",title="Fraction of 7 day running mean >= 4 storms",extend="both") #np.arange(2,12.1,1)
plt.savefig("7dayrunmean_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(runstats_7days_DJF[4,::]*100.0,grid,cmap="PuBuGn",title="Fraction of 7 day running mean >= 4 storms",extend="both") #np.arange(2,12.1,1)
plt.savefig("7dayrunmean_" + datachar + "_" + distchar + "_DJF.pdf")

plt.figure()
fig.map(dispersion_7days,grid,cmap="PuBuGn",title="7 day dispersion statistic",extend="both") #np.arange(2,12.1,1)
plt.savefig("Dispersion_7days_" + datachar + "_" + distchar + ".pdf")

runningmean_temp = np.apply_along_axis(running_mean,0,tracks,w=4*30 + 1)*(4*30 + 1)
dispersion_30days = np.nanvar(runningmean_temp,axis=0)/np.nanmean(runningmean_temp,axis=0) - 1

plt.figure()
scale_dispersion = [-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9]
fig.map(dispersion_30days,grid,cmap="PuBuGn",title="30 day dispersion statistic",scale=scale_dispersion,extend="both") #np.arange(2,12.1,1)
plt.savefig("Dispersion_30days_" + datachar + "_" + distchar + ".pdf")

#Different season
plt.figure()
tidxs = (months[60:-60] <= 2) | (months[60:-60] == 12)
dispersion_30days_DJF = np.nanvar(runningmean_temp[tidxs,::],axis=0)/np.nanmean(runningmean_temp[tidxs,::],axis=0) - 1 
fig.map(dispersion_30days_DJF,grid,cmap="PuBuGn",title="30 day dispersion statistic",scale=scale_dispersion,extend="both") #np.arange(2,12.1,1)
plt.savefig("Dispersion_30days_" + datachar + "_" + distchar + "_DJF.pdf")

##########################################
# Median and quantiles of recurrence time
##########################################
quantiles = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
recur_quantiles = np.full((7,len(lats),len(lons)),np.nan)
recur_quantiles_DJF = np.full((7,len(lats),len(lons)),np.nan)
tidxs = (months <= 2) | (months == 12)
for latidx in range(len(lats)):
	for lonidx in range(len(lons)):
		#Differences between times when there is a new storm at certain lat and lon
		onesidxs = np.where(tracks[:,latidx,lonidx] >= 1)[0]
		onesidxs_DJF = np.where(tracks[tidxs,latidx,lonidx] >= 1)[0]
		difftim  = (onesidxs[1:] - onesidxs[:-1])*0.25
		difftim_DJF  = (onesidxs_DJF[1:] - onesidxs_DJF[:-1])*0.25
		difftim_DJF = difftim_DJF[difftim_DJF <= 93] #Exclude the storm difference between different seasons

		#The above does not include the 
		multiple = np.sum((tracks[:,latidx,lonidx]-1)[np.where(tracks[:,latidx,lonidx] > 1)[0]]) #Nr. of times with dt=0
		if(multiple > 0):
			difftim = np.append(difftim,np.zeros(np.int(multiple)))

		multiple_DJF = np.sum((tracks[tidxs,latidx,lonidx]-1)[np.where(tracks[tidxs,latidx,lonidx] > 1)[0]]) #Nr. of times with dt=0
		if(multiple_DJF > 0):
			difftim_DJF = np.append(difftim_DJF,np.zeros(np.int(multiple_DJF)))

		#Get quantiles
		if(len(difftim) > 0):
			recur_quantiles[:,latidx,lonidx] = np.quantile(difftim,quantiles)
		if(len(difftim_DJF) > 0):
			recur_quantiles_DJF[:,latidx,lonidx] = np.quantile(difftim_DJF,quantiles)

strm_mask = (mean_tracks < 0.01)
#Median
plt.figure()
fig.map(recur_quantiles[3,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="Median recurrence rate",extend="both", scale=np.arange(1,3.6,0.25)) #np.arange(2,12.1,1)
plt.savefig("Median_distance_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(recur_quantiles_DJF[3,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="Median recurrence rate",extend="both", scale=np.arange(1,3.6,0.25)) #np.arange(2,12.1,1)
plt.savefig("Median_distance_" + datachar + "_" + distchar + "_DJF.pdf")

#Median
plt.figure()
fig.map(recur_quantiles[1,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="10% quantile recurrence rate",extend="both", scale=np.arange(0,2,0.25)) #np.arange(2,12.1,1)
plt.savefig("Quantile_10_distance_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(recur_quantiles[2,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="25% quantile recurrence rate",extend="both", scale=np.arange(0.25,2.6,0.25)) #np.arange(2,12.1,1)
plt.savefig("Quantile_25_distance_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(recur_quantiles[4,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="75% quantile recurrence rate",extend="both", scale=np.arange(1.5,5.1,0.5)) #np.arange(2,12.1,1)
plt.savefig("Quantile_75_distance_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(recur_quantiles[5,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="90% quantile recurrence rate",extend="both", scale=np.arange(4,10,1)) #np.arange(2,12.1,1)
plt.savefig("Quantile_90_distance_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(recur_quantiles[6,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="95% quantile recurrence rate",extend="both", scale=np.arange(6,12,1.0)) #np.arange(2,12.1,1)
plt.savefig("Quantile_95_distance_" + datachar + "_" + distchar + ".pdf")


