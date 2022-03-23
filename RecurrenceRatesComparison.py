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
from dynlib import sphere
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
#execfile("projection.py")

#Add directories to path
conf.datapath.insert(0,"/Data/gfi/spengler/cwe022/EI/slope_avg/")
conf.datapath.insert(0,"//Data/gfi/spengler/cwe022/averages/")
conf.datapath.insert(2,"/Data/gfi/share/Reanalysis/ERA_INTERIM/TOPO_LANDMASK/")
conf.datapath.insert(0,"/Data/gfi/share/Reanalysis/ERA_INTERIM/6HOURLY/TENDENCY2/")
conf.datapath.insert(0,"//Data/gfi/spengler/cwe022/EI/other_avg/press_perctl/")

#Options
datachar = "Leonidas" #"LeonidasAll"
distchar = "Rossby"
#distchar = "250km"
dist_thresh = 250 #500 700
#datachar = "New_namelist"
datachar = "New_namelistAll"
selyear = 2013

lng_thresh = "1.5"
tim_thresh = "36.0"

#Options related to histograms
slope_bins = np.arange(0,18,0.25)

#Plotting path
conf.plotpath = "/home/cwe022/dynlib/examples/pinto_plots/"

#Switches
calcDensity = True
plotDensities = False


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


def calmap(ax, year, data):
    ax.tick_params('x', length=0, labelsize="medium", which='major')
    ax.tick_params('y', length=0, labelsize="x-small", which='major')

    # Month borders
    xticks, labels = [], []
    start = datetime(year,1,1).weekday()
    for month in range(1,13):
        first = dt(year, month, 1)
        last = first + relativedelta(months=1, days=-1)

        y0 = first.weekday()
        y1 = last.weekday()
        x0 = (int(first.strftime("%j"))+start-1)//7
        x1 = (int(last.strftime("%j"))+start-1)//7

        P = [ (x0,   y0), (x0,    7),  (x1,   7),
              (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
              (x0+1,  0), (x0+1,  y0) ]
        xticks.append(x0 +(x1-x0+1)/2)
        labels.append(first.strftime("%b"))
        poly = Polygon(P, edgecolor="black", facecolor="None",
                       linewidth=1, zorder=20, clip_on=False)
        ax.add_artist(poly)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(0.5 + np.arange(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("{}".format(year), weight="semibold")
    
    # Clearing first and last day from the data
    valid = datetime(year, 1, 1).weekday()
    data[:valid,0] = np.nan
    valid = datetime(year, 12, 31).weekday()
    # data[:,x1+1:] = np.nan
    data[valid+1:,x1] = np.nan

    # Showing data
    ax.imshow(data, extent=[0,53,0,7], zorder=10, vmin=-1, vmax=1,
              cmap="RdYlBu", origin="lower", alpha=.75)





###
from dateutil.relativedelta import relativedelta 
from matplotlib.patches import Polygon 
def calmap_winter(ax, year, data):
    ax.tick_params('x', length=0, labelsize="medium", which='major')
    ax.tick_params('y', length=0, labelsize="x-small", which='major')

    # Month borders
    xticks, labels = [], []
    start = dt(year,10,1).weekday()
    for month in range(10,13):
        first = dt(year, month, 1)
        last = first + relativedelta(months=1, days=-1)

        y0 = first.weekday()
        y1 = last.weekday()
        x0 = (int(first.strftime("%j"))+start-1)//7
        x1 = (int(last.strftime("%j"))+start-1)//7

        P = [ (x0,   y0), (x0,    7),  (x1,   7),
              (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
              (x0+1,  0), (x0+1,  y0) ]
        xticks.append(x0 +(x1-x0+1)/2)
        labels.append(first.strftime("%b"))
        poly = Polygon(P, edgecolor="black", facecolor="None",
                       linewidth=1, zorder=20, clip_on=False)
        ax.add_artist(poly)
 
    #start = datetime(year+1,1,1).weekday()
    for month in range(1,4):
        first = dt(year+1, month, 1)
        last = first + relativedelta(months=1, days=-1)

        y0 = first.weekday()
        y1 = last.weekday()
        x0 = (int(first.strftime("%j"))+start-1)//7
        x1 = (int(last.strftime("%j"))+start-1)//7

        P = [ (x0,   y0), (x0,    7),  (x1,   7),
              (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
              (x0+1,  0), (x0+1,  y0) ]
        xticks.append(x0 +(x1-x0+1)/2)
        labels.append(first.strftime("%b"))
        poly = Polygon(P, edgecolor="black", facecolor="None",
                       linewidth=1, zorder=20, clip_on=False)
        ax.add_artist(poly)   
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(0.5 + np.arange(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("{}".format(year), weight="semibold")
    
    # Clearing first and last day from the data
    valid = dt(year, 10, 1).weekday()
    #data[:valid,0] = np.nan
    valid = dt(year+1, 3, 31).weekday()
    # data[:,x1+1:] = np.nan
    #data[valid+1:,x1] = np.nan

    # Showing data
    ax.imshow(data, extent=[0,26,0,7], zorder=10, vmin=-1, vmax=1,
              cmap="RdYlBu", origin="lower", alpha=.75)

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

#Read pressure pctl

nc, pres_pctl, presgrid = dynlib.metio.metopen("yseaspctl_slp.nc", q="msl")

pres_pctl = pres_pctl[:,::3,::3]

#########################
# Load storm tracks
#########################


#Storm tracks file
#st_file = "stormtracks_LTs.txt"
#st_file = "tracks_NH.txt"
if(datachar == "Leonidas"):
	st_file = "/home/cwe022/dynlib/examples/Selected_tracks_1979to2016_0101to1231_ei_Globe_Leonidas_with_stationary"
elif(datachar == "LeonidasAll"):
	st_file = "/home/cwe022/dynlib/examples/Selected_tracks_1979to2016_0101to1231_ei_Globe_Leonidas_with_stationary_all"
elif(datachar == "New_namelist"):
	st_file = "/Data/gfi/spengler/cwe022/TRACKS/Selected_tracks_1979to2016_0101to1231_ei_Globe_New_namelist_with_stationary"
elif(datachar == "New_namelistAll"):
	st_file = "/Data/gfi/spengler/cwe022/TRACKS/Selected_tracks_1979to2016_0101to1231_ei_Globe_New_namelist_with_stationary_all"
st_file = "test_tracks"
#st_file = "Tracks_Pinto_Final"
#st_file = "Selected_tracks_1979to2016_1201to0228_EI_IMILAST"

if(st_file == "tracks_NH.txt"):
	nrskip = 1
	str_id   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[0],dtype=int)
	str_nr   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[1],dtype=int)
	str_date = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[2],dtype=int)
	str_lat  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[3])
	str_lon  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[4])
else:
	nrskip = 0
	str_id   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[0],dtype=int)
	str_nr   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[1],dtype=int)
	str_date = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[2],dtype=int)
	str_lat  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[4])
	str_lon  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[3])

str_pres = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[5],dtype=float)
str_lapl = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[6],dtype=float)
#f = open('/Data/gfi/spengler/cwe022/file_3.txt', 'r')

#datachar = "New_namelistTest"


#Determine datetime array for the tracks
str_dt = []
str_mon = []
str_year = []
for idx in range(len(str_date)):
	year = int(str(str_date[idx])[:4])
	month = int(str(str_date[idx])[4:6])
	day   = int(str(str_date[idx])[6:8])
	hour   = int(str(str_date[idx])[8:10])
	str_mon.append(month)
	str_year.append(year)
	str_dt.append(dt(year,month,day,hour))

#Convert to an array
str_dt          = np.array(str_dt)
str_mon         = np.array(str_mon)
str_year 	= np.array(str_year)

#Select specific season
selidxs = (((str_year == selyear) & (str_mon >= 10)) | ((str_year == selyear + 1) & (str_mon <= 3))) 
str_id = str_id[selidxs]
str_dt = str_dt[selidxs]
str_nr = str_nr[selidxs]
str_mon = str_mon[selidxs]
str_lon = str_lon[selidxs]
str_lat = str_lat[selidxs]
str_pres = str_pres[selidxs]
str_lapl = str_lapl[selidxs]

#Define result arrays
str_reltime     = np.zeros(len(str_lon))
str_reltimestep = np.zeros(len(str_lon))
str_reltimestepAtl = np.zeros(len(str_lon))
str_isclustered = np.zeros(len(str_lon))
str_rot         = np.zeros(len(str_lon))

#Construct array with datetimes
dt_array = []
for yidx in range(selyear,selyear+1):
	
	# To get year (integer input) from the user
	# year = int(input("Enter a year: "))
	if (yidx % 4) == 3:
		leapyear = True
		nr_times = 732 #366*4 #(whole year) 364 (just winter)
	else:
		leapyear = False
		nr_times = 728 #365*4 #(whole year) 360 (just winter)

#	start = dt(yidx, 12, 1, 0) (just winter)
	start = dt(yidx, 10, 1, 0) 

	dt_array_temp = np.array([start + td(hours=i*6) for i in range(nr_times)])
	dt_array.extend(dt_array_temp)


#####################################################
# Load Clustering stats
#####################################################

Results = np.load("/home/cwe022/Clusters_Sensitivity/Results_test1.0_tim_"+ tim_thresh + "_length_" + lng_thresh + ".npz", allow_pickle=True)
#Results_test1.0_tim_36.0_length_1.5
sorted_clusters = Results["sorted_clusters"]
lengthclust = Results["lengthclust"]
lengths = Results["lengths"]
str_connected = Results["str_connected"]
str_connected = str_connected[1272894:][selidxs] #Filter storms from 2011

#Filter clusterd storms
strmidxs = np.unique(str_id)
clststroms = [strm for cluster in sorted_clusters for strm in cluster if len(cluster) > 1 and strm in strmidxs]


#####################################################
# 1. Determine storm density
# i.e. first time a storm hits a particular region
#####################################################
if(calcDensity):
	 

	#Define arrays
	storms = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density
	storms_clust = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density

	tracks = np.zeros((len(dt_array),len(lats),len(lons))) #Track density all storms
	tracks_strong = np.zeros((len(dt_array),len(lats),len(lons))) #Track density strong storms
	tracks_clust = np.zeros((len(dt_array),len(lats),len(lons))) #Track density clustered stroms
	tracks_clust_connect = np.zeros((len(dt_array),len(lats),len(lons))) #Track density clustered stroms
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
	for stridx in strmidxs: #range(1,nr_storms+1): #clust_idxs: #range(1,nr_storms+1):
		print("Storm: " + str(stridx))
		temp_lat = str_lat[str_id == stridx]
		temp_lon = str_lon[str_id == stridx]
		temp_dt  = str_dt[str_id == stridx]
		temp_pres = str_pres[str_id == stridx]
		temp_lapl = str_lapl[str_id == stridx]
		temp_minpres = np.nanmin(temp_pres)
		temp_maxlapl = np.nanmax(str_lapl[str_id == stridx])
		temp_meanlapl = np.nanmean(str_lapl[str_id == stridx])
		temp_dpdt = np.zeros(len(temp_pres))
		temp_dpdt[1:-1] = temp_pres[2:] - temp_pres[:-2]
		#minpresidx = np.nanargmin(temp_pres)
		mindpdtidx = np.nanargmin(temp_dpdt)
		minpresidx = np.nanargmin(temp_pres)
		str_reltime[str_id == stridx] =  np.arange(0,len(temp_lat))*0.25 - minpresidx *0.25
		str_reltimestep[str_id == stridx] =  np.arange(0,len(temp_lat)) - minpresidx
		temp_rot = np.zeros(len(temp_lat))
		temp_connected = str_connected[str_id == stridx]

		#Switch to prevent double counting	
		bool_tracks   = np.full((len(lats),len(lons)),False)
		bool_tracks_strong = np.full((len(lats),len(lons)),False)
		bool_tracks_clust = np.full((len(lats),len(lons)),False)
		bool_tracks_clust_connect = np.full((len(lats),len(lons)),False)

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
									if((temp_pres[tridx] <= pres_pctl[0,latidx,lonidx]/100.0) & (bool_tracks_strong[latidx,lonidx] == False)): 											
										tracks_strong[tidx,latidx,lonidx] += 1
										bool_tracks_strong[latidx,lonidx] = True
									if(stridx in clststroms):
										storms_clust[tidx,latidx,lonidx] += 1
										if(bool_tracks_clust[latidx,lonidx] == False):
											tracks_clust[tidx,latidx,lonidx] += 1	
										if((bool_tracks_clust_connect[latidx,lonidx] == False) & (temp_connected[tridx] == 1) ):
											tracks_clust_connect[tidx,latidx,lonidx] += 1	
											bool_tracks_clust_connect[latidx,lonidx] = True
										bool_tracks_clust[latidx,lonidx] = True
#Another method
calcDensity2 = False
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
	overlays = [fig.map_overlay_contour(mean_storms*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)]
	fig.map(mean_storms*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_density,title="Storm Density",cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("StormDensity_IMILAST_" + datachar + "_" + distchar + ".pdf")

	plt.figure()
	overlays = [fig.map_overlay_contour(mean_tracks*mul_fac,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[10], linewidths=0.75)]
	fig.map(mean_tracks*mul_fac,grid,m=worldShift,overlays=overlays,scale=scale_tracks,title="Track Density", cmap="PuBuGn",cb_label='% per 1000 km$^2$',extend="both") #np.arange(2,12.1,1)
	plt.savefig("TracDensity_IMILAST_" + datachar + "_" + distchar + ".pdf")


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
runningmean_strong = np.apply_along_axis(running_mean,0,tracks_strong,w=4*7 + 1)*(4*7+1)
runningmean_clust_connect = np.apply_along_axis(running_mean,0,tracks_clust_connect,w=4*7 + 1)*(4*7+1)
runningmean_clust = np.apply_along_axis(running_mean,0,tracks_clust,w=4*7 + 1)*(4*7+1)
runstats_7days = np.zeros([7,grid.x.shape[0],grid.x.shape[1]])
#tidxs = (months[14:-14] <= 2) | (months[14:-14] == 12)
for nrstrm in range(7):
	runstats_7days[nrstrm,::] = np.nanmean((runningmean_temp>=nrstrm),axis=0)

clustPriestley_65N = (runningmean_strong[:,17,117] >= 3)
clustPriestley_55N = (runningmean_strong[:,23,117] >= 3)
clustPriestley_45N = (runningmean_strong[:,30,117] >= 3)

#Running means at specific locations
rnmean_65N = np.concatenate([np.zeros(14),runningmean_strong[:,17,117],np.zeros(14)])
rnmean_55N = np.concatenate([np.zeros(14),runningmean_strong[:,23,117],np.zeros(14)])
rnmean_45N = np.concatenate([np.zeros(14),runningmean_strong[:,30,117],np.zeros(14)])

rnmean_65N_clust_connect = np.concatenate([np.zeros(14),runningmean_clust_connect[:,17,117],np.zeros(14)])
rnmean_55N_clust_connect = np.concatenate([np.zeros(14),runningmean_clust_connect[:,23,117],np.zeros(14)])
rnmean_45N_clust_connect = np.concatenate([np.zeros(14),runningmean_clust_connect[:,30,117],np.zeros(14)])

#Daymeans
#from skimage.measure import block_reduce
#rnmean_65N_daymean  = block_reduce(rnmean_65N, block_size=(1,4), func=np.mean, cval=np.mean(rnmean_65N))
rnmean_65N_daymean = np.mean(rnmean_65N[:(len(rnmean_65N)//4)*4].reshape(-1,4), axis=1)  
rnmean_55N_daymean = np.mean(rnmean_55N[:(len(rnmean_55N)//4)*4].reshape(-1,4), axis=1)  
rnmean_45N_daymean = np.mean(rnmean_45N[:(len(rnmean_45N)//4)*4].reshape(-1,4), axis=1)  

dt_array_runmean = dt_array[14:-14]

plt.figure()
#plt.plot(dt_array,rnmean_45N)
plt.bar(dt_array, storms_clust[:,23,117], color = 'b', width = 0.25)
plt.bar(dt_array,rnmean_55N,color='r',width=0.25)
plt.bar(dt_array, storms_clust[:,23,117], color = 'b', width = 0.25)
#plt.plot(dt_array,rnmean_65N)
plt.savefig("Runstats_wint201314.png")

# plot it

fig = plt.figure(figsize=(8, 4))
ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan =5)
ax0.plot(dt_array, rnmean_45N,color='r')
ax0.plot(dt_array, rnmean_45N_clust_connect,color='dodgerblue')
plt.title("Comparison clustering 45 N")
plt.ylabel("Running mean (strong) storms")
ax1 = plt.subplot2grid((6, 1), (5, 0),sharex=ax0)
dts = np.where((tracks[:,30,117]>=1) & (tracks_strong[:,30,117] ==0))[0]
ax1.scatter(np.array(dt_array)[dts], np.ones(len(dts))*1.5, color = 'black',s=4.0)
dts_strong = np.where(tracks_strong[:,30,117]>=1)[0]
ax1.scatter(np.array(dt_array)[dts_strong], np.ones(len(dts_strong))*1.5, color = 'red',s=4.0)
dts_clst = np.where((tracks_clust[:,30,117]>=1) & (tracks_clust_connect[:,30,117]==0))[0]
ax1.scatter(np.array(dt_array)[dts_clst], np.ones(len(dts_clst))*1.2, color = 'black',s=4.0)
dts_clst_connect = np.where((tracks_clust_connect[:,30,117]>=1))[0]
ax1.scatter(np.array(dt_array)[dts_clst_connect], np.ones(len(dts_clst_connect))*1.2, color = 'dodgerblue',s=4.0)
y_axis = ax1.get_yaxis().set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.ylim(1,1.6)
plt.tight_layout()

plt.savefig('Comparison_45N_' + str(selyear) + "-" + str(selyear+1) + '_lngth_'+ lng_thresh + '_tim_'+ tim_thresh +  '.pdf')

# plot it

fig = plt.figure(figsize=(8, 4))
ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan =5)
ax0.plot(dt_array, rnmean_55N,color='r')
ax0.plot(dt_array, rnmean_55N_clust_connect,color='dodgerblue')
plt.title("Comparison clustering 55 N")
plt.ylabel("Running mean (strong) storms")
ax1 = plt.subplot2grid((6, 1), (5, 0),sharex=ax0)
dts = np.where((tracks[:,23,117]>=1) & (tracks_strong[:,23,117] ==0))[0]
ax1.scatter(np.array(dt_array)[dts], np.ones(len(dts))*1.5, color = 'black',s=4.0)
dts_strong = np.where(tracks_strong[:,23,117]>=1)[0]
ax1.scatter(np.array(dt_array)[dts_strong], np.ones(len(dts_strong))*1.5, color = 'red',s=4.0)
dts_clst = np.where((tracks_clust[:,23,117]>=1) & (tracks_clust_connect[:,23,117]==0))[0]
ax1.scatter(np.array(dt_array)[dts_clst], np.ones(len(dts_clst))*1.2, color = 'black',s=4.0)
dts_clst_connect = np.where((tracks_clust_connect[:,23,117]>=1))[0]
ax1.scatter(np.array(dt_array)[dts_clst_connect], np.ones(len(dts_clst_connect))*1.2, color = 'dodgerblue',s=4.0)
y_axis = ax1.get_yaxis().set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.ylim(1,1.6)
plt.tight_layout()

plt.savefig('Comparison_55N_' + str(selyear) + "-" + str(selyear+1) + '_lngth_'+ lng_thresh + '_tim_'+ tim_thresh +  '.pdf')

# plot it

fig = plt.figure(figsize=(8, 4))
ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan =5)
ax0.plot(dt_array, rnmean_65N,color='r')
ax0.plot(dt_array, rnmean_65N_clust_connect,color='dodgerblue')
plt.title("Comparison clustering 65 N")
plt.ylabel("Running mean (strong) storms")
ax1 = plt.subplot2grid((6, 1), (5, 0),sharex=ax0)
dts = np.where((tracks[:,17,117]>=1) & (tracks_strong[:,17,117] ==0))[0]
ax1.scatter(np.array(dt_array)[dts], np.ones(len(dts))*1.5, color = 'black',s=4.0)
dts_strong = np.where(tracks_strong[:,17,117]>=1)[0]
ax1.scatter(np.array(dt_array)[dts_strong], np.ones(len(dts_strong))*1.5, color = 'red',s=4.0)
dts_clst = np.where((tracks_clust[:,17,117]>=1) & (tracks_clust_connect[:,17,117]==0))[0]
ax1.scatter(np.array(dt_array)[dts_clst], np.ones(len(dts_clst))*1.2, color = 'black',s=4.0)
dts_clst_connect = np.where((tracks_clust_connect[:,17,117]>=1))[0]
ax1.scatter(np.array(dt_array)[dts_clst_connect], np.ones(len(dts_clst_connect))*1.2, color = 'dodgerblue',s=4.0)
y_axis = ax1.get_yaxis().set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.ylim(1,1.6)
plt.tight_layout()

plt.savefig('Comparison_65N_' + str(selyear) + "-" + str(selyear+1) + '_lngth_'+ lng_thresh + '_tim_'+ tim_thresh + '.pdf')

"""
fig = plt.figure(figsize=(8,7.5), dpi=100)
year = 2014
n = 3

ax = plt.subplot(n,1,1, xlim=[0,26], ylim=[0,7], frameon=False, aspect=1)
calmap_winter(ax, year, rnmean_45N_daymean.reshape(26,7).T)

ax = plt.subplot(n,1,2, xlim=[0,26], ylim=[0,7], frameon=False, aspect=1)
calmap_winter(ax, year, rnmean_55N_daymean.reshape(26,7).T)

ax = plt.subplot(n,1,3, xlim=[0,26], ylim=[0,7], frameon=False, aspect=1)
calmap_winter(ax, year, rnmean_65N_daymean.reshape(26,7).T)

plt.tight_layout()
plt.savefig("heatmapWinter.pdf", dpi=600)
plt.show()



plt.figure()
fig.map(runstats_7days[4,::]*100.0,grid,cmap="PuBuGn",title="Fraction of 7 day running mean >= 4 storms",extend="both") #np.arange(2,12.1,1)
plt.savefig("7dayrunmean_" + datachar + "_" + distchar + ".pdf")

plt.figure()
fig.map(runstats_7days_DJF[4,::]*100.0,grid,cmap="PuBuGn",title="Fraction of 7 day running mean >= 4 storms",extend="both") #np.arange(2,12.1,1)
plt.savefig("7dayrunmean_" + datachar + "_" + distchar + "_DJF.pdf")
"""


