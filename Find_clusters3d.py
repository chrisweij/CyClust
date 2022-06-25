#!/usr/bin/env python
# -*- encoding: utf-8

from scipy.ndimage.filters import gaussian_filter
#from geopy.distance import great_circle as great_circle_old

from datetime import datetime as dt, timedelta as td
import numpy as np
import sparse
from numpy import loadtxt

#exec(open("projection.py").read())
#from dynlib import sphere, cm
import time
from numpy import loadtxt
from Cluster_functions import calc_Rossby_radius, compare_trks_np, find_cluster

from timeit import default_timer as timer

#########################
# Thresholds
#########################

#1. Distance criterium
distthresh = 1.0 #1000.0

#2. Time criterium
timthresh = 30.0

#3. Length criterium 
lngthresh = 1.5 #1.5 #2.0 #calc_Rossby_radius(lat=45)*2.0 # 1000.0

#New set of experiments
timthreshs = np.arange(0.25,2.6,0.25)*24.0
lngthreshs = np.arange(0.6,2.21,0.2)
distthreshs = np.arange(0.5,1.51,0.1)

timmeth = "absolute" #"median" 
Rossby_45 = calc_Rossby_radius(lat=45)

str_result = "Results_DJF_ERA5_" 
outdir = "/home/WUR/weije043/Clusters_Sensitivity/"

#########################
# Load storm tracks
#########################
#Storm tracks file
st_file = "Selected_tracks_2011_2012"

nrskip = 0
str_id   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[0],dtype=int)
str_nr   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[1],dtype=int)
str_date = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[2],dtype=int)
str_lat  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[4])
str_lon  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[3])
str_result = "Results_DJF_NH_2011_2012_"

#Determine datetime array for the tracks
str_dt = []
str_hour = np.zeros(len(str_date))
str_day = np.zeros(len(str_date))
str_month = np.zeros(len(str_date))
str_year = np.zeros(len(str_date))
for idx in range(len(str_date)):
	year = int(str(str_date[idx])[:4])
	month = int(str(str_date[idx])[4:6])
	day   = int(str(str_date[idx])[6:8])
	hour   = int(str(str_date[idx])[8:10])
	str_hour[idx] = hour
	str_day[idx] = day
	str_month[idx] = month
	str_year[idx] = year
	str_dt.append(dt(year,month,day,hour))

#Convert to an array
str_dt          = np.array(str_dt)
str_connected   = np.zeros(str_dt.shape)
str_id = str_id - np.nanmin(str_id) + 1

nrstorms = len(np.unique(str_id))

str_connected   = np.zeros(str_dt.shape)

nrstorms = np.nanmax(str_id)

#Check which year, month, hemisphere belongs storms to
start = time.time()
wintstrms = np.zeros(np.nanmax(str_id))
yrstorms = np.zeros(np.nanmax(str_id))
mnstorms = np.zeros(np.nanmax(str_id))
hemstorms = np.full(np.nanmax(str_id),"Undefined")
firstdt = []
lastdt = []
for strid in range(nrstorms):
	dt_temp = str_dt[str_id == strid + 1]
	lat_temp = str_lat[str_id == strid + 1]

	#Check which winter it belongs to
	tmpyear = dt_temp[0].year
	tmpmonth = dt_temp[0].month
	yrstorms[strid] = tmpyear
	mnstorms[strid] = tmpmonth

	if(tmpmonth < 11):
		tmpyear = tmpyear - 1
	wintstrms[strid] = tmpyear

	#Save the first and last dt
	firstdt.append(dt_temp[0])
	lastdt.append(dt_temp[-1])

	#Check if the storm is in the NH or SH
	if(np.nanmean(lat_temp) > 0):
		hemstorms[strid] = "NH"
	elif(np.nanmean(lat_temp) < 0):
		hemstorms[strid] = "SH"

end = time.time()
firstdt = np.array(firstdt)
lastdt = np.array(lastdt)
print(start-end)

#Months of storm, relative to beginning of 1979
mnstorms_rel = (yrstorms - 1979)*12.0 + mnstorms
refdt = dt(1979,1,1,0,0)
diffs = [(x - refdt).total_seconds()/3600 for x in str_dt]                                                                                                                                      

# START CALCULATION OF CLUSTERS
print("---------------------------------------------")
print("Start checking for:                          ")
print("Distance threshold = " + str(distthresh))
print("Time threshold = " + str(timthresh))
print("Length threshold = " + str(lngthresh))
print("---------------------------------------------")

######################################################
# Find connected and clustered storms
#######################################################
connTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])

maxdists = []
maxdistsown = []
angles = []
anglesClust = []
clusterTypes = []
clusterTypes2 = []
angleTypes = []


#Convert timthresh to td object 
timthresh_dt = td(hours=timthresh)

starttime = timer()
for strm1 in range(nrstorms): #range(6500,7000): # 
	print("Strm1 :" + str(strm1 + 1))
	#minstidx = np.max([0,strm1 - 250])
	#maxstidx =  np.min([7777,strm1 + 250])
	
	selidxs1 = np.where(str_id == strm1 + 1)
	lats1 = str_lat[selidxs1]	
	lons1 = str_lon[selidxs1]
	times1 = str_dt[selidxs1]
    
    #Only check if the storm is in the current month, or one after or before it, in the same hemisphere. 
	#diffmon = mnstorms_rel[strm1] - mnstorms_rel
	diffdt1  = firstdt - lastdt[strm1]
	diffdt2  = firstdt[strm1] - lastdt

	#strm2idxs = np.where( (np.abs(diffmon) <= 1) & (hemstorms == hemstorms[strm1]))[0]
	strm2idxs = np.where((np.arange(nrstorms) >= strm1) & ((diffdt1 <= timthresh_dt) & (diffdt2 <= timthresh_dt)) & (hemstorms == hemstorms[strm1]))[0]

	print("Nr strm2: " + str(len(strm2idxs)))
	#print(strm2idxs)

	for strm2 in strm2idxs: #range(minstidx,maxstidx)
				#print("Strm1 :" + str(strm1 + 1) + " Strm2: " + str(strm2 + 1))

				selidxs2 = np.where(str_id == strm2 + 1)
				lats2 = str_lat[selidxs2]
				lons2 = str_lon[selidxs2] 
				times2 = str_dt[selidxs2]

				if(timmeth == "median"): 
					dists, timdiffs, = compare_trks_median(lons2,lats2,times2,lons1,lats1,times1,medians)
				elif(timmeth == "absolute"):
					dists, timdiffs, = compare_trks_np(lons2,lats2,times2,lons1,lats1,times1)

				#Calculate distance over which storms are connected
				#First select just the lons, lats times over which a particular storm is connected
				pntselect = np.nanmax((np.abs(timdiffs) <= timthresh) & (dists <= distthresh),axis=0) #*Rossby_45
				test1 = np.nansum(pntselect) 
				#Do the same for the other track
				pntselect2 = np.nanmax((np.abs(timdiffs) <= timthresh) & (dists <= distthresh),axis=1) #*Rossby_45
				test2 = np.nansum(pntselect2)

				#If both are connected over at least two points, check the maximum distance
				if((test1 >=2) & (test2 >= 2)):
					#Just select the points which are connected and calculate distances between the points for both tracks
					owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
					owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
					avelat = np.nanmean(np.append(lats1[pntselect],lats2[pntselect2]))
					#corrfac = np.abs(calc_Rossby_radius(lat=avelat)/calc_Rossby_radius(lat=45))
					maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
					maxdists.append(maxdist)
					if(strm1 == strm2):
						maxdistsown.append(maxdist)

					if(maxdist >= lngthresh): #*Rossby_45*corrfac
						print("Strm2: " + str(strm2 + 1) + " Max dist: " + str(maxdist) + " Trck1: " + str(np.nanmax(owndists)) + " Trck2: " + str(np.nanmax(owndists2)))
						connTracks[strm1,strm2] = 1
						connTracks[strm2,strm1] = 1
						str_contemp1 = str_connected[selidxs1]
						str_contemp1[pntselect] = 1.0 
						str_contemp2 = str_connected[selidxs2]
						str_contemp2[pntselect2] = 1.0

						#Save connected points
						str_connected[selidxs1] = str_contemp1
						str_connected[selidxs2] = str_contemp2
		
endtime = timer()
print(endtime - starttime) # Time in seconds, e.g. 5.38091952400282
timing = endtime -starttime
np.fill_diagonal(connTracks,0)

#Find clusters
clusters = []
maxlength = 1

for stridx in range(np.nanmax(str_id)):
    print(stridx)
    clusttemp = find_cluster([stridx + 1],connTracks) 
    if(len(clusttemp) > maxlength):
        maxlength = len(clusttemp)
    clusters.append(clusttemp)

#Delete duplicates and sort on the first number in cluster:
unique_clusters = [list(x) for x in set(tuple(x) for x in clusters)]
#from operator import itemgetter
sorted_clusters =  sorted(unique_clusters)

######################################################
# Statistics --> TODO: Move to different file
######################################################

#PDF with length of clusters
lengthclust = np.zeros(maxlength)
lengths     = []

#Clusters per winter 
winters = np.arange(1979,2016)
nrclst_wint = np.zeros(len(winters))
nrclst_wintNH = np.zeros(len(winters))
nrstrm_wint = np.zeros(len(winters))
nrstrmclst_wint = np.zeros(len(winters))
nrstrm_wintNH = np.zeros(len(winters))
nrstrmclst_wintNH = np.zeros(len(winters))
nrdays_wint = np.zeros(len(winters))

test = 0
for clustidx in range(len(sorted_clusters)):
    clusttemp = sorted_clusters[clustidx]

    lengths.append(len(clusttemp))
    lengthclust[len(clusttemp)-1] += 1

    #Check which winter it belongs to
    tmpyear = str_dt[str_id == clusttemp[0]][0].year
    tmpmonth = str_dt[str_id == clusttemp[0]][0].month
    if(tmpmonth < 11):
        tmpyear = tmpyear - 1

    nrstrm_wint[winters == tmpyear] += len(clusttemp)
    if(len(clusttemp) > 1):
        nrclst_wint[winters == tmpyear] += 1
        nrstrmclst_wint[winters == tmpyear] += len(clusttemp)
        if(np.nanmean(str_lat[str_id == clusttemp[0]]) > 0):
            nrclst_wintNH[winters == tmpyear] += 1
            nrstrmclst_wintNH[winters == tmpyear] += len(clusttemp)

######################################################
# Save results
######################################################
formatter =  "{:1.1f}"
outfile = outdir + str_result + formatter.format(distthresh) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthresh)
np.savez(outfile, sorted_clusters=sorted_clusters, lengths = lengths, lengthclust= lengthclust, winters=winters,nrclst_wint = nrclst_wint, nrstrm_wint = nrstrm_wint, nrstrmclst_wint = nrstrmclst_wint,maxdists=np.array(maxdists),str_connected = str_connected)

