#!/usr/bin/env python
# -*- encoding: utf-8

#from geopy.distance import great_circle as great_circle_old
from datetime import datetime as dt, timedelta as td
import numpy as np
from numpy import loadtxt

import sparse
from scipy.sparse import dok_matrix
import time
from timeit import default_timer as timer

#########################
# Options
#########################
Options = {
#1. Distance criterium
"distthresh" : 1.0, #1000.0,

#2. Time criterium
"timthresh" : 36.0,

#3. Length/Time criterium 
"lngthresh" : 1.5, #1.5 #2.0 #calc_Rossby_radius(lat=45)*2.0 # 1000.0
"timlngthresh" : 6,
"minPairs" : 24,

# Options
"timmeth" : "absolute",
"timspace" : False,  #Use a combined Time-Space criterium 
"connTime" : False,  #Connect on lengths
"connSpaceTime" : False,
"connSpaceOrTime" : True,
"connPairs" : False,
"excludeLong" : False, #Exclude other type
"distmeth" : "AlongTracksDirect", #"MaxDist" #"AlongTracks" "AlongTrackDirect"
"frameworkSparse" : False, #True

# Output directory
"outdir" : "Results/", 

# Minimum of nr. of storms in a "family"
"minstorms" : 3,
"outdir" : "Clusters_output/",
"str_result" : "EI_2011_2012"
}

#########################
# Load storm tracks --> TO DO: Move to function
#########################
#Storm tracks file
st_file = "Selected_tracks_2011_2012"
st_file = "Selected_tracks_1979to2018_0101to1231_ei_Globe_Leonidas_with_stationary_all"

nrskip = 1
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

#########################
# Define result arrays
#########################

if(Options["frameworkSparse"] == False):
    connTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    angleTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    drTracks  = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    dtTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
else:
    connTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    angleTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    drTracks  = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    dtTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))


from Cluster_functions import *
#calc_Rossby_radius, compare_trks_np, find_cluster, find_cluster_type_dokm, unnest, get_indices_sparse, find_cluster_type, find_cluster_type3, connect_cyclones

#########################
# Get indices of storms
#########################
uniq_ids = np.unique(str_id)
ids_storms = get_indices_sparse(str_id)

#########################
# Preprocess storm tracks
#########################

#Check which year, month, hemisphere belongs storms to
start = time.time()

wintstrms = np.zeros(nrstorms)
yrstorms = np.zeros(nrstorms)
mnstorms = np.zeros(nrstorms)
hemstorms = np.full(nrstorms,"Undefined")
firstdt = []
lastdt = []

for strid in range(nrstorms):    
	dt_temp = str_dt[ids_storms[uniq_ids[strid]]]
	lat_temp = str_lat[ids_storms[uniq_ids[strid]]]

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
print("Distance threshold = " + str(Options["distthresh"]))
print("Time threshold = " + str(Options["timthresh"]))
print("Length threshold = " + str(Options["lngthresh"]))
print("---------------------------------------------")

#Convert timthresh to td object 
timthresh_dt = td(hours=Options["timthresh"])

######################################################
# Find connected and clustered storms
#######################################################

starttime = timer()
for strm1 in range(nrstorms): #range(nrstorms): #[1]: # 
    if(strm1%100 == 0):
        print(strm1)
    #range(6500,7000): # 
    #print("Strm1 :" + str(uniq_ids[strm1]))
    selidxs1 = ids_storms[uniq_ids[strm1]] #np.where(str_id == uniq_ids[strm1])
    
    lats1 = str_lat[selidxs1]	
    #print(lats1)
    lons1 = str_lon[selidxs1]
    times1 = str_dt[selidxs1]
    
    #Only check if the storm is in the current month, or one after or before it, in the same hemisphere. 
    #diffmon = mnstorms_rel[strm1] - mnstorms_rel
    diffdt1  = firstdt - lastdt[strm1]
    diffdt2  = firstdt[strm1] - lastdt
    
    strm2idxs = np.where((np.arange(nrstorms) > strm1) & ((diffdt1 <= timthresh_dt) & (diffdt2 <= timthresh_dt)) & (hemstorms == hemstorms[strm1]))[0]
    #print("Nr strm2: " + str(len(strm2idxs)))
    
    for strm2 in strm2idxs: #[5]: #strm2idxs: #range(minstidx,maxstidx)
        #print("Strm1 :" + str(strm1 + 1) + " Strm2: " + str(strm2 + 1))

        selidxs2 = ids_storms[uniq_ids[strm2]] #np.where(str_id == uniq_ids[strm2])
        lats2 = str_lat[selidxs2]
        lons2 = str_lon[selidxs2] 
        times2 = str_dt[selidxs2]
        #print(lats2)

        conn, angle, dt, dr, strConn1, strConn2  =\
                        connect_cyclones(lons1,lats1,times1,
                                                lons2,lats2,times2,
                                                Options)
        
        connTracks[strm2,strm1] = conn
        connTracks[strm1,strm2] = conn
        angleTracks[strm1,strm2] = angle
        dtTracks[strm1,strm2] = dt
        drTracks[strm1,strm2] = dr
        
        str_connected[selidxs1] = strConn1
        str_connected[selidxs2] = strConn2
                     
endtime = timer()
print(endtime - starttime) # Time in seconds, e.g. 5.38091952400282
timing = endtime -starttime
#np.fill_diagonal(connTracks,0)

if(Options["frameworkSparse"] == True):
    connTracks = connTracks.tocsr()

########################
# Step 2 Find clusters
########################
clusters = []
maxlength = 1

for stridx in range(nrstorms):
    #print(stridx)
    if(Options["frameworkSparse"] == True):
        clusttemp = find_cluster_type_dokm([stridx],connTracks)        
    else:
        clusttemp, connTypes, clusterType = find_cluster_type([stridx],connTracks) 
    #clusttemp2, connTypes2, anglesClust2, clusterType2, angleType = find_cluster_type2([stridx],connTracks, angleTracks)
    
    #if(clusterType != clusterType2):
    #    print(connTypes)
    #    print(clusterType)
    #    print(connTypes2)
    #    print(clusterType2)
    
    if(len(clusttemp) > maxlength):
        maxlength = len(clusttemp)
    
    clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
    clusters.append(clusttemp)
    #clusterTypes.append(clusterType)
    #clusterTypes2.append(clusterType2)
    #angleTypes.append(angleType)
    
#Delete duplicates and sort on the first number in clusters:
unique_clusters = [list(x) for x in set(tuple(x) for x in clusters)]

#from operator import itemgetter
sorted_clusters =  sorted(unique_clusters)
print(timer() - starttime) # Time in seconds, e.g. 5.38091952400282

############################
# Step 3 Suborder clusters
############################
sorted_subclusters_length = []
sorted_subclusters_nolength = []

for cluster in sorted_clusters:
    #print(stridx)
    subclusters_length = []
    subclusters_nolength = []
    
    for stridx in cluster:
        
        #Length clusters
        if(Options["frameworkSparse"] == True):
            clusttemp = find_cluster_type_dokm([stridx - 1],connTracks,contype="Length")
        else:
            clusttemp, connTypes, clusterType = find_cluster_type3([stridx - 1],connTracks,contype="Length")

        clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
        subclusters_length.append(clusttemp)
        
        #Stationary clusters
        if(Options["frameworkSparse"] == True):
            clusttemp = find_cluster_type_dokm([stridx - 1],connTracks,contype="NoLength")
        else:
            clusttemp, connTypes, clusterType = find_cluster_type3([stridx - 1],connTracks,contype="NoLength") 

        clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
        subclusters_nolength.append(clusttemp)
        
    #Delete duplicates and sort on the first number in (sub)clusters:
    unique_subclusters = [list(x) for x in set(tuple(x) for x in subclusters_length)]
    sorted_subclusters_length.append(sorted(unique_subclusters))
    
    #Delete duplicates and sort on the first number in (sub)clusters:
    unique_subclusters = [list(x) for x in set(tuple(x) for x in subclusters_nolength)]
    sorted_subclusters_nolength.append(sorted(unique_subclusters))
    
sorted_clusters_length = sorted(unnest(sorted_subclusters_length))
sorted_clusters_nolength = sorted(unnest(sorted_subclusters_nolength))

######################################################
# Save results
######################################################
formatter =  "{:1.1f}"
outfile = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"])

np.savez(outfile, sorted_clusters=sorted_clusters, lengths = lengths, lengthclust= lengthclust, winters=winters,nrclst_wint = nrclst_wint, nrstrm_wint = nrstrm_wint, nrstrmclst_wint = nrstrmclst_wint,maxdists=np.array(maxdists),str_connected = str_connected)

