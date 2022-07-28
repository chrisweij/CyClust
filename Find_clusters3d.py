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
import yaml
from Cluster_functions import read_file, dt_array

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)

#########################
# Load storm tracks 
#########################

#Storm tracks file
str_id, str_nr, str_date, str_lat, str_lon = read_file(Options["st_file"],Options["nrskip"])
str_dt = dt_array(str_date)

#Convert to an array
str_dt          = np.array(str_dt)
str_connected   = np.zeros(str_dt.shape)
str_id = str_id - np.nanmin(str_id) + 1

str_result = "Results_DJF_NH_2011_2012_"

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

hemstorms = np.full(nrstorms,"Undefined")
firstdt = []
lastdt = []

for strid in range(nrstorms):    
	dt_temp = str_dt[ids_storms[uniq_ids[strid]]]
	lat_temp = str_lat[ids_storms[uniq_ids[strid]]]

	#Check which winter it belongs to
	tmpyear = dt_temp[0].year
	tmpmonth = dt_temp[0].month

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
    #print("Strm1 :" + str(uniq_ids[strm1]))
    selidxs1 = ids_storms[uniq_ids[strm1]] #np.where(str_id == uniq_ids[strm1])
    
    lats1 = str_lat[selidxs1]	
    lons1 = str_lon[selidxs1]
    times1 = str_dt[selidxs1]
    
    #Only compare with storms which are close enought im time compared to strm1 
    diffdt1  = firstdt - lastdt[strm1]
    diffdt2  = firstdt[strm1] - lastdt
    
    strm2idxs = np.where((np.arange(nrstorms) > strm1) & ((diffdt1 <= timthresh_dt) & (diffdt2 <= timthresh_dt)) & (hemstorms == hemstorms[strm1]))[0]
    
    for strm2 in strm2idxs: #[5]: #strm2idxs: #range(minstidx,maxstidx)

        selidxs2 = ids_storms[uniq_ids[strm2]] #np.where(str_id == uniq_ids[strm2])
        lats2 = str_lat[selidxs2]
        lons2 = str_lon[selidxs2] 
        times2 = str_dt[selidxs2]

        conn, angle, dt, dr, strConn1, strConn2  =\
            connect_cyclones(lons1,lats1,times1,lons2,lats2,times2,Options)
        
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
    
    if(len(clusttemp) > maxlength):
        maxlength = len(clusttemp)
    
    clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
    clusters.append(clusttemp)
    
#Delete duplicates and sort on the first number in clusters:
unique_clusters = [list(x) for x in set(tuple(x) for x in clusters)]

#from operator import itemgetter
sorted_clusters =  sorted(unique_clusters)
print(timer() - starttime) # Time in seconds

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

np.savez(outfile, sorted_clusters=sorted_clusters, maxdists=np.array(maxdists),str_connected = str_connected)
