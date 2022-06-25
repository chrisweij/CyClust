#!/usr/bin/env python
# -*- encoding: utf-8

from scipy.ndimage.filters import gaussian_filter
#from geopy.distance import great_circle as great_circle_old

from datetime import datetime as dt, timedelta as td
import numpy as np
import sparse
from scipy.sparse import dok_matrix
from numpy import loadtxt

#exec(open("projection.py").read())
#from dynlib import sphere, cm
import time
from numpy import loadtxt
from Cluster_functions import calc_Rossby_radius, compare_trks_np, find_cluster, find_cluster_type_dokm, unnest

from timeit import default_timer as timer

#########################
# Thresholds
#########################

#1. Distance criterium
distthresh = 1.0 #1000.0

#2. Time criterium
timthresh = 36.0

#3. Length/Time criterium 
lngthresh = 1.5 #1.5 #2.0 #calc_Rossby_radius(lat=45)*2.0 # 1000.0
timlngthresh = 6
minPairs = 24

# Options
timmeth = "absolute"
timspace = False #Use a combined Time-Space criterium 
connTime = False #Connect on lengths
connSpaceTime = False
connSpaceOrTime = True
connPairs = False
excludeLong = False #Exclude other type
distmeth = "AlongTracksDirect" #"MaxDist" #"AlongTracks" "AlongTrackDirect"
frameworkSparse = True #True


# Output directory
outdir = "Results/"

# Minimum of nr. of storms in a "family"
minstorms = 3
outdir = "Clusters_output/"
str_result = "EI_2011_2012"

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

wintstrms = np.zeros(nrstorms)
yrstorms = np.zeros(nrstorms)
mnstorms = np.zeros(nrstorms)
hemstorms = np.full(nrstorms,"Undefined")
firstdt = []
lastdt = []

uniq_ids = np.unique(str_id)

for strid in range(nrstorms):
    
	dt_temp = str_dt[str_id == uniq_ids[strid]]
	lat_temp = str_lat[str_id == uniq_ids[strid]]

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

#Convert timthresh to td object 
timthresh_dt = td(hours=timthresh)

######################################################
# Find connected and clustered storms
#######################################################
if(frameworkSparse == False):
    connTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    angleTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    drTracks  = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
    dtTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])
else:
    connTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    angleTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    drTracks  = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))
    dtTracks = dok_matrix((np.nanmax(str_id),np.nanmax(str_id)))


    maxdists = []
maxdistsown = []
angles = []
anglesClust = []
clusterTypes = []
clusterTypes2 = []
angleTypes = []

starttime = timer()
for strm1 in range(nrstorms): #range(nrstorms): #[1]: # #range(6500,7000): # 
    #print("Strm1 :" + str(uniq_ids[strm1]))
    selidxs1 = np.where(str_id == uniq_ids[strm1])
    lats1 = str_lat[selidxs1]	
    #print(lats1)
    lons1 = str_lon[selidxs1]
    times1 = str_dt[selidxs1]
    
    #Only check if the storm is in the current month, or one after or before it, in the same hemisphere. 
    #diffmon = mnstorms_rel[strm1] - mnstorms_rel
    diffdt1  = firstdt - lastdt[strm1]
    diffdt2  = firstdt[strm1] - lastdt
    
    #strm2idxs = np.where( (np.abs(diffmon) <= 1) & (hemstorms == hemstorms[strm1]))[0]
    strm2idxs = np.where((np.arange(nrstorms) > strm1) & ((diffdt1 <= timthresh_dt) & (diffdt2 <= timthresh_dt)) & (hemstorms == hemstorms[strm1]))[0]
    #print("Nr strm2: " + str(len(strm2idxs)))
    
    for strm2 in strm2idxs: #[5]: #strm2idxs: #range(minstidx,maxstidx)
        #print("Strm1 :" + str(strm1 + 1) + " Strm2: " + str(strm2 + 1))

        selidxs2 = np.where(str_id == uniq_ids[strm2])
        lats2 = str_lat[selidxs2]
        lons2 = str_lon[selidxs2] 
        times2 = str_dt[selidxs2]
        #print(lats2)

        if(timmeth == "median"): 
            dists, timdiffs, = compare_trks_median(lons2,lats2,times2,lons1,lats1,times1,medians)
        elif(timmeth == "absolute"):
            dists, timdiffs, timspacediff  = compare_trks_np(lons2,lats2,times2,lons1,lats1,times1,timthresh) #,timthresh timspacediff,

        if(timspace == True):
            #Calculate distance over which storms are connected
            #First select just the lons, lats times over which a particular storm is connected
            pntselect = np.nanmax((timspacediff < 1.0),axis=0) #*Rossby_45
            test1 = np.nansum(pntselect) 
            #Do the same for the other track
            pntselect2 = np.nanmax((timspacediff < 1.0),axis=1) #*Rossby_45
            test2 = np.nansum(pntselect2)       
        else:
            #Calculate distance over which storms are connected
            #First select just the lons, lats times over which a particular storm is connected
            pntselect = np.nanmax((np.abs(timdiffs) <= timthresh) & (dists <= distthresh),axis=0) #*Rossby_45
            test1 = np.nansum(pntselect) 
            #Do the same for the other track
            pntselect2 = np.nanmax((np.abs(timdiffs) <= timthresh) & (dists <= distthresh),axis=1) #*Rossby_45
            test2 = np.nansum(pntselect2)
            
            nrPairs = np.nansum((np.abs(timdiffs) <= timthresh) & (dists <= distthresh))
            #print(nrPairs)

        if((test1 >=2) & (test2 >= 2)):
            pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])

            #Just select the points which are connected and calculate distances between the points for both tracks
            owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
            owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
            maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            maxtime = (np.nanmax(np.abs(owntims)) + np.nanmax(np.abs(owntims2)))/2.0
            #maxtime = (test1 + test2)/2.0*6.0

            maxtimspacediff = ((maxdist/lngthresh)**2.0 + (maxtime/(timlngthresh*6.0))**2.0)**(0.5)
            
            ratio = (maxtime/(timlngthresh*6.0))/(maxdist/lngthresh)
            angle = np.arctan(ratio)
            
            
            if(maxtimspacediff >=1.0):
                #print("Max distance: " + str(maxdist))
                #print("Max time diff: " + str(maxtime))
                #print("Max spacetime diff: " + str(maxtimspacediff))
            

                #print("Angle: " + str(angle*180/np.pi))
                #print("Ratio: " + str(ratio))
            
                angles.extend([angle*180/np.pi])
        else:
            maxdist = 0
            maxtime = 0
            maxtimspacediff = 0
        
        if(connSpaceOrTime == True):
            if((maxtime > (timlngthresh*6.0)) or (maxdist >= lngthresh)):
                if(maxtime > (timlngthresh*6.0)):
                    connTracks[strm1,strm2] = connTracks[strm1,strm2] + 2
                    connTracks[strm2,strm1] = connTracks[strm2,strm1] + 2
                    
                if(maxdist >= lngthresh):
                    connTracks[strm1,strm2] = connTracks[strm1,strm2] + 1
                    connTracks[strm2,strm1] = connTracks[strm2,strm1] + 1   
                                
                str_contemp1 = str_connected[selidxs1]
                str_contemp1[pntselect] = 1.0
                str_contemp2 = str_connected[selidxs2]
                str_contemp2[pntselect2] = 1.0
                str_connected[selidxs1] = str_contemp1
                str_connected[selidxs2] = str_contemp2  

                anglesClust.extend([angle*180/np.pi])
                angleTracks[strm1,strm2] = angle*180/np.pi
                
                if(angle == 0):
                    print("Zero angle")
                    print((maxdist/lngthresh))
                    print((maxtime/(timlngthresh*6.0)))

                drTracks[strm1,strm2] = (maxdist/lngthresh)
                dtTracks[strm1,strm2] = (maxtime/(timlngthresh*6.0))
            else:
                anglesClust.extend([np.nan])
        elif(connSpaceTime == True):
            if(maxtimspacediff > 1.0):
                connTracks[strm1,strm2] = 1
                connTracks[strm2,strm1] = 1
                str_contemp1 = str_connected[selidxs1]
                str_contemp1[pntselect] = 1.0
                str_contemp2 = str_connected[selidxs2]
                str_contemp2[pntselect2] = 1.0
                str_connected[selidxs1] = str_contemp1
                str_connected[selidxs2] = str_contemp2  

                anglesClust.extend([angle*180/np.pi])
                angleTracks[strm1,strm2] = angle*180/np.pi

                drTracks[strm1,strm2] = (maxdist/lngthresh)
                dtTracks[strm1,strm2] = (maxtime/(timlngthresh*6.0))
            else:
                anglesClust.extend([np.nan])
        elif(connPairs == True):        
                if(nrPairs>= minPairs):
                    connTracks[strm1,strm2] = 1
                    connTracks[strm2,strm1] = 1
                    str_contemp1 = str_connected[selidxs1]
                    str_contemp1[pntselect] = 1.0
                    str_contemp2 = str_connected[selidxs2]
                    str_contemp2[pntselect2] = 1.0
                    str_connected[selidxs1] = str_contemp1
                    str_connected[selidxs2] = str_contemp2 
                    anglesClust.extend([angle*180/np.pi])
                    angleTracks[strm1,strm2] = angle*180/np.pi
                    
                    print(maxdist)
        elif(connTime & (test1 >=2) & (test2 >= 2)):
            if(distmeth == "MaxDist"):
                pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
                maxdist = np.nanmax(pntdists)
            elif(distmeth == "AlongTracksDirect"):
                #Just select the points which are connected and calculate distances between the points for both tracks
                owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
                owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
                maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            elif(distmeth == "AlongTracks"):
                alongdists1, totaldist1 = dist_along_track_np(lons1[pntselect],lats1[pntselect])
                alongdists2, totaldist2 = dist_along_track_np(lons2[pntselect2],lats2[pntselect2])
                maxdist = (totaldist1 + totaldist2)/2.0
            else:
                raise ValueError("Max dist has not the right value")
            
            #Check if long tracks should be excluded
            TestLength = True
            if(excludeLong == True):
                if(maxdist >= lngthresh):
                    TestLength = False
            
            if((test1 >= timlngthresh) & (test2 >= timlngthresh) & TestLength):
                connTracks[strm1,strm2] = 1
                connTracks[strm2,strm1] = 1
                str_contemp1 = str_connected[selidxs1]
                str_contemp1[pntselect] = 1.0
                str_contemp2 = str_connected[selidxs2]
                str_contemp2[pntselect2] = 1.0
                str_connected[selidxs1] = str_contemp1
                str_connected[selidxs2] = str_contemp2           
        #If both are connected over at least two points, check the maximum distance
        elif((test1 >=2) & (test2 >= 2)):
            
            avelat = np.nanmean(np.append(lats1[pntselect],lats2[pntselect2]))
            
            if(distmeth == "MaxDist"):
                pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
                maxdist = np.nanmax(pntdists)
            elif(distmeth == "AlongTracksDirect"):
                #Just select the points which are connected and calculate distances between the points for both tracks
                owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
                owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
                maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            elif(distmeth == "AlongTracks"):
                alongdists1, totaldist1 = dist_along_track_np(lons1[pntselect],lats1[pntselect])
                alongdists2, totaldist2 = dist_along_track_np(lons2[pntselect2],lats2[pntselect2])
                maxdist = (totaldist1 + totaldist2)/2.0
            else:
                raise ValueError("Max dist has not the right value")
            
            maxdists.append(maxdist)
            if(strm1 == strm2):
                maxdistsown.append(maxdist)
                
            #print("Strm2: " + str(strm2 + 1) + " Max dist: " + str(maxdist) + " Trck1: " + str(np.nanmax(owndists)) + " Trck2: " + str(np.nanmax(owndists2)))
                
            if(maxdist >= lngthresh): #*Rossby_45*corrfac
                #print("Strm2: " + str(strm2 + 1) + " Max dist: " + str(maxdist) + " Trck1: " + str(np.nanmax(owndists)) + " Trck2: " + str(np.nanmax(owndists2)))
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
#np.fill_diagonal(connTracks,0)

if(frameworkSparse == True):
    connTracks = connTracks.tocsr()

########################
# Step 2 Find clusters
########################
clusters = []
maxlength = 1

for stridx in range(nrstorms):
    #print(stridx)
    if(frameworkSparse == True):
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
        if(frameworkSparse == True):
            clusttemp = find_cluster_type_dokm([stridx - 1],connTracks,contype="Length")
        else:
            clusttemp, connTypes, clusterType = find_cluster_type3([stridx - 1],connTracks,contype="Length")


        clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
        subclusters_length.append(clusttemp)
        
        #Stationary clusters
        if(frameworkSparse == True):
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
#sorted_clusters = sorted_clusters_length

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
'''
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
'''

######################################################
# Save results
######################################################
formatter =  "{:1.1f}"
outfile = outdir + str_result + formatter.format(distthresh) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthresh)
np.savez(outfile, sorted_clusters=sorted_clusters, lengths = lengths, lengthclust= lengthclust, winters=winters,nrclst_wint = nrclst_wint, nrstrm_wint = nrstrm_wint, nrstrmclst_wint = nrstrmclst_wint,maxdists=np.array(maxdists),str_connected = str_connected)

