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
from Cluster_functions import read_file, read_file_clim, dt_array

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)

start = time.time()

#########################
# Thresholds
#########################

#1. Distance criterium
distthresh = 1.0 #1000.0

#2. Time criterium
timthresh = 36.0

#3. Length criterium 
lngthresh = 1.5 #1.5 #2.0 #calc_Rossby_radius(lat=45)*2.0 # 1000.0

#New set of experiments
timthreshs = np.arange(0.25,2.6,0.25)*24.0
distthreshs = np.arange(0.5,1.51,0.1)

timlength_threshs = np.arange(0.5,2.6,0.25)*24.0/6.0
lngthreshs = np.arange(0.6,2.41,0.2)

#Just one threshold
#timthreshs =  [30.0] #[1.0]
#distthreshs = [1.0]
#lngthreshs = [1.5] #[1.5]
#timmeth = "absolute" #"median" 
#Rossby_45 = calc_Rossby_radius(lat=45)

str_result = "Results_DJF_ERA5_" 
outdir = "/home/WUR/weije043/Clusters_Sensitivity/"

#########################
# Load storm tracks 
#########################
str_id, str_nr, str_dt, str_lat, str_lon = read_file(Options["st_file"],Options["nrskip"])
#str_id, str_nr, str_dt, str_lat, str_lon = read_file_clim(Options["st_file"],Options["nrskip"])

#Convert to an array
str_dt          = np.array(str_dt)
str_connected   = np.zeros(str_dt.shape)

from Cluster_functions import *

#########################
# Get indices of storms 
# so that ids_storms[id] gives the ids in the arrays
# str_id, str_lon,.. belonging to that specific storm
#########################
uniq_ids = np.unique(str_id)
ids_storms = get_indices_sparse(str_id)
nrstorms = len(uniq_ids)

#########################
# Preprocess storm tracks
#########################

#Check which hemisphere belongs storms to
hemstorms = np.full(nrstorms,"Undefined")
firstdt = []
lastdt = []

for strid in range(nrstorms):    
	dt_temp = str_dt[ids_storms[uniq_ids[strid]]]
	lat_temp = str_lat[ids_storms[uniq_ids[strid]]]

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

#######################################
# START SENSITIVITY EXPERIMENTS
#######################################
#for distthresh in distthreshs: #[0:1]
for lngthresh in lngthreshs:
    #for distthresh in distthreshs[0:4]:
    #for timthresh in timthreshs: #[0:1]
    for timlength_thresh in timlength_threshs: #[0:1]
        

        #Convert timthresh to td object 
        timthresh_dt = td(hours=timthresh)
        #Options["timthresh"] = timthresh
        #Options["distthresh"] = distthresh
        Options["lngthresh"] = lngthresh
        Options["timlngthresh"] = timlength_thresh        
        
        
        #for timthresh in timthreshs:
        print("---------------------------------------------")
        print("Start checking for:                          ")
        print("Distance threshold = " + str(distthresh))
        print("Time threshold = " + str(timthresh))
        print("Length threshold = " + str(lngthresh))
        print("Length threshold = " + str(lngthresh))
        print("---------------------------------------------")
        
        #Convert timthresh to td object 
        timthresh_dt = td(hours=Options["timthresh"])
        
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

        ######################################################
        # Step 1 Find connected and clustered storms
        #######################################################

        starttime = timer()
        for strm1 in range(nrstorms): 
            if(strm1%100 == 0):
                #print(strm1) 
                print("Strm1 :" + str(uniq_ids[strm1]))
            selidxs1 = ids_storms[uniq_ids[strm1]] 

            lats1 = str_lat[selidxs1]	
            lons1 = str_lon[selidxs1]
            times1 = str_dt[selidxs1]

            #Only compare with storms which are close enought im time compared to strm1 
            diffdt1  = firstdt - lastdt[strm1]
            diffdt2  = firstdt[strm1] - lastdt

            #To do: Check if this can be speed up
            strm2idxs = np.where((np.arange(nrstorms) > strm1) & ((diffdt1 <= timthresh_dt) & (diffdt2 <= timthresh_dt)) & (hemstorms == hemstorms[strm1]))[0]

            for strm2 in strm2idxs: 

                selidxs2 = ids_storms[uniq_ids[strm2]] 
                lats2 = str_lat[selidxs2]
                lons2 = str_lon[selidxs2] 
                times2 = str_dt[selidxs2]

                #Check if storm 1 and 2 are connected
                conn, angle, dt, dr, strConn1, strConn2  =\
                    connect_cyclones(lons1,lats1,times1,lons2,lats2,times2,Options)

                #if conn != 0: 
                #    print(strConn1)
                #    print(strConn2)

                #Save Results in arrays
                connTracks[strm2,strm1] = conn
                connTracks[strm1,strm2] = conn
                angleTracks[strm1,strm2] = angle
                dtTracks[strm1,strm2] = dt
                drTracks[strm1,strm2] = dr

                str_connected[selidxs1] += strConn1
                str_connected[selidxs2] += strConn2

        print(timer() - starttime) # Time in seconds

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

            for strid in cluster:

                #Convert strid to index
                stridx = [i for i in range(len(uniq_ids)) if uniq_ids[i] == strid]
                #np.where(uniq_ids == strid)[0]

                #Length clusters
                if(Options["frameworkSparse"] == True):
                    clusttemp = find_cluster_type_dokm(stridx,connTracks,contype="Length")
                else:
                    clusttemp, connTypes, clusterType = find_cluster_type3(stridx,connTracks,contype="Length")

                clusttemp = [uniq_ids[x] for x in clusttemp] #Convert indices to storm id
                subclusters_length.append(clusttemp)

                #Stationary clusters
                if(Options["frameworkSparse"] == True):
                    clusttemp = find_cluster_type_dokm(stridx,connTracks,contype="NoLength")
                else:
                    clusttemp, connTypes, clusterType = find_cluster_type3(stridx,connTracks,contype="NoLength") 

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
        outfile = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"]) + "_timoverlap_" +  formatter.format(Options["timlngthresh"]*6.0)
        #outfile = outdir + str_result + formatter.format(distthresh) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthresh)
        
        # TO DO: Update to remove warning message
        np.savez(outfile, sorted_clusters = np.array(sorted_clusters,dtype=object), sorted_subclusters_length = np.array(sorted_subclusters_length,dtype=object), sorted_subclusters_nolength = np.array(sorted_subclusters_nolength,dtype=object), connTracks = connTracks,str_connected = str_connected, dtTracks=dtTracks, drTracks=drTracks,angleTracks=angleTracks)

        
        
'''        
		######################################################
		# Find connected and clustered storms
		#######################################################
		connTracks = np.zeros([np.nanmax(str_id),np.nanmax(str_id)])

		maxdists = []
		maxdistsown = []
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
		# Statistics
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
'''

"""
#NH fraction
frac_mon_NH = np.zeros([12])
frac_mon_NHatlantic = np.zeros([12])
frac_mon_NHpacific = np.zeros([12])
frac_mon_SH = np.zeros([12])
for mn in range(12):
	frac_mon_NH[mn] = np.nansum(( str_connected > 0) & (str_lat > 0) & (str_month == mn + 1))/np.nansum((str_lat > 0) &  (str_month == mn + 1)) 

#SH fraction
for mn in range(12):
	frac_mon_SH[mn] = np.nansum(( str_connected > 0) & (str_lat < 0)  & (str_month == mn + 1))/np.nansum((str_lat < 0) &  (str_month == mn + 1)) 

#Atlantic NH fraction
for mn in range(12):
	frac_mon_NHatlantic[mn] = np.nansum(( str_connected > 0) & (str_lat > 0) & (str_lon > 280) & (str_month == mn + 1))/np.nansum((str_lat > 0) & (str_lon > 280) & (str_month == mn + 1)) 

#Pacific fraction

"""
