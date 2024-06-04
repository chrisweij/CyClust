#!/usr/bin/env python
# -*- encoding: utf-8
from __future__ import absolute_import, unicode_literals
import numpy as np 
import math
from scipy.sparse import csr_matrix
from numpy import loadtxt
from datetime import datetime as dt, timedelta as td

maxdists = []
maxdistsown = []
angles = []
anglesClust = []
clusterTypes = []
clusterTypes2 = []
angleTypes = []



# TO DO: Add description
# TO DO: Clean up of code (not all options are necessary anymore)
def connect_cyclones_test(lons1,lats1,times1,lons2,lats2,times2,
                     Options):
    
        conn = 0
        angle = 0
        dt = 0
        dr = 0
        str_contemp1 = np.zeros(len(lons1))
        str_contemp2 = np.zeros(len(lons2))
    
        if(Options["timmeth"] == "median"): 
            dists, timdiffs, = compare_trks_median(lons2,lats2,times2,lons1,lats1,times1,medians)
        elif(Options["timmeth"]  == "absolute"):
            dists, timdiffs, timspacediff  = compare_trks_np(lons2,lats2,times2,lons1,lats1,times1,Options["timthresh"]) #,timthresh timspacediff,

        if(Options["timspace"] == True):
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
            pntselect = np.nanmax((np.abs(timdiffs) <= Options["timthresh"]) & (dists <= Options["distthresh"]),axis=0) #*Rossby_45
            test1 = np.nansum(pntselect) 
            #Do the same for the other track
            pntselect2 = np.nanmax((np.abs(timdiffs) <= Options["timthresh"]) & (dists <= Options["distthresh"]),axis=1) #*Rossby_45
            test2 = np.nansum(pntselect2)
            
            nrPairs = np.nansum((np.abs(timdiffs) <= Options["timthresh"]) & (dists <= Options["distthresh"]))
            #print(nrPairs)

        if((test1 >=2) & (test2 >= 2)):
            pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])

            #Just select the points which are connected and calculate distances between the points for both tracks
            owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
            owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
            maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            maxtime = (np.nanmax(np.abs(owntims)) + np.nanmax(np.abs(owntims2)))/2.0
            #maxtime = (test1 + test2)/2.0*6.0

            maxtimspacediff = ((maxdist/Options["lngthresh"])**2.0 + (maxtime/(Options["timlngthresh"]*6.0))**2.0)**(0.5)
            
            ratio = (maxtime/(Options["timlngthresh"]*6.0))/(maxdist/Options["lngthresh"])
            angle = np.arctan(ratio)
            
            
            if(maxtimspacediff >=1.0):            
                angles.extend([angle*180/np.pi])
        else:
            maxdist = 0
            maxtime = 0
            maxtimspacediff = 0
        
        if(Options["connSpaceOrTime"] == True):
            if((maxtime >= (Options["timlngthresh"]*6.0)) or (maxdist >= Options["lngthresh"])): #(maxtime > (Options["timlngthresh"]*6.0))
                if(maxtime >= (Options["timlngthresh"]*6.0)): #maxtime > (Options["timlngthresh"]*6.0)
                    conn += 2
                    
                if(maxdist >= Options["lngthresh"]):
                    conn += 1
                                
                str_contemp1[pntselect] = 1.0
                str_contemp2[pntselect2] = 1.0

                anglesClust.extend([angle*180/np.pi])
                angle = angle*180/np.pi
                
                if(angle == 0):
                    print("Zero angle")
                    print((maxdist/lngthresh))
                    print((maxtime/(timlngthresh*6.0)))

                dr = (maxdist/Options["lngthresh"])
                dt = (maxtime/(Options["timlngthresh"]*6.0))
            else:
                anglesClust.extend([np.nan])
        elif(Options["connSpaceTime"] == True):
            if(maxtimspacediff > 1.0):
                conn = 1

                str_contemp1[pntselect] = 1.0
                str_contemp2[pntselect2] = 1.0

                anglesClust.extend([angle*180/np.pi])
                angle = angle*180/np.pi

                dr = (maxdist/Options["lngthresh"])
                dt = (maxtime/(Options["timlngthresh"]*6.0))
            else:
                anglesClust.extend([np.nan])
        elif(Options["connPairs"] == True):        
                if(nrPairs>= minPairs):
                    conn = 1
  
                    str_contemp1[pntselect] = 1.0
                    str_contemp2[pntselect2] = 1.0

                    anglesClust.extend([angle*180/np.pi])
                    angle = angle*180/np.pi
                    
                    print(maxdist)
        elif(Options["connTime"] & (test1 >=2) & (test2 >= 2)):
            if(Options["distmeth"] == "MaxDist"):
                pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
                maxdist = np.nanmax(pntdists)
            elif(Options["distmeth"] == "AlongTracksDirect"):
                #Just select the points which are connected and calculate distances between the points for both tracks
                owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
                owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
                maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            elif(Options["distmeth"] == "AlongTracks"):
                alongdists1, totaldist1 = dist_along_track_np(lons1[pntselect],lats1[pntselect])
                alongdists2, totaldist2 = dist_along_track_np(lons2[pntselect2],lats2[pntselect2])
                maxdist = (totaldist1 + totaldist2)/2.0
            else:
                raise ValueError("Max dist has not the right value")
            
            #Check if long tracks should be excluded
            TestLength = True
            if(Options["excludeLong"] == True):
                if(maxdist >= Options["lngthresh"]):
                    TestLength = False
            
            if((test1 >= Options["timlngthresh"]) & (test2 >= Options["timlngthresh"]) & TestLength):
                conn = 1
 
                str_contemp1[pntselect] = 1.0
                str_contemp2[pntselect2] = 1.0    
        #If both are connected over at least two points, check the maximum distance
        elif((test1 >=2) & (test2 >= 2)):
            
            avelat = np.nanmean(np.append(lats1[pntselect],lats2[pntselect2]))
            
            if(Options["distmeth"] == "MaxDist"):
                pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
                maxdist = np.nanmax(pntdists)
            elif(Options["distmeth"] == "AlongTracksDirect"):
                #Just select the points which are connected and calculate distances between the points for both tracks
                owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
                owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            
                maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            elif(Options["distmeth"] == "AlongTracks"):
                alongdists1, totaldist1 = dist_along_track_np(lons1[pntselect],lats1[pntselect])
                alongdists2, totaldist2 = dist_along_track_np(lons2[pntselect2],lats2[pntselect2])
                maxdist = (totaldist1 + totaldist2)/2.0
            else:
                raise ValueError("Max dist has not the right value")
            
            maxdists.append(maxdist)
            if(strm1 == strm2):
                maxdistsown.append(maxdist)
                
            #print("Strm2: " + str(strm2 + 1) + " Max dist: " + str(maxdist) + " Trck1: " + str(np.nanmax(owndists)) + " Trck2: " + str(np.nanmax(owndists2)))
                
            if(maxdist >= Options["lngthresh"]): #*Rossby_45*corrfac
                #print("Strm2: " + str(strm2 + 1) + " Max dist: " + str(maxdist) + " Trck1: " + str(np.nanmax(owndists)) + " Trck2: " + str(np.nanmax(owndists2)))
                conn = 1

                str_contemp1[pntselect] = 1.0 
                str_contemp2[pntselect2] = 1.0


                
        return conn, angle, dt, dr, str_contemp1, str_contemp2



#Recursive function to find uniquely connected cluster of storms
def find_cluster(cluster,connTracks):
    #print("CLustering analysis for the following storms:")
    #print(cluster)
    cluster_old = cluster

    #Loop over storms to find connected storms
    for stridx in cluster: 
        conntemp = connTracks[stridx,::] #-1
        if(np.nansum(conntemp) > 0):
            strmstemp = np.where(conntemp > 0)[0]  #+ 1
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))

    #Remove duplicate storms
    cluster = np.unique(cluster)

    #Check if all storms are counted??
    if(len(cluster) == len(cluster_old)):
        return cluster_old
    else:
        return find_cluster(cluster,connTracks)
    
#Recursive function to find uniquely connected cluster of storms + Type of cluster
def find_cluster_type(cluster,connTracks):
    #print("CLustering analysis for the following storms:")
    #print(cluster)
    cluster_old = cluster

    #Loop over storms to find connected storms
    for stridx in cluster: 
        conntemp = connTracks[stridx,::] #-1
        if( np.nansum(conntemp) > 0):
            strmstemp = np.where(conntemp > 0)[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))
            
    #Remove duplicate storms
    cluster = np.unique(cluster)

    #Check if all storms are counted??
    if(len(cluster) == len(cluster_old)):
        connTypes = [x for strm in cluster for x in list(connTracks[strm,:][connTracks[strm,:] != 0])]
        
        if(len(cluster) == 1):
            clusterType = "None"
        elif(all((x == 2.0 or x == 3.0) for x in connTypes)):
            clusterType = "Time"
        elif(all((x == 1.0 or x == 3.0) for x in connTypes)):
            clusterType = "Length"
        else:
            clusterType = "Mixed"
        
        return cluster_old, connTypes, clusterType
    else:
        #connTypes.extend(list(typetemp))
        return find_cluster_type(cluster,connTracks)
           

# TO DO: Check if function is still necessary
def find_cluster_type2(cluster,connTracks,angleTracks):
    #print("CLustering analysis for the following storms:")
    #print(cluster)
    cluster_old = cluster

    #Loop over storms to find connected storms
    for stridx in cluster: 
        conntemp = connTracks[stridx,::] #-1
        if(np.nansum(conntemp) > 0):
            strmstemp = np.where(conntemp > 0)[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))

    #Remove duplicate storms
    cluster = np.unique(cluster)

    #Check if all storms are counted??
    if(len(cluster) == len(cluster_old)):
        connTypes = [x for strm in cluster for x in list(connTracks[strm,:][connTracks[strm,:] != 0])]
        anglesClust = [x for strm in cluster for x in list(angleTracks[strm,:][angleTracks[strm,:] != 0])]
        
        if(len(cluster) == 1):
            clusterType = "None"
        elif(all((x == 1.0) for x in connTypes)):
            clusterType = "Length"
        elif(all((x == 2.0) for x in connTypes)):
            clusterType = "Time"
        else:
            clusterType = "Mixed"
        
        meanAngle = np.nanmean(anglesClust)
        stdAngle = np.nanstd(anglesClust)

        if(len(cluster) == 1):
            angleType = "None"        
        elif(meanAngle > 60):
            angleType = "Time"
        elif(meanAngle < 45):
            angleType = "Length"
        else:
            angleType = "Mixed"
        return cluster_old, connTypes, anglesClust, clusterType, angleType
    else:
        #connTypes.extend(list(typetemp))
        return find_cluster_type2(cluster,connTracks,angleTracks)
