#!/usr/bin/env python
# -*- encoding: utf-8
from __future__ import absolute_import, unicode_literals
import numpy as np 
import math

##################################################
# FUNCTIONS TO COMPARE DISTANCES
###################################################
def calc_Rossby_radius(lat=45,N=1.3e-2,H=10):
	return N*H/(2*7.29*10**-5*np.sin(lat*np.pi/180))

#https://gist.github.com/nickjevershed/6480846
def great_circle(lat1, long1, lat2, long2,dist="kilometers"):

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
    #print(cos)
    if(cos > 1):
       cos = 1.0
    arc = math.acos( cos )

    # Calculate distance in meters
    if (dist == "kilometers"): 
       dist = arc*6378.16
    elif(dist == "meters"):
       dist = arc*6378.16*1000.0

    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return dist

#Similar as previous function, but using numpy instead of math
def great_circle_np(lat1, long1, lat2, long2,dist="kilometers"):

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
        
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
        
    theta2 = long2*degrees_to_radians
        
    # Compute spherical distance from spherical coordinates.
        
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    
    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) + 
           np.cos(phi1)*np.cos(phi2))
    cos[np.where(cos>1)] = 1
    arc = np.arccos( cos )

    # Calculate distance in kilometers or meters
    if (dist == "kilometers"): 
       dist = arc*6378.16
    elif(dist == "meters"):
       dist = arc*6378.16*1000.0

    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return dist


#Symmetrize matrix
def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
    for i != j.
    """
    return a + a.T - numpy.diag(a.diagonal())

#Compute the spatial and distance between two tracks
def compare_trks(x_1,y_1,t_1,x_2,y_2,t_2):
	len1 = len(x_1)
	len2 = len(x_2)

	dist = np.empty([len1,len2])*np.nan
	timdiff = np.empty([len1,len2])*np.nan

	for idx1 in range(len1):
		for idx2 in range(len2): #range(len2): #
			gctemp = great_circle(y_1[idx1],x_1[idx1],y_2[idx2],x_2[idx2])
			avelat = (y_2[idx2] + y_1[idx1])*0.5
			corrfac = np.abs(calc_Rossby_radius(lat=avelat)) #/calc_Rossby_radius(lat=45))
			dist[idx1,idx2] = gctemp/corrfac
			dttemp = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600
			timdiff[idx1,idx2] = dttemp
	return dist, timdiff

#Compute the spatial and distance between two tracks
def compare_trks_np(x_1,y_1,t_1,x_2,y_2,t_2):
	len1 = len(x_1)
	len2 = len(x_2)

	timdiff = np.empty([len1,len2])*np.nan

	lt1 = np.outer(y_1,np.ones(len2))
	ln1 = np.outer(x_1,np.ones(len2))
	lt2 = np.outer(np.ones(len1),y_2)
	ln2 =np.outer(np.ones(len1),x_2)

	avelat = (lt1 + lt2)*0.5
	corrfac = np.abs(calc_Rossby_radius(lat=avelat)) #/calc_Rossby_radius(lat=45))
	dist = great_circle_np(lt1,ln1,lt2,ln2)/corrfac


	for idx1 in range(len1):
		for idx2 in range(len2): #range(len2): #
			dttemp = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600
			timdiff[idx1,idx2] = dttemp


	#t1 = np.outer(t_1,np.ones(len2))
	#t2 = np.outer(np.ones(len1),t_2)
	#timdiff = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600

	return dist, timdiff
	
#Compare 
def compare_trks_median(x_1,y_1,t_1,x_2,y_2,t_2,median):
	len1 = len(x_1)
	len2 = len(x_2)

	lats = np.arange(90,-90.1,-1.5)
	lons = np.arange(-180,180,1.5)

	dist = np.zeros([len1,len2])
	timdiff = np.zeros([len1,len2])

	for idx1 in range(len1):
		for idx2 in range(idx1,len2):
			avelat = (y_2[idx2] + y_1[idx1])*0.5
			diff2 = (360 - np.nanmax([x_2[idx2],x_1[idx1]]) + np.nanmin([x_2[idx2],x_1[idx1]]))%360

			if( np.abs(x_2[idx2] - x_1[idx1]) < 180):
				#print("Option 1:")
				avelon = (x_2[idx2] + x_1[idx1])*0.5  
			else:
				#print("Option 2:")
				avelon = (np.nanmax([x_2[idx2],x_1[idx1]]) + diff2/2.0)%360
			#print("Lat 1: " + str(y_1[idx1]) + " Lat 2: " + str(y_2[idx2]) + " Ave: " + str(avelat))
			#print("Lon 1: " + str(x_1[idx1]) + " Lon 2: " + str(x_2[idx2]) + " Ave: " + str(avelon))
			#print(str(np.abs((x_2[idx2] - x_1[idx1]))) + " " +  str((360 - x_1[idx1] + x_2[idx2])%360))

			gctemp = great_circle(y_1[idx1],x_1[idx1],y_2[idx2],x_2[idx2])
			corrfac = np.abs(calc_Rossby_radius(lat=avelat)) #/calc_Rossby_radius(lat=45))
			dist[idx1,idx2] = gctemp/corrfac
			dttemp = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600	
			
			#Get closest grid point
			if (avelon > 180): 
				avelon -= 360.0
			latidx = np.argmin(np.abs(lats - avelat))
			lonidx = np.argmin(np.abs(lons - avelon))

			#print("Closest Lat: " + str(lats[latidx]))
			#print("Closest Lon: " + str(lons[lonidx]))
			timdiff[idx1,idx2] = dttemp/(median[latidx,lonidx]*6.0)
	return dist, timdiff
	
#def connect_tracks


#Recursive function to find uniquely connected cluster of storms
def find_cluster(cluster,connTracks):
	print("CLustering analysis for the following storms:")
	print(cluster)
	cluster_old = cluster
	
	#Loop over storms to find connected storms
	for stridx in cluster: 
		conntemp = connTracks[stridx - 1,::]
		if(np.nansum(conntemp) > 0):
			strmstemp = np.where(conntemp > 0)[0]  + 1
			cluster = np.append(cluster,np.array(strmstemp,dtype=int))

	#Remove duplicate storms
	cluster = np.unique(cluster)

	#Check if all storms are counted??
	if(len(cluster) == len(cluster_old)):
		return cluster_old
	else:
		return find_cluster(cluster,connTracks)
