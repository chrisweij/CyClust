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

##################################################
# Data processing functions
###################################################

def read_file(st_file, nrskip=1):
    str_id   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[0],dtype=int)
    str_nr   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[1],dtype=int)
    str_date = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[2],dtype=int)
    str_lat  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[4])
    str_lon  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[3])
    
    str_dt = dt_array(str_date)
    
    return str_id, str_nr, str_dt, str_lat, str_lon

def read_file_clim(st_file, nrskip=0):
    str_id   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[0],dtype=int)
    str_nr   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[1],dtype=int)
    str_year = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[2],dtype=int)
    str_day = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[3],dtype=int)
    str_lat  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[5])
    str_lon  = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[4])
  
    str_dt = []
    for idx in range(len(str_year)):  

        str_dt.append(dt(str_year[idx]-1,12,1,0) + td(days=(str_day[idx]-1)/4))
	
    str_dt = np.array(str_dt)
    
    return str_id, str_nr, str_dt, str_lat, str_lon
    
#Function to convert timestamp (YYYYMMDDHH) to datetime    
def dt_array(str_date):
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
        
    return str_dt

##################################################
# General FUNCTIONS
###################################################

def unnest(l):
    '''
    Function to unnest list
    
    Input: list l
    Output: Unnested list 
    '''
    
    
    l_unnest = [item for sublist in l for item in sublist]
    return l_unnest

    
def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

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
    
    theta1 = long1*degrees_to_radians
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
def compare_trks_np(x_1,y_1,t_1,x_2,y_2,t_2,timthresh=None):
    len1 = len(x_1)
    len2 = len(x_2)

    timdiff = np.empty([len1,len2])*np.nan

    lt1 = np.outer(y_1,np.ones(len2))
    ln1 = np.outer(x_1,np.ones(len2))
    lt2 = np.outer(np.ones(len1),y_2)
    ln2 = np.outer(np.ones(len1),x_2)

    avelat = (lt1 + lt2)*0.5
    corrfac = np.abs(calc_Rossby_radius(lat=avelat)) #/calc_Rossby_radius(lat=45))
    dist = great_circle_np(lt1,ln1,lt2,ln2)/corrfac


    for idx1 in range(len1):
        for idx2 in range(len2): #range(len2): #
            dttemp = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600
            timdiff[idx1,idx2] = dttemp
            
    #Calculate time space diff
    if( timthresh == None):
        timspace_diff = None
    else:
        timspace_diff = (dist**2 + timdiff**2/timthresh**2)**(0.5)
    #t1 = np.outer(t_1,np.ones(len2))
    #t2 = np.outer(np.ones(len1),t_2)
    #timdiff = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600
    if( timthresh == None):
        return dist, timdiff
    else:
        return dist, timdiff, timspace_diff
    
    
def dist_along_track_np(x,y):
    lentrk = len(x)
    
    if(lentrk <= 1):
        raise ValueError("Vector must have at least two points")
    else:
        avelat = (y[1:] + y[0:(lentrk-1)])*0.5
        corrfac = np.abs(calc_Rossby_radius(lat=avelat)) 
        dist = great_circle_np(y[1:],x[1:],y[0:(lentrk-1)],x[0:(lentrk-1)])/corrfac
        
    totaldist = np.nansum(dist)
    
    return dist, totaldist
        
#Compare with local median threshold
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


			gctemp = great_circle(y_1[idx1],x_1[idx1],y_2[idx2],x_2[idx2])
			corrfac = np.abs(calc_Rossby_radius(lat=avelat)) #/calc_Rossby_radius(lat=45))
			dist[idx1,idx2] = gctemp/corrfac
			dttemp = ((t_1[idx1] - t_2[idx2]).total_seconds())/3600	
			
			#Get closest grid point
			if (avelon > 180): 
				avelon -= 360.0
			latidx = np.argmin(np.abs(lats - avelat))
			lonidx = np.argmin(np.abs(lons - avelon))

			timdiff[idx1,idx2] = dttemp/(median[latidx,lonidx]*6.0)
	return dist, timdiff


def connect_cyclones(lons1,lats1,times1,lons2,lats2,times2,
                     Options):
        '''
        Function to check if two storm tracks are 'clustered', according to certain distance criteria
        
        Input:
        lons1,  longitude (in degrees) of track 1
        lats1, latitude (in degrees) of track 1
        times1, times of track 1
        lons2,  longitude (in degrees) of track 2
        lats2, latitude (in degrees) of track 2
        times2, times of track 2
        Options, 
        
        In Options:
        '''
    
        conn = 0
        angle = 0
        dt = 0
        dr = 0
        str_contemp1 = np.zeros(len(lons1))
        str_contemp2 = np.zeros(len(lons2))
    
        dists, timdiffs, timspacediff  = compare_trks_np(lons2,lats2,times2,lons1,lats1,times1,Options["timthresh"]) #,timthresh timspacediff,

        ###############################################################################
        # Step 1: Selection of points over which a particular storm tracks are connected
        # First it is checked at each point if the local distance and temporal criteria are satisfied, 
        # gives a matrix of m x n, with m length of track 1, and n length track 2
        ###############################################################################
        point_check = (np.abs(timdiffs) <= Options["timthresh"]) & (dists <= Options["distthresh"])
        #Secondly, check which points along track 1 are connected to any (arbitrary) point of track 2
        pntselect = np.nanmax(point_check,axis=0) #*Rossby_45
        test1 = np.nansum(pntselect) 
        
        #Do the same for the other track
        pntselect2 = np.nanmax(point_check,axis=1) #*Rossby_45
        test2 = np.nansum(pntselect2)

        nrPairs = np.nansum((np.abs(timdiffs) <= Options["timthresh"]) & (dists <= Options["distthresh"]))

        ###############################################################################
        # Step 2: Calculate distance and time over which storm tracks are connected
        # Only makes sense if at least two points are connected along each track
        ###############################################################################
        if((test1 >=2) & (test2 >= 2)):
            pntdists, pnttimdiffs, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])

            #Just select the points which are connected and calculate distances between the points for both tracks
            owndists, owntims, = compare_trks_np(lons1[pntselect],lats1[pntselect],times1[pntselect],lons1[pntselect],lats1[pntselect],times1[pntselect])
            owndists2, owntims2, = compare_trks_np(lons2[pntselect2],lats2[pntselect2],times2[pntselect2],lons2[pntselect2],lats2[pntselect2],times2[pntselect2])
            if(Options["distmeth"] == "AlongTracksDirect"):
                #Just select the points which are connected and calculate distances between the points for both tracks
                maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            elif(Options["distmeth"] == "AlongTracks"):
                alongdists1, totaldist1 = dist_along_track_np(lons1[pntselect],lats1[pntselect])
                alongdists2, totaldist2 = dist_along_track_np(lons2[pntselect2],lats2[pntselect2])
                maxdist = (totaldist1 + totaldist2)/2.0
                
            #maxdist = (np.nanmax(owndists) + np.nanmax(owndists2))/2.0
            maxtime = (np.nanmax(np.abs(owntims)) + np.nanmax(np.abs(owntims2)))/2.0
            
            ratio = (maxtime/(Options["timlngthresh"]*6.0))/(maxdist/Options["lngthresh"])
            angle = np.arctan(ratio)
                
            #if(maxtimspacediff >=1.0):            
            #    angles.extend([angle*180/np.pi])
        else:
            maxdist = 0
            maxtime = 0
        
        if((maxtime > (Options["timlngthresh"]*6.0)) or (maxdist >= Options["lngthresh"])):
            if(maxtime > (Options["timlngthresh"]*6.0)):
                conn += 2

            if(maxdist >= Options["lngthresh"]):
                conn += 1

            str_contemp1[pntselect] = 1.0
            str_contemp2[pntselect2] = 1.0

            anglesClust.extend([angle*180/np.pi])
            angle = angle*180/np.pi

            if(angle == 0):
                print("Zero angle")
                print((maxdist/Options["lngthresh"]))
                print((maxtime/(Options["timlngthresh"]*6.0)))

            dr = (maxdist/Options["lngthresh"])
            dt = (maxtime/(Options["timlngthresh"]*6.0))
                
        return conn, angle, dt, dr, str_contemp1, str_contemp2


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
            if((maxtime > (Options["timlngthresh"]*6.0)) or (maxdist >= Options["lngthresh"])):
                if(maxtime > (Options["timlngthresh"]*6.0)):
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
    

#Recursive function to find uniquely connected cluster of storms + Type of cluster
def find_cluster_type3(cluster,connTracks,contype="All"):
    #print("CLustering analysis for the following storms:")
    #print(cluster)
    cluster_old = cluster

    #Loop over storms to find connected storms
    for stridx in cluster: 
        conntemp = connTracks[stridx,::] #-1
        if(contype == "All" and np.nansum(conntemp) > 0):
            strmstemp = np.where(conntemp > 0)[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))
        elif(contype == "Length" and np.nansum((conntemp == 1.0) | (conntemp == 3.0)) > 0):
            strmstemp = np.where((conntemp == 1.0) | (conntemp == 3.0))[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))
        elif(contype == "Time" and np.nansum(conntemp >= 2.0) > 0):
            strmstemp = np.where(conntemp >= 2.0)[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))    
        if(contype == "NoLength" and np.nansum((conntemp == 2.0)) > 0):
            strmstemp = np.where(conntemp == 2.0)[0]  #+ stridx
            #typetemp = conntemp[conntemp > 0] 
            cluster = np.append(cluster,np.array(strmstemp,dtype=int))    
            
            
    #Remove duplicate storms
    cluster = np.unique(cluster)

    #Check if all storms are counted??
    if(len(cluster) == len(cluster_old)):
        if(contype == "All"):
            connTypes = [x for strm in cluster for x in list(connTracks[strm,:][connTracks[strm,:] != 0])]
        elif(contype == "Length"):
            connTypes = [x for strm in cluster for x in list(connTracks[strm,:][(connTracks[strm,:] == 1.0) | (connTracks[strm,:] == 3.0)])]
        elif(contype == "Time"):
            connTypes = [x for strm in cluster for x in list(connTracks[strm,:][connTracks[strm,:] >= 2.0])]
        elif(contype == "NoLength"):
            connTypes = [x for strm in cluster for x in list(connTracks[strm,:][connTracks[strm,:] == 2.0])]
        else:
            connTypes = []
            
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
        return find_cluster_type3(cluster,connTracks,contype=contype)
    
    
#Recursive function to find uniquely connected cluster of storms + Type of cluster
def find_cluster_type_dokm(cluster,connTracks,contype="All"):
    #print("CLustering analysis for the following storms:")
    #print(cluster)
    cluster_old = cluster

    #Loop over storms to find connected storms
    for stridx in cluster: 
        conntemp = connTracks.getrow(stridx).data #-1
        #nonzero  = connTracks.getrow(stridx).nonzero()[1]
        nonzero  = connTracks[stridx,::].nonzero()[1]
        
        if(len(conntemp) > 0):    
            if(contype == "All"):
                strmstemp = np.where(conntemp > 0)[0]  #+ stridx
                #typetemp = conntemp[conntemp > 0] 
                cluster = np.append(cluster,nonzero)
            elif(contype == "Length" and np.nansum((conntemp == 1.0) | (conntemp == 3.0)) > 0):
                strmstemp = nonzero[np.where((conntemp == 1.0) | (conntemp == 3.0))[0]]  #+ stridx
                cluster = np.append(cluster,np.array(strmstemp,dtype=int))
            elif(contype == "Time" and np.nansum(conntemp >= 2.0) > 0):
                strmstemp = nonzero[np.where(conntemp >= 2.0)[0]]  
                cluster = np.append(cluster,np.array(strmstemp,dtype=int))    
            if(contype == "NoLength" and np.nansum((conntemp == 2.0)) > 0):
                strmstemp = nonzero[np.where(conntemp == 2.0)[0]]
                cluster = np.append(cluster,np.array(strmstemp,dtype=int))     
            
    #Remove duplicate storms
    cluster = np.unique(cluster)

    #Check if all storms are counted??
    if(len(cluster) == len(cluster_old)):
        return cluster_old
    else:
        #connTypes.extend(list(typetemp))
        return find_cluster_type_dokm(cluster,connTracks,contype=contype)
