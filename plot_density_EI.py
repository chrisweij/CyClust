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
#from Cluster_functions import read_file, read_file_clim, dt_array

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)

#Specific functions related to the clustering algorithm
from Cluster_functions import *
from dynlib.metio.erainterim import conf, dt, get_instantaneous, metsave, metopen

#Plotting related
import matplotlib.pyplot as plt
import matplotlib as mpl
import dynlib.proj as proj
import dynlib.figures as fig
import copy
from cmocean import cm as cmoc
from mpl_toolkits.basemap import Basemap
import os

clustchar = "All" #length" #"All" #"length""nolength" or "all"
minstorms = 2
distchar = "250km"
calcDensityEI = False
calcDensityClust = False
calcQuants = True #True
calcDensitySolo = False

# PLOTTING SETTINGS "
scale_density = np.arange(3,25,1)
scale_anomaly = np.arange(-5,5.1,1)
scale_anomaly_perc = np.arange(-40,40,5)
cmap_IMILAST = (mpl.colors.ListedColormap(["yellow","orange","green","lightblue","blue","darkblue","midnightblue","purple","red"])
        .with_extremes(over='grey', under='white'))
scale_IMILAST = [2,5,10,15,20,25,35,50,70,100]
scale_clust = [0.5,1.0,2.0,3.0,4.0,6.0,9.0,12.0,15.0,20.0]
colors_clust = ["white","yellow","orange","green","dodgerblue","red","grey"]
norm = mpl.colors.BoundaryNorm(scale_IMILAST, cmap_IMILAST.N)
colors_IMILAST=["white","yellow","orange","green","deepskyblue","dodgerblue","royalblue","midnightblue","purple","red","grey"]

def n_hemisphere_new():
        ''' Stereographic map, centered on the north pole, covering most of the northern hemisphere

        Returns
        -------
        Basemap
                map projection instance
        '''

        return Basemap(projection='npstere',boundinglat=25,lon_0=0,resolution='c', area_thresh=50000)


ncmask, landmask, maskgrid = metopen("ei.ans.land-sea",q="lsm")
ncoro, oro, orogrid = metopen("ei.ans.orog",q="z")
oro /= 9.81
oro = np.roll(oro,360,-1)
nrtimes = 90*4*(2014-1979 + 1)

grid = copy.copy(maskgrid)
grid.x = maskgrid.x[::3,::3] + 180.0 #[0:60,::]
grid.y = maskgrid.y[::3,::3] #[0:60,::]
grid.dx = maskgrid.dx[::3,::3] #[0:60,::]
grid.dy = maskgrid.dy[::3,::3] #[0:60,::]
grid.oro = maskgrid.oro[::3,::3] #[0:60,::]
grid.nx = 240
grid.ny = 121 #60
conf.gridsize = (121,240) #(60,240)

def calc_density():
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
            min_lat = np.nanmax([-90.0,lats[latidx] - 1.5])
            max_lat = np.nanmin([90.0,lats[latidx] + 1.5])
            min_lon = lons[lonidx] - 1.5
            max_lon = lons[lonidx] + 1.5
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
            diff_lon = 3.0
            #area = np.abs((max_lat - min_lat)*diff_lon)*np.cos((min_lat+max_lat)/2.*np.pi/180.)*111111**2 #
            area = np.nansum(np.abs(grid.dy[areaidxs]*grid.dx[areaidxs]))

            mean_storms[latidx,lonidx] = len(temp_id)/(area*nrtimes)*10**12
            mean_tracks[latidx,lonidx] = len(np.unique(temp_id))/(area*nrtimes)*10**12
            
    return mean_storms, mean_tracks
    
#Construct array with datetimes
dt_array = []
for yidx in range(1979,2019):

    # To get year (integer input) from the user
    # year = int(input("Enter a year: "))
    if ((yidx + 1) % 4) == 0:
        leapyear = True
        nr_times = 366*4 #(whole year) 364 (just winter)
    else:
        leapyear = False
        nr_times = 365*4 #(whole year) 360 (just winter)

    start = dt(yidx, 1, 1, 0) #dt(yidx, 12, 1, 0) (just winter)

    dt_array_temp = np.array([start + td(hours=i*6) for i in range(nr_times)])
    dt_array.extend(dt_array_temp)

def calc_density_radius(distchar = "250km", dist_thresh=250, connect=False, save=False, outfile="Density.npz",
                       calcQuant=False):
    #Define arrays
    lats = np.arange(90,-90.1,-1.5)
    lons = np.arange(0,360,1.5)

    ################################
    # Get indices of storms 
    # so that ids_storms[id] gives the ids in the arrays
    # str_id, str_lon,.. belonging to that specific storm
    #########################
    uniq_ids = np.unique(str_id)
    ids_storms = get_indices_sparse(str_id)
    nrstorms = len(uniq_ids)

    #Define arrays
    storms = np.zeros((len(dt_array),len(lats),len(lons))) #Cyclone centre density
    tracks = np.zeros((len(dt_array),len(lats),len(lons))) #Track density
    lysis  = np.zeros((len(dt_array),len(lats),len(lons))) #Track density
    genesis  = np.zeros((len(dt_array),len(lats),len(lons))) #Track density
    
    mean_storms_seas = np.zeros((4,len(lats),len(lons)))
    mean_tracks_seas = np.zeros((4,len(lats),len(lons)))
    mean_genesis_seas = np.zeros((4,len(lats),len(lons)))
    mean_lysis_seas = np.zeros((4,len(lats),len(lons)))

    #Loop over storm_tracks
    #nr_storms = np.max(str_id)

    if(distchar != "Rossby"):
        dist_temp = dist_thresh

    #Loop over storms
    for strid in range(nrstorms): #clust_idxs: #range(1,nr_storms+1):
        print("Storm: " + str(strid))
        if(connect):
            temp_conn = str_connected[ids_storms[uniq_ids[strid]]]
            temp_lat = str_lat[ids_storms[uniq_ids[strid]]][temp_conn >= 1]
            temp_lon = str_lon[ids_storms[uniq_ids[strid]]][temp_conn >= 1]
            temp_dt  = str_dt[ids_storms[uniq_ids[strid]]][temp_conn >= 1]
        else:
            temp_lat = str_lat[ids_storms[uniq_ids[strid]]]
            temp_lon = str_lon[ids_storms[uniq_ids[strid]]]
            temp_dt  = str_dt[ids_storms[uniq_ids[strid]]]
        #temp_vort = str_vort[ids_storms[uniq_ids[strid]]]
        #temp_maxvort = np.nanmax(str_vort[ids_storms[uniq_ids[strid]]])

        lngth = len(temp_dt)

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
                    
                    difflat = np.abs(temp_lat[tridx] - lats[latidx])
                    difflon = np.abs(temp_lon[tridx] - lons[latidx])
                    if(difflat <= dist_temp/111): 
                        for lonidx in range(len(lons)):
                            #Prevent
                            distest = (difflat**2.0 + difflon**2.0)**0.5*111.0 
                            
                            #Calculate distance to grid point
                            if(distest < dist_temp): #Overestimates the actual distance
                                dist = dist_temp
                            else: #If distance is to big, calculate the real distance
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

                                bool_tracks[latidx,lonidx] = True   

    if(distchar != "Rossby"):
        mul_fac = (500/dist_thresh)**2.0*4.0/np.pi

    #Loop over lons and lats
    if(distchar =="Rossby"):
        mul_fac = np.zeros(conf.gridsize)
        for latidx in range(len(lats)):
            lattemp = np.abs(lats[latidx])
            Rossby_temp = calc_Rossby_radius(lat=lattemp)
            mul_fac[latidx,:] = (500/Rossby_temp)**2.0*4.0/np.pi

    mean_storms = np.nanmean(storms,axis=0)*mul_fac
    mean_tracks = np.nanmean(tracks,axis=0)*mul_fac
    mean_genesis = np.nanmean(genesis,axis=0)*mul_fac
    mean_lysis = np.nanmean(lysis,axis=0)*mul_fac
    
    ## seasonal differences ##
    seasons = ["DJF","MAM","JJA","SON"]
    
    i=0
    for season in seasons:
        months = np.array([x.month for x in dt_array])
        if(season == "DJF"):
            selidxs = (months < 3) | (months >= 12)
        elif(season == "MAM"):
            selidxs = (months < 6) & (months >= 3)
        elif(season == "JJA"):
            selidxs = (months < 9) & (months >= 6)
        elif(season == "SON"):
            selidxs = (months < 12) & (months >= 9)

        mean_storms_seas[i,::] = np.nanmean(storms[selidxs,::],axis=0)*mul_fac
        mean_tracks_seas[i,::] = np.nanmean(tracks[selidxs,::],axis=0)*mul_fac
        mean_genesis_seas[i,::] = np.nanmean(genesis[selidxs,::],axis=0)*mul_fac
        mean_lysis_seas[i,::]   = np.nanmean(lysis[selidxs,::],axis=0)*mul_fac
        i+=1 
        
    ##########################################
    # Median and quantiles of recurrence time
    ##########################################
    if(calcQuant):   
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
        plt.savefig("Plots/Final/Median_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles_DJF[3,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="Median recurrence rate",extend="both", scale=np.arange(1,3.6,0.25)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Median_distance_" + datachar + "_" + distchar + "_DJF.pdf")

        #Quantile
        plt.figure()
        fig.map(recur_quantiles[1,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="10% quantile recurrence rate",extend="both", scale=np.arange(0,2,0.25)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_5_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles[1,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="10% quantile recurrence rate",extend="both", scale=np.arange(0,2,0.25)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_10_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles[2,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="25% quantile recurrence rate",extend="both", scale=np.arange(0.25,2.6,0.25)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_25_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles[4,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="75% quantile recurrence rate",extend="both", scale=np.arange(1.5,5.1,0.5)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_75_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles[5,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="90% quantile recurrence rate",extend="both", scale=np.arange(4,10,1)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_90_distance_" + datachar + "_" + distchar + ".pdf")

        plt.figure()
        fig.map(recur_quantiles[6,::],grid,cmap="PuBuGn",mask=strm_mask,maskcolor="white",title="95% quantile recurrence rate",extend="both", scale=np.arange(6,12,1.0)) #np.arange(2,12.1,1)
        plt.savefig("Plots/Final/Quantile_95_distance_" + datachar + "_" + distchar + ".pdf")        


    ## Optionally saving results
    if(save and calcQuant):
        np.savez(outfile, 
        #storms=storms,
        #tracks=tracks,
        #genesis=genesis,
        #lysis=lysis,
        #yearly means
        mean_storms=mean_storms,
        mean_tracks=mean_tracks,
        mean_genesis=mean_genesis,
        mean_lysis=mean_lysis,
        #Seasonal means
        mean_storms_seas=mean_storms_seas,
        mean_tracks_seas=mean_tracks_seas,
        mean_genesis_seas=mean_genesis_seas,
        mean_lysis_seas=mean_lysis_seas,
        recur_quantiles=recur_quantiles,
        recur_quantiles_DJF=recur_quantiles_DJF)      
    elif(save):
        np.savez(outfile, 
        #storms=storms,
        #tracks=tracks,
        #genesis=genesis,
        #lysis=lysis,
        #yearly means
        mean_storms=mean_storms,
        mean_tracks=mean_tracks,
        mean_genesis=mean_genesis,
        mean_lysis=mean_lysis,
        #Seasonal means
        mean_storms_seas=mean_storms_seas,
        mean_tracks_seas=mean_tracks_seas,
        mean_genesis_seas=mean_genesis_seas,
        mean_lysis_seas=mean_lysis_seas)
        

 
    return mean_storms, mean_tracks, mean_genesis, mean_lysis, mean_storms_seas, mean_tracks_seas, mean_genesis_seas, mean_lysis_seas

#ERA 5 Densities calculation  #######
datachar = "EI"
st_file_ei = "Tracks/Selected_tracks_1979to2018_0101to1231_ei_Globe_Leonidas_with_stationary_all"
str_id, str_nr, str_dt, str_lat, str_lon = read_file(st_file_ei)
str_dt = np.array(str_dt)


#Save density
outfile="Density/Density_" + datachar + "_" + distchar + ".npz"
#np.savez(outfile, mean_storms=mean_storms_ei, mean_tracks=mean_tracks_ei,mean_genesis=mean_genesis_ei,mean_lysis=mean_lysis_ei)

if(calcDensityEI):
    #Density
    mean_storms_ei, mean_tracks_ei, mean_genesis_ei, mean_lysis_ei, mean_storms_seas_ei, mean_tracks_seas_ei, mean_genesis_seas_ei, mean_lysis_seas_ei = calc_density_radius(save=True, outfile=outfile)
else:
    ResultsDensity = np.load(outfile)
    mean_storms_ei = ResultsDensity["mean_storms"]
    mean_tracks_ei = ResultsDensity["mean_tracks"]

#Mean density of ERA5
plt.figure() 
overlays = [fig.map_overlay_contour(mean_storms_ei[::,::]*100,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)] 
fig.map(mean_storms_ei[::,::]*100,grid,m=n_hemisphere_new,overlays=overlays,mask=(oro[0,::3,::3] >= 1000),maskcolor="white",
scale=scale_density,title="Storm Density",cmap=cmoc.tempo,cb_label='% per 1000 km$^2$',extend="both",
save="Mean_Density_EI_final.pdf") #np.arange(2,12.1,1)

plt.figure()
overlays = [fig.map_overlay_contour(mean_storms_ei[::,::]*100,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)] 
fig.map(mean_storms_ei[::,::]*100,grid,m=n_hemisphere_new,overlays=overlays,mask=(oro[0,::3,::3] >= 1000),maskcolor="white",
title="Storm Density",colors=colors_IMILAST, scale=scale_IMILAST, cmap=None,#cmap=cmap_IMILAST,
cb_label='% per 1000 km$^2$',extend="both",save="Mean_Density_EI_imilast_final.pdf") #np.arange(2,12.1,1)

if(calcQuants):
    outfile="Density/Density_" + datachar + "_" + distchar + "_withQuants.npz"
    
    ResultsQuants = calc_density_radius(distchar = "Rossby", calcQuant=True,save=True, outfile=outfile, connect=False)
    
######################################################
# load clustering results
######################################################
formatter =  "{:1.1f}"
infile = Options["outdir"] +  Options["str_result"]  + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"]) + ".npz"
Results = np.load(infile,allow_pickle=True)
 
# All clusters
if(clustchar == "All"):
    sorted_clusters = Results["sorted_clusters"]
elif(clustchar == "nolength"):
    # Only Subclusters (lengtg or nolength)
    sorted_subclusters = Results["sorted_subclusters_nolength"]
    sorted_clusters = sorted(unnest(sorted_subclusters))
elif(clustchar == "length"):
    # Only Subclusters (lengtg or nolength)
    sorted_subclusters = Results["sorted_subclusters_length"]
    sorted_clusters = sorted(unnest(sorted_subclusters))

#Get str_connected
str_connected = Results["str_connected"]
strmidxs = np.unique(str_id)

if(calcDensityClust):
    #Filter clusterd storms
    clststroms = [strm for cluster in sorted_clusters for strm in cluster if len(cluster) >= minstorms and strm in strmidxs]
    clststroms = sorted(clststroms)

    #Subselect arrays
    str_id = np.array([idx for idx in str_id if idx in clststroms])
    str_lat = np.array([lat for (idx, lat) in zip(str_id, str_lat) if idx in clststroms])
    str_lon = np.array([lon for (idx, lon) in zip(str_id, str_lon) if idx in clststroms])
    str_dt = np.array([dattim for (idx, dattim) in zip(str_id, str_dt) if idx in clststroms])
    str_connected = np.array([conn for (idx, conn) in zip(str_id, str_connected) if idx in clststroms])

    #calc_density()
    #Save density
    outfile="Density/Density_" + "_" + distchar + "_" +\
            Options["str_result"]  + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"    ]) +\
            "_Clust_" + clustchar + "_minstorms_" + str(minstorms) + "_connect.npz"

    #mean_storms_clst_era5, mean_tracks_clst_era5 = calc_density()
    mean_storms_clst_ei, mean_tracks_clst_ei, mean_genesis_clst_ei, mean_lysis_clst_ei, mean_storms_clst_seas_ei, mean_tracks_clst_seas_ei, mean_genesis_clst_seas_ei, mean_lysis_clst_seas_ei = calc_density_radius(save=True, outfile=outfile, connect=True)

    #np.savez(outfile, mean_storms=mean_storms_clst_ei, mean_tracks=mean_tracks_clst_ei,mean_genesis=mean_genesis_clst_ei,mean_lysis=mean_lysis_clst_ei)

    #Mean density of EI clustered
    plt.figure() 
    overlays = [fig.map_overlay_contour(mean_storms_ei[::,::]*100,  grid,title=None,cb_label='% per 1000 km$^2$', scale=[15], linewidths=0.75)] 
    fig.map(mean_storms_clst_ei[::,::]*100,grid,m=n_hemisphere_new,overlays=overlays,mask=(oro[0,::3,::3] >= 1500),maskcolor="white",
    title="Storm Density",colors=colors_IMILAST, scale=scale_IMILAST, cmap=None, #cmap=cmap_IMILAST, #np.arange(2,12.1,1)
    cb_label='% per 1000 km$^2$',extend="both",
    save="Mean_Density_ERA5_Clust_" + formatter.format( Options["timthresh"]) + "_" + clustchar + "_minstorms_" + str(minstorms) + "_connect.pdf")  
    
elif(calcDensitySolo):
    #Filter clusterd storms
    solo_storms = [strm for cluster in sorted_clusters for strm in cluster if len(cluster) == 1 and strm in strmidxs]
    solo_storms = sorted(solo_storms)
    
    #Subselect arrays
    str_id = np.array([idx for idx in str_id if idx in solo_storms])
    str_lat = np.array([lat for (idx, lat) in zip(str_id, str_lat) if idx in solo_storms])
    str_lon = np.array([lon for (idx, lon) in zip(str_id, str_lon) if idx in solo_storms])
    str_dt = np.array([dattim for (idx, dattim) in zip(str_id, str_dt) if idx in solo_storms])
    str_connected = np.array([conn for (idx, conn) in zip(str_id, str_connected) if idx in solo_storms])

    #Output file for density
    outfile="Density/Density_" + "_" + distchar + "_" +\
            Options["str_result"]  + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"    ]) +\
            "_Clust_" + clustchar + "_minstorms_" + str(minstorms) + "_solo.npz"
    #mean_storms_clst_era5, mean_tracks_clst_era5 = calc_density()
    mean_storms_solo_ei, mean_tracks_solo_ei, mean_genesis_solo_ei, mean_lysis_solo_ei, mean_storms_solo_seas_ei, mean_tracks_solo_seas_ei, mean_genesis_solo_seas_ei, mean_lysis_solo_seas_ei = calc_density_radius(save=True, outfile=outfile, connect=False)
    
 
