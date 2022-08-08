#!/usr/bin/env python
# -*- encoding: utf-8

from datetime import datetime as dt, timedelta as td
import numpy as np
from numpy import loadtxt
import yaml
import time
from Cluster_functions import read_file, get_indices_sparse

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)
    
#########################
# Load storm tracks 
#########################

#Storm tracks file
st_file = Options["st_file"]
nrskip = Options["nrskip"]

str_id, str_nr, str_dt, str_lat, str_lon = read_file(st_file)
str_pres   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[5],dtype=float)
str_lapl   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[9],dtype=float)

#Convert to an array
str_dt          = np.array(str_dt)
str_connected   = np.zeros(str_dt.shape)
#str_id = str_id - np.nanmin(str_id) + 1

nrstorms = len(np.unique(str_id))
str_connected   = np.zeros(str_dt.shape)
#nrstorms = np.nanmax(str_id)

#########################
# Get indices of storms 
# so that ids_storms[id] gives the ids in the arrays
# str_id, str_lon,.. belonging to that specific storm
#########################
uniq_ids = np.unique(str_id)
ids_storms = get_indices_sparse(str_id)
nrstorms = len(uniq_ids)

#########################
# Load clustering data
#########################
formatter =  "{:1.1f}"
outfile = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"]) + ".npz"
Results = np.load(outfile,allow_pickle=True)
sorted_clusters = Results["sorted_clusters"]

#########################
# Preprocess storm tracks
#########################

#Check which year, month, hemisphere belongs storms to
start = time.time()

yrstorms = np.zeros(nrstorms)
mnstorms = np.zeros(nrstorms)
hemstorms = np.full(nrstorms,"Undefined")
minpres = np.zeros(nrstorms)
mindpdt = np.zeros(nrstorms)
maxlapl = np.zeros(nrstorms)
maxdldt = np.zeros(nrstorms)

firstdt = []
lastdt = []

for strid in range(nrstorms):    
    dt_temp = str_dt[ids_storms[uniq_ids[strid]]]
    lat_temp = str_lat[ids_storms[uniq_ids[strid]]]
    pres_temp = str_pres[ids_storms[uniq_ids[strid]]]
    lapl_temp = str_lapl[ids_storms[uniq_ids[strid]]]

    #Check which winter it belongs to
    tmpyear = dt_temp[0].year
    tmpmonth = dt_temp[0].month
    yrstorms[strid] = tmpyear
    mnstorms[strid] = tmpmonth

    #Save the first and last dt
    firstdt.append(dt_temp[0])
    lastdt.append(dt_temp[-1])

    #Check if the storm is in the NH or SH
    if(np.nanmean(lat_temp) > 0):
        hemstorms[strid] = "NH"
    elif(np.nanmean(lat_temp) < 0):
        hemstorms[strid] = "SH"
        
    #Min pres and dpdt, max lapl and dldt
    minpres[strid] = np.nanmin(pres_temp)
    delta = (dt_temp[1] - dt_temp[0]).total_seconds()/3600
    mindpdt[strid] = np.nanmin(pres_temp[1:] - pres_temp[:-1])/delta
    maxlapl[strid] = np.nanmax(lapl_temp)
    maxdldt[strid] = np.nanmax(lapl_temp[1:] - lapl_temp[:-1])/delta

end = time.time()
firstdt = np.array(firstdt)
lastdt = np.array(lastdt)
print(str(end - start) + " seconds")

#Months of storm, relative to beginning of 1979
mnstorms_rel = (yrstorms - 1979)*12.0 + mnstorms
refdt = dt(1979,1,1,0,0)
diffs = [(x - refdt).total_seconds()/3600 for x in str_dt]   

######################################################
# Statistics 
######################################################
'''
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
'''            
            
######################################################
# Check which basin storms and clusters belong to
######################################################
ext_winter = [(x.month <= 3) | (x.month >= 10) for x in str_dt]
ext_swinter = [(x.month >= 4) & (x.month <= 9) for x in str_dt]

if(Options["checkBasin"]):
    str_basin = np.full(np.nanmax(str_id),"Undefined")
    str_hemis = np.full(np.nanmax(str_id),"Undefined")
    #Check for basin for each storm
    for strm in range(1,np.nanmax(str_id)+1):
        print("Strm " + str(strm))
        selidxs = (str_id == strm) & (str_connected == True)
        lon_temp = str_lon[selidxs] 
        lat_temp = str_lat[selidxs] 
        dt_temp = str_dt[selidxs] 
        wint_temp = (np.array(ext_winter))[selidxs] 
        swint_temp = (np.array(ext_swinter))[selidxs] 
         
        #nr_Atlantic = np.nansum((lon_temp >= 260) & (lon_temp <= 360) & (lat_temp >= 20) & (lat_temp <= 75) & (wint_temp == True) )
        nr_Atlantic = np.nansum(((lon_temp >= 280) | (lon_temp <= 10)) & (lat_temp >= 20) & (lat_temp <= 70) & (wint_temp == True) )
        nr_Pacific = np.nansum((lon_temp >= 120) & (lon_temp <= 240) & (lat_temp >= 20) & (lat_temp <= 70) & (wint_temp == True) )
        nr_nhemis    = np.nansum((lat_temp <= 75) & (lat_temp >= 20) & (wint_temp == True)) 

        nr_sAtlantic = np.nansum(((lon_temp >= 295) | (lon_temp <= 25)) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True) )
        nr_sPacific = np.nansum((lon_temp >= 180) & (lon_temp <= 280) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True) )
        nr_sIndian  = np.nansum((lon_temp >= 25) & (lon_temp <= 115) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True))

        nr_shemis    = np.nansum((lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True)) 

        if( nr_Atlantic/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "Atlantic"

        if( nr_Pacific/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "Pacific"

        if( nr_sAtlantic/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "sAtlantic"

        if( nr_sPacific/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "sPacific"

        if( nr_sIndian/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "sIndian"

        if( nr_nhemis/len(lon_temp) >= 0.5):
            str_hemis[strm -1] = "nhemis"

        if( nr_shemis/len(lon_temp) >= 0.5):
            str_hemis[strm -1] = "shemis"

    #Atlantic clusters
    sorted_clusters_Atlantic = []
    sorted_clusters_Pacific = []
    sorted_clusters_sAtlantic = []
    sorted_clusters_sPacific = []
    sorted_clusters_sIndian = []
    sorted_clusters_shemisphere = []

    lenclust = np.zeros(len(sorted_clusters))
    for clidx in range(len(sorted_clusters)):
        storms_temp = sorted_clusters[clidx]
        lenclust[clidx] = len(storms_temp)
        if(len(storms_temp) > 0): #3
        #sorted_clusters_Atlantic.append(storms_temp)
            if(np.nansum(str_basin[np.array(storms_temp) - 1] == "Atlantic")/len(storms_temp) >= 0.5):
                sorted_clusters_Atlantic.append(storms_temp)

            #sorted_clusters_Atlantic.append(storms_temp)
            if(np.nansum(str_basin[np.array(storms_temp) - 1] == "Pacific")/len(storms_temp) >= 0.5):
                sorted_clusters_Pacific.append(storms_temp)

            if(np.nansum(str_basin[np.array(storms_temp) - 1] == "sAtlantic")/len(storms_temp) >= 0.5):
                sorted_clusters_sAtlantic.append(storms_temp)

            if(np.nansum(str_basin[np.array(storms_temp) - 1] == "sPacific")/len(storms_temp) >= 0.5):
                sorted_clusters_sPacific.append(storms_temp)

            if(np.nansum(str_basin[np.array(storms_temp) - 1] == "sIndian")/len(storms_temp) >= 0.5):
                sorted_clusters_sIndian.append(storms_temp)

            if(np.nansum(str_hemis[np.array(storms_temp) - 1] == "shemis")/len(storms_temp) >= 0.5):
                sorted_clusters_shemisphere.append(storms_temp)

    #np.savez("/Data/gfi/spengler/cwe022/Sorted_Clusters_Areas" + timchar + ".npz",sorted_clusters_Atlantic=sorted_clusters_Atlantic,sorted_clusters_Pacific= sorted_clusters_Pacific, sorted_clusters_sAtlantic=sorted_clusters_sAtlantic,sorted_clusters_sPacific=sorted_clusters_sPacific,sorted_clusters_sIndian=sorted_clusters_sIndian, sorted_clusters_shemisphere= sorted_clusters_shemisphere, str_basin=str_basin,str_hemis=str_hemis)
else:
    Results = np.load("/Data/gfi/spengler/cwe022/Sorted_Clusters_Areas" + timchar + ".npz",allow_pickle=True)
    sorted_clusters_Atlantic = Results["sorted_clusters_Atlantic"]
    sorted_clusters_Pacific = Results["sorted_clusters_Pacific"]
    sorted_clusters_sAtlantic = Results["sorted_clusters_sAtlantic"]
    sorted_clusters_sPacific = Results["sorted_clusters_sPacific"]
    sorted_clusters_sIndian = Results["sorted_clusters_sIndian"]
    sorted_clusters_shemisphere = Results["sorted_clusters_shemisphere"]

######################################################
# Save statistics in a file
######################################################
outfile_stats = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"]) + "_stats.npz"
np.savez(outfile_stats,minpres=minpres,maxlapl=maxlapl,  maxdldt=maxdldt,mindpdt=mindpdt, lengths = lengths, lengthclust= lengthclust, winters=winters,nrclst_wint = nrclst_wint, nrstrm_wint = nrstrm_wint, nrstrmclst_wint = nrstrmclst_wint)