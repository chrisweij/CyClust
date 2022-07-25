#!/usr/bin/env python
# -*- encoding: utf-8

from datetime import datetime as dt, timedelta as td
import numpy as np
from numpy import loadtxt
import yaml

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)
    
#########################
# Load storm tracks
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
# Load cluster stats
#########################
formatter =  "{:1.1f}"
#outfile = Options["outdir"] +  Options["str_result"] + formatter.format( Options["distthresh"]) + "_tim_" + formatter.format( Options["timthresh"]) + "_length_" + formatter.format( Options["lngthresh"])
outfile = "Clusters_output/Results_DJF_NH_2011_2012_1.0_tim_36.0_length_1.5"
Results = np.load(outfile)
sorted_clusters = Results["sorted_clusters"]

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
# Save statistics in a file
######################################################
#lengths = lengths, lengthclust= lengthclust, winters=winters,nrclst_wint = nrclst_wint, nrstrm_wint = nrstrm_wint, nrstrmclst_wint = nrstrmclst_wint,