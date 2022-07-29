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
st_file = Options["st_file"]
nrskip = Options["nrskip"]

str_id, str_nr, str_date, str_lat, str_lon = read_file(st_file)
str_dt = dt_array(str_date)

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