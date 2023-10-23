#!/usr/bin/env python
# -*- encoding: utf-8

from datetime import datetime as dt, timedelta as td
import numpy as np
from numpy import loadtxt
from matplotlib import pyplot as plt
import yaml
import time
import random
from scipy import stats
from Cluster_functions import read_file, get_indices_sparse, unnest

with open("Options.yaml") as f:
    Options = yaml.safe_load(f)
    
reschar = Options["str_result"] + Options["whichClusters"]
    
#########################
# Load storm tracks 
#########################

#Storm tracks file
st_file = Options["st_file"]
nrskip = Options["nrskip"]

str_id, str_nr, str_dt, str_lat, str_lon = read_file(st_file)
str_pres   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[5],dtype=float)
str_lapl   = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[9],dtype=float)
str_radi = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[7],dtype=float)
str_delp = loadtxt(st_file, comments="#", unpack=False,skiprows=nrskip,usecols=[8],dtype=float)

#Convert to an array
str_dt          = np.array(str_dt)
str_month = np.array([x.month for x in str_dt])
str_year = np.array([x.year for x in str_dt])
#str_id = str_id - np.nanmin(str_id) + 1

nrstorms = len(np.unique(str_id))
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
print(outfile)
Results = np.load(outfile,allow_pickle=True)
try:
    if(Options["whichClusters"] == "All"):
        sorted_clusters = Results["sorted_clusters"]
    elif(Options["whichClusters"] == "length"):
        sorted_subclusters_length = Results["sorted_subclusters_length"]
        sorted_clusters =  sorted(unnest(sorted_subclusters_length))
    elif(Options["whichClusters"] == "nolength"):
        sorted_subclusters_nolength = Results["sorted_subclusters_nolength"]
        sorted_clusters =  sorted(unnest(sorted_subclusters_nolength))
except ValueError:
        print("Invalid option for whichClusters")
str_connected = Results['str_connected']

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
ext_winter = np.array([(x.month <= 3) | (x.month >= 10) for x in str_dt])
ext_swinter = np.array([(x.month >= 4) & (x.month <= 9) for x in str_dt])

if(Options["checkBasin"]):
    str_basin = np.full(np.nanmax(str_id),"Undefined")
    str_hemis = np.full(np.nanmax(str_id),"Undefined")
    str_first = np.full(np.nanmax(str_id),dt(9999,12,1,0))
    
    #Check for basin for each storm
    for strm in uniq_ids:
        print("Strm " + str(strm))
        #selidxs = (str_id == strm) #& (str_connected == True)
        lon_temp = str_lon[ids_storms[strm]] 
        lat_temp = str_lat[ids_storms[strm]] 
        dt_temp = str_dt[ids_storms[strm]] 
        wint_temp = ext_winter[ids_storms[strm]]
        swint_temp = ext_swinter[ids_storms[strm]]
        #conn_temp = str_connected[ids_storms[strm]]
        
        
        
        #if(np.any(conn_temp !=0)):
        #    str_first[strm -1] = np.nanmin(dt_temp[conn_temp >0])
         
        nr_EuroAsia = np.nansum((lon_temp >= 10) & (lon_temp <= 120) & (lat_temp >= 20) & (lat_temp <= 75) & (wint_temp == True) )
        nr_America = np.nansum((lon_temp >= 240) & (lon_temp <= 280) & (lat_temp >= 20) & (lat_temp <= 75) & (wint_temp == True) )
        
        nr_Atlantic = np.nansum(((lon_temp >= 280) | (lon_temp <= 10)) & (lat_temp >= 20) & (lat_temp <= 70) & (wint_temp == True) )
        nr_Pacific = np.nansum((lon_temp >= 120) & (lon_temp <= 240) & (lat_temp >= 20) & (lat_temp <= 70) & (wint_temp == True) )
        nr_nhemis    = np.nansum((lat_temp <= 75) & (lat_temp >= 20) & (wint_temp == True)) 

        nr_sAtlantic = np.nansum(((lon_temp >= 295) | (lon_temp <= 25)) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True) )
        nr_sPacific = np.nansum((lon_temp >= 180) & (lon_temp <= 280) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True) )
        nr_sIndian  = np.nansum((lon_temp >= 25) & (lon_temp <= 115) & (lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True))

        nr_shemis    = np.nansum((lat_temp >= -75) & (lat_temp <= -20) & (swint_temp == True)) 

        if( nr_EuroAsia/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "EuroAsia"
            
        if( nr_America/len(lon_temp) >= 0.5):
            str_basin[strm -1] = "America"
            
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

    #np.savez("/Data/gfi/spengler/cwe022/Sorted_Clusters_Areas" + reschar + ".npz",sorted_clusters_Atlantic=sorted_clusters_Atlantic,sorted_clusters_Pacific= sorted_clusters_Pacific, sorted_clusters_sAtlantic=sorted_clusters_sAtlantic,sorted_clusters_sPacific=sorted_clusters_sPacific,sorted_clusters_sIndian=sorted_clusters_sIndian, sorted_clusters_shemisphere= sorted_clusters_shemisphere, str_basin=str_basin,str_hemis=str_hemis)
else:
    Results = np.load("/Data/gfi/spengler/cwe022/Sorted_Clusters_Areas" + reschar + ".npz",allow_pickle=True)
    sorted_clusters_Atlantic = Results["sorted_clusters_Atlantic"]
    sorted_clusters_Pacific = Results["sorted_clusters_Pacific"]
    sorted_clusters_sAtlantic = Results["sorted_clusters_sAtlantic"]
    sorted_clusters_sPacific = Results["sorted_clusters_sPacific"]
    sorted_clusters_sIndian = Results["sorted_clusters_sIndian"]
    sorted_clusters_shemisphere = Results["sorted_clusters_shemisphere"]
    
    
##################################################
# Histogram of length of clusters for each basin
##################################################
lenAtlantic = np.zeros(40)
lengthsAtlantic = np.zeros(len(sorted_clusters_Atlantic))
lenPacific = np.zeros(40)
lengthsPacific = np.zeros(len(sorted_clusters_Pacific))
lenSAtlantic = np.zeros(40)
lengthsSAtlantic = np.zeros(len(sorted_clusters_sAtlantic))
lenSPacific = np.zeros(40)
lengthsSPacific = np.zeros(len(sorted_clusters_sPacific))
lenSIndian = np.zeros(40)
lengthsSIndian = np.zeros(len(sorted_clusters_sIndian))

for k in range(len(sorted_clusters_Atlantic)):
	strmsTemp = sorted_clusters_Atlantic[k]
	lenAtlantic[len(strmsTemp)-1] += 1
	lengthsAtlantic[k] = len(strmsTemp)
avAtlantic = np.nansum(lenAtlantic*range(1,41))/np.nansum(lenAtlantic)

for k in range(len(sorted_clusters_Pacific)):
	strmsTemp = sorted_clusters_Pacific[k]
	lenPacific[len(strmsTemp)-1] += 1
	lengthsPacific[k] = len(strmsTemp)
avPacific = np.nansum(lenPacific*range(1,41))/np.nansum(lenPacific)

for k in range(len(sorted_clusters_sPacific)):
	strmsTemp = sorted_clusters_sPacific[k]
	lenSPacific[len(strmsTemp)-1] += 1
	lengthsSPacific[k] = len(strmsTemp)
avSPacific = np.nansum(lenSPacific*range(1,41))/np.nansum(lenSPacific)

for k in range(len(sorted_clusters_sAtlantic)):
	strmsTemp = sorted_clusters_sAtlantic[k]
	lenSAtlantic[len(strmsTemp)-1] += 1
	lengthsSAtlantic[k] = len(strmsTemp)
avSAtlantic = np.nansum(lenSAtlantic*range(1,41))/np.nansum(lenSAtlantic)

for k in range(len(sorted_clusters_sIndian)):
	strmsTemp = sorted_clusters_sIndian[k]
	lenSIndian[len(strmsTemp)-1] += 1
	lengthsSIndian[k] = len(strmsTemp)
avSIndian = np.nansum(lenSIndian*range(1,41))/np.nansum(lenSIndian)


lenAll = lenAtlantic + lenPacific + lenSAtlantic + lenSPacific + lenSIndian
avAll = np.nansum(lenAll*range(1,41))/np.nansum(lenAll)

plt.figure()
f, axs = plt.subplots(2, 3, sharey=True)
plt.subplots_adjust(hspace=0.45)
axs[0,0].bar(range(1,26,1),lenAll[0:25],log=True,color="dodgerblue")
axs[0,0].set_title("All")
strAll = "{:2.2f}".format(avAll)
axs[0,0].text(0.95, 0.95, strAll,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[0,0].transAxes,
        color='dodgerblue', fontsize=8)
axs[0,1].bar(range(1,26,1),lenAtlantic[0:25],log=True,color="dodgerblue")
axs[0,1].set_title("Atlantic")
strAtl = "{:2.2f}".format(avAtlantic)
axs[0,1].text(0.95, 0.95, strAtl,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[0,1].transAxes,
        color='dodgerblue', fontsize=8)
axs[0,2].bar(range(1,26,1),lenPacific[0:25],log=True,color="dodgerblue")
axs[0,2].set_title("Pacific")
strPac = "{:2.2f}".format(avPacific)
axs[0,2].text(0.95, 0.95, strPac,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[0,2].transAxes,
        color='dodgerblue', fontsize=8)
axs[1,0].bar(range(1,26,1),lenSAtlantic[0:25],log=True,color="dodgerblue")
axs[1,0].set_title("South Atlantic")
strSAtl = "{:2.2f}".format(avSAtlantic)
axs[1,0].text(0.95, 0.95, strSAtl,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[1,0].transAxes,
        color='dodgerblue', fontsize=8)
axs[1,1].bar(range(1,26,1),lenSPacific[0:25],log=True,color="dodgerblue")
axs[1,1].set_title("South Pacific")
strSPac = "{:2.2f}".format(avSPacific)
axs[1,1].text(0.95, 0.95, strSPac,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[1,1].transAxes,
        color='dodgerblue', fontsize=8)
axs[1,2].bar(range(1,26,1),lenSIndian[0:25],log=True,color="dodgerblue")
axs[1,2].set_title("South Indian")
strSInd = "{:2.2f}".format(avSIndian)
axs[1,2].text(0.95, 0.95, strSInd,
        verticalalignment='top', horizontalalignment='right',
        transform=axs[1,2].transAxes,
        color='dodgerblue', fontsize=8)
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Length of cluster")
plt.ylabel("Count")
plt.savefig("ClustersLength_Areas_"+ Options["whichClusters"] + ".pdf")


'''
def discrete_matshow(data):
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', np.nanmax(data) - np.nanmin(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.nanmin(data), np.nanmax(data) + 1))

# generate data
data = np.log10(arrayLengths)
data[np.isinf(data)] = np.nan
plt.figure()
a = np.random.randint(1, 9, size=(10, 10))
discrete_matshow(data)
plt.savefig("Testing.png")
'''


lenAll = lenAtlantic + lenPacific + lenSAtlantic + lenSPacific + lenSIndian
arrayLengths = np.zeros((6,24))
arrayLengths[0,::] = lenAll[1:25]
arrayLengths[1,::] = lenAtlantic[1:25]
arrayLengths[2,::] = lenPacific[1:25]
arrayLengths[3,::] = lenSAtlantic[1:25]
arrayLengths[4,::] = lenSPacific[1:25]
arrayLengths[5,::] = lenSIndian[1:25]

fig = plt.figure(figsize=(12,2.67))
ax = plt.subplot(1, 1, 1)
data = np.log10(arrayLengths)
data[np.isinf(data)] = np.nan
plt.rcParams['font.sans-serif'] = ['Verdana'] #,'Arial','Verdana','Helvetica']
#["black",'xkcd:baby puke green','firebrick','darkblue','xkcd:rich purple','xkcd:pumpkin']
ax.set_prop_cycle(color=["black",'xkcd:deep red','darkblue','xkcd:pumpkin','xkcd:sky blue','xkcd:grass green'],
linewidth=[1.6,0.9,0.9,0.9,0.9,0.9],linestyle=['-','--','--','-','-','-'])
ax.plot(np.arange(2,26,1),np.transpose(data))
ax.set_xticks(np.arange(2,21,1))
yvals = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000]
yvals_log10 = np.log10(yvals)
ax.set_yticks(yvals_log10) #np.arange(0, 5)
ax.set_yticklabels(yvals) #10.0**np.arange(0, 5)
ax.set_xlabel("Length of cluster")
ax.set_xlim(2,21)
ax.set_ylim(-.2,np.log10(5000))
ax.grid(linewidth = 0.5)

# Hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

plt.legend(labels=["All","North Atlantic","North Pacific","South Atlantic","South Pacific","South Indian"])
plt.tight_layout()
plt.savefig("ClustersLength_Areas_lines_"+ Options["whichClusters"] + ".pdf")
############################################################################

############################################################################
# Determine Histogram of intensity/radius measures of clusters for each basin
############################################################################
#Subselect cyclone families and non-cyclone families
storms_Atlantic_solo = [item for sublist in sorted_clusters_Atlantic for item in sublist if (len(sublist) <= 3)]
storms_Pacific_solo = [item for sublist in sorted_clusters_Pacific for item in sublist if (len(sublist) <= 3)]
storms_sAtlantic_solo = [item for sublist in sorted_clusters_sAtlantic for item in sublist if (len(sublist) <= 3)]
storms_sPacific_solo = [item for sublist in sorted_clusters_sPacific for item in sublist if (len(sublist) <= 3)]
storms_sIndian_solo = [item for sublist in sorted_clusters_sIndian for item in sublist if (len(sublist) <= 3)]

storms_Atlantic_clust = [item for sublist in sorted_clusters_Atlantic for item in sublist if (len(sublist) > 3)]
storms_Pacific_clust = [item for sublist in sorted_clusters_Pacific for item in sublist if (len(sublist) > 3)]
storms_sAtlantic_clust = [item for sublist in sorted_clusters_sAtlantic for item in sublist if (len(sublist) > 3)]
storms_sPacific_clust = [item for sublist in sorted_clusters_sPacific for item in sublist if (len(sublist) > 3)]
storms_sIndian_clust = [item for sublist in sorted_clusters_sIndian for item in sublist if (len(sublist) > 3)]

storms_Atlantic = [item for sublist in sorted_clusters_Atlantic for item in sublist]
length_Atlantic = [len(sublist) for sublist in sorted_clusters_Atlantic for item in sublist]
storms_Pacific = [item for sublist in sorted_clusters_Pacific for item in sublist]
length_Pacific = [len(sublist) for sublist in sorted_clusters_Pacific for item in sublist]
storms_sAtlantic = [item for sublist in sorted_clusters_sAtlantic for item in sublist]
length_sAtlantic = [len(sublist) for sublist in sorted_clusters_sAtlantic for item in sublist]
storms_sPacific = [item for sublist in sorted_clusters_sPacific for item in sublist]
length_sPacific = [len(sublist) for sublist in sorted_clusters_sPacific for item in sublist]
storms_sIndian = [item for sublist in sorted_clusters_sIndian for item in sublist]
length_sIndian = [len(sublist) for sublist in sorted_clusters_sIndian for item in sublist]

#Define results arrays
pres_Atlantic = np.zeros(len(storms_Atlantic))
lapl_Atlantic = np.zeros(len(storms_Atlantic))
pres_Pacific = np.zeros(len(storms_Pacific))
lapl_Pacific = np.zeros(len(storms_Pacific))
pres_sAtlantic = np.zeros(len(storms_sAtlantic))
lapl_sAtlantic = np.zeros(len(storms_sAtlantic))
pres_sPacific = np.zeros(len(storms_sPacific))
lapl_sPacific = np.zeros(len(storms_sPacific))
pres_sIndian = np.zeros(len(storms_sIndian))
lapl_sIndian = np.zeros(len(storms_sIndian))
month_Atlantic = np.zeros(len(storms_Atlantic))
month_Pacific = np.zeros(len(storms_Pacific))
month_sAtlantic = np.zeros(len(storms_sAtlantic))
month_sPacific = np.zeros(len(storms_sPacific))
month_sIndian = np.zeros(len(storms_sIndian))
year_Atlantic = np.zeros(len(storms_Atlantic))
year_Pacific = np.zeros(len(storms_Pacific))
year_sAtlantic = np.zeros(len(storms_sAtlantic))
year_sPacific = np.zeros(len(storms_sPacific))
year_sIndian = np.zeros(len(storms_sIndian))
radi_Atlantic = np.zeros(len(storms_Atlantic))
radi_Pacific = np.zeros(len(storms_Pacific))
radi_sAtlantic = np.zeros(len(storms_sAtlantic))
radi_sPacific = np.zeros(len(storms_sPacific))
radi_sIndian = np.zeros(len(storms_sIndian))


pres_Atlantic_clust = np.zeros(len(storms_Atlantic_clust))
lapl_Atlantic_clust = np.zeros(len(storms_Atlantic_clust))
pres_Pacific_clust = np.zeros(len(storms_Pacific_clust))
lapl_Pacific_clust = np.zeros(len(storms_Pacific_clust))
pres_sAtlantic_clust = np.zeros(len(storms_sAtlantic_clust))
lapl_sAtlantic_clust = np.zeros(len(storms_sAtlantic_clust))
pres_sPacific_clust = np.zeros(len(storms_sPacific_clust))
lapl_sPacific_clust = np.zeros(len(storms_sPacific_clust))
pres_sIndian_clust = np.zeros(len(storms_sIndian_clust))
lapl_sIndian_clust = np.zeros(len(storms_sIndian_clust))

pres_Atlantic_solo = np.zeros(len(storms_Atlantic_solo))
lapl_Atlantic_solo = np.zeros(len(storms_Atlantic_solo))
pres_Pacific_solo = np.zeros(len(storms_Pacific_solo))
lapl_Pacific_solo = np.zeros(len(storms_Pacific_solo))
pres_sAtlantic_solo = np.zeros(len(storms_sAtlantic_solo))
lapl_sAtlantic_solo = np.zeros(len(storms_sAtlantic_solo))
pres_sPacific_solo = np.zeros(len(storms_sPacific_solo))
lapl_sPacific_solo = np.zeros(len(storms_sPacific_solo))
pres_sIndian_solo = np.zeros(len(storms_sIndian_solo))
lapl_sIndian_solo = np.zeros(len(storms_sIndian_solo))

radi_Atlantic_clust = np.zeros(len(storms_Atlantic_clust))
radi_Pacific_clust = np.zeros(len(storms_Pacific_clust))
radi_sAtlantic_clust = np.zeros(len(storms_sAtlantic_clust))
radi_sPacific_clust = np.zeros(len(storms_sPacific_clust))
radi_sIndian_clust = np.zeros(len(storms_sIndian_clust))

radi_Atlantic_solo = np.zeros(len(storms_Atlantic_solo))
radi_Pacific_solo = np.zeros(len(storms_Pacific_solo))
radi_sAtlantic_solo = np.zeros(len(storms_sAtlantic_solo))
radi_sPacific_solo = np.zeros(len(storms_sPacific_solo))
radi_sIndian_solo = np.zeros(len(storms_sIndian_solo))

month_Atlantic_clust = np.zeros(len(storms_Atlantic_clust))
month_Pacific_clust = np.zeros(len(storms_Pacific_clust))
month_sAtlantic_clust = np.zeros(len(storms_sAtlantic_clust))
month_sPacific_clust = np.zeros(len(storms_sPacific_clust))
month_sIndian_clust = np.zeros(len(storms_sIndian_clust))

month_Atlantic_solo = np.zeros(len(storms_Atlantic_solo))
month_Pacific_solo = np.zeros(len(storms_Pacific_solo))
month_sAtlantic_solo = np.zeros(len(storms_sAtlantic_solo))
month_sPacific_solo = np.zeros(len(storms_sPacific_solo))
month_sIndian_solo = np.zeros(len(storms_sIndian_solo))

year_Atlantic_clust = np.zeros(len(storms_Atlantic_clust))
year_Pacific_clust = np.zeros(len(storms_Pacific_clust))
year_sAtlantic_clust = np.zeros(len(storms_sAtlantic_clust))
year_sPacific_clust = np.zeros(len(storms_sPacific_clust))
year_sIndian_clust = np.zeros(len(storms_sIndian_clust))

year_Atlantic_solo = np.zeros(len(storms_Atlantic_solo))
year_Pacific_solo = np.zeros(len(storms_Pacific_solo))
year_sAtlantic_solo = np.zeros(len(storms_sAtlantic_solo))
year_sPacific_solo = np.zeros(len(storms_sPacific_solo))
year_sIndian_solo = np.zeros(len(storms_sIndian_solo))

#Intensification
int_Atlantic_clust = []
int_Pacific_clust = []
int_sAtlantic_clust = []
int_sPacific_clust = []
int_sIndian_clust = []

int_Atlantic_solo = []
int_Pacific_solo = []
int_sAtlantic_solo = []
int_sPacific_solo = []
int_sIndian_solo = []

for strm in range(len(storms_Atlantic)):
	pres_Atlantic[strm] = np.nanmin(str_pres[ids_storms[storms_Atlantic[strm]]])
	lapl_Atlantic[strm] = np.nanmax(str_lapl[ids_storms[storms_Atlantic[strm]]])
	radi_Atlantic[strm] = np.nanmean(str_radi[ids_storms[storms_Atlantic[strm]]])
	month_Atlantic[strm] = stats.mode(str_month[ids_storms[storms_Atlantic[strm]]])[0]
	year_Atlantic[strm] = stats.mode(str_year[ids_storms[storms_Atlantic[strm]]])[0]
	
	pres_temp = str_pres[ids_storms[storms_Atlantic[strm]]][1:] - str_pres[ids_storms[storms_Atlantic[strm]]][:-1]
	
	int_Atlantic_clust.append(pres_temp)

for strm in range(len(storms_Atlantic_clust)):
	pres_Atlantic_clust[strm] = np.nanmin(str_pres[ids_storms[storms_Atlantic_clust[strm]]])
	lapl_Atlantic_clust[strm] = np.nanmax(str_lapl[ids_storms[storms_Atlantic_clust[strm]]])
	radi_Atlantic_clust[strm] = np.nanmean(str_radi[ids_storms[storms_Atlantic_clust[strm]]])
	month_Atlantic_clust[strm] = stats.mode(str_month[ids_storms[storms_Atlantic_clust[strm]]])[0]
	year_Atlantic_clust[strm] = stats.mode(str_year[ids_storms[storms_Atlantic_clust[strm]]])[0]

for strm in range(len(storms_Atlantic_solo)):
	pres_Atlantic_solo[strm] = np.nanmin(str_pres[ids_storms[storms_Atlantic_solo[strm]]])
	lapl_Atlantic_solo[strm] = np.nanmax(str_lapl[ids_storms[storms_Atlantic_solo[strm]]])
	radi_Atlantic_solo[strm] = np.nanmean(str_radi[ids_storms[storms_Atlantic_solo[strm]]])
	month_Atlantic_solo[strm] = stats.mode(str_month[ids_storms[storms_Atlantic_solo[strm]]])[0]
	year_Atlantic_solo[strm] = stats.mode(str_year[ids_storms[storms_Atlantic_solo[strm]]])[0]

for strm in range(len(storms_Pacific)):
	pres_Pacific[strm] = np.nanmin(str_pres[ids_storms[storms_Pacific[strm]]])
	lapl_Pacific[strm] = np.nanmax(str_lapl[ids_storms[storms_Pacific[strm]]])
	radi_Pacific[strm] = np.nanmean(str_radi[ids_storms[storms_Pacific[strm]]])
	month_Pacific[strm] = stats.mode(str_month[ids_storms[storms_Pacific[strm]]])[0]
	year_Pacific[strm] = stats.mode(str_year[ids_storms[storms_Pacific[strm]]])[0]

for strm in range(len(storms_Pacific_clust)):
	pres_Pacific_clust[strm] = np.nanmin(str_pres[ids_storms[storms_Pacific_clust[strm]]])
	lapl_Pacific_clust[strm] = np.nanmax(str_lapl[ids_storms[storms_Pacific_clust[strm]]])
	radi_Pacific_clust[strm] = np.nanmean(str_radi[ids_storms[storms_Pacific_clust[strm]]])
	month_Pacific_clust[strm] = stats.mode(str_month[ids_storms[storms_Pacific_clust[strm]]])[0]
	year_Pacific_clust[strm] = stats.mode(str_year[ids_storms[storms_Pacific_clust[strm]]])[0]

for strm in range(len(storms_Pacific_solo)):
	pres_Pacific_solo[strm] = np.nanmin(str_pres[ids_storms[storms_Pacific_solo[strm]]])
	lapl_Pacific_solo[strm] = np.nanmax(str_lapl[ids_storms[storms_Pacific_solo[strm]]])
	radi_Pacific_solo[strm] = np.nanmean(str_radi[ids_storms[storms_Pacific_solo[strm]]])
	month_Pacific_solo[strm] = stats.mode(str_month[ids_storms[storms_Pacific_solo[strm]]])[0]
	year_Pacific_solo[strm] = stats.mode(str_year[ids_storms[storms_Pacific_solo[strm]]])[0]

for strm in range(len(storms_sAtlantic)):
	pres_sAtlantic[strm] = np.nanmin(str_pres[ids_storms[storms_sAtlantic[strm]]])
	lapl_sAtlantic[strm] = np.nanmax(str_lapl[ids_storms[storms_sAtlantic[strm]]])
	radi_sAtlantic[strm] = np.nanmean(str_radi[ids_storms[storms_sAtlantic[strm]]])
	month_sAtlantic[strm] = stats.mode(str_month[ids_storms[storms_sAtlantic[strm]]])[0]
	year_sAtlantic[strm] = stats.mode(str_year[ids_storms[storms_sAtlantic[strm]]])[0]

for strm in range(len(storms_sAtlantic_clust)):
	pres_sAtlantic_clust[strm] = np.nanmin(str_pres[ids_storms[storms_sAtlantic_clust[strm]]])
	lapl_sAtlantic_clust[strm] = np.nanmax(str_lapl[ids_storms[storms_sAtlantic_clust[strm]]])
	radi_sAtlantic_clust[strm] = np.nanmean(str_radi[ids_storms[storms_sAtlantic_clust[strm]]])
	month_sAtlantic_clust[strm] = stats.mode(str_month[ids_storms[storms_sAtlantic_clust[strm]]])[0]
	year_sAtlantic_clust[strm] = stats.mode(str_year[ids_storms[storms_sAtlantic_clust[strm]]])[0]

for strm in range(len(storms_sAtlantic_solo)):
	pres_sAtlantic_solo[strm] = np.nanmin(str_pres[ids_storms[storms_sAtlantic_solo[strm]]])
	lapl_sAtlantic_solo[strm] = np.nanmax(str_lapl[ids_storms[storms_sAtlantic_solo[strm]]])
	radi_sAtlantic_solo[strm] = np.nanmean(str_radi[ids_storms[storms_sAtlantic_solo[strm]]])
	month_sAtlantic_solo[strm] = stats.mode(str_month[ids_storms[storms_sAtlantic_solo[strm]]])[0]
	year_sAtlantic_solo[strm] = stats.mode(str_year[ids_storms[storms_sAtlantic_solo[strm]]])[0]

for strm in range(len(storms_sPacific)):
	pres_sPacific[strm] = np.nanmin(str_pres[ids_storms[storms_sPacific[strm]]])
	lapl_sPacific[strm] = np.nanmax(str_lapl[ids_storms[storms_sPacific[strm]]])
	radi_sPacific[strm] = np.nanmean(str_radi[ids_storms[storms_sPacific[strm]]])
	month_sPacific[strm] = stats.mode(str_month[ids_storms[storms_sPacific[strm]]])[0]
	year_sPacific[strm] = stats.mode(str_year[ids_storms[storms_sPacific[strm]]])[0]

for strm in range(len(storms_sPacific_clust)):
	pres_sPacific_clust[strm] = np.nanmin(str_pres[ids_storms[storms_sPacific_clust[strm]]])
	lapl_sPacific_clust[strm] = np.nanmax(str_lapl[ids_storms[storms_sPacific_clust[strm]]])
	radi_sPacific_clust[strm] = np.nanmean(str_radi[ids_storms[storms_sPacific_clust[strm]]])
	month_sPacific_clust[strm] = stats.mode(str_month[ids_storms[storms_sPacific_clust[strm]]])[0]
	year_sPacific_clust[strm] = stats.mode(str_year[ids_storms[storms_sPacific_clust[strm]]])[0]

for strm in range(len(storms_sPacific_solo)):
	pres_sPacific_solo[strm] = np.nanmin(str_pres[ids_storms[storms_sPacific_solo[strm]]])
	lapl_sPacific_solo[strm] = np.nanmax(str_lapl[ids_storms[storms_sPacific_solo[strm]]])
	radi_sPacific_solo[strm] = np.nanmean(str_radi[ids_storms[storms_sPacific_solo[strm]]])
	month_sPacific_solo[strm] = stats.mode(str_month[ids_storms[storms_sPacific_solo[strm]]])[0]
	year_sPacific_solo[strm] = stats.mode(str_year[ids_storms[storms_sPacific_solo[strm]]])[0]

for strm in range(len(storms_sIndian)):
	pres_sIndian[strm] = np.nanmin(str_pres[ids_storms[storms_sIndian[strm]]])
	lapl_sIndian[strm] = np.nanmax(str_lapl[ids_storms[storms_sIndian[strm]]])
	radi_sIndian[strm] = np.nanmean(str_radi[ids_storms[storms_sIndian[strm]]])
	month_sIndian[strm] = stats.mode(str_month[ids_storms[storms_sIndian[strm]]])[0]
	year_sIndian[strm] = stats.mode(str_year[ids_storms[storms_sIndian[strm]]])[0]

for strm in range(len(storms_sIndian_clust)):
	pres_sIndian_clust[strm] = np.nanmin(str_pres[ids_storms[storms_sIndian_clust[strm]]])
	lapl_sIndian_clust[strm] = np.nanmax(str_lapl[ids_storms[storms_sIndian_clust[strm]]])
	radi_sIndian_clust[strm] = np.nanmean(str_radi[ids_storms[storms_sIndian_clust[strm]]])
	month_sIndian_clust[strm] = stats.mode(str_month[ids_storms[storms_sIndian_clust[strm]]])[0]
	year_sIndian_clust[strm] = stats.mode(str_year[ids_storms[storms_sIndian_clust[strm]]])[0]

for strm in range(len(storms_sIndian_solo)):
	pres_sIndian_solo[strm] = np.nanmin(str_pres[ids_storms[storms_sIndian_solo[strm]]])
	lapl_sIndian_solo[strm] = np.nanmax(str_lapl[ids_storms[storms_sIndian_solo[strm]]])
	radi_sIndian_solo[strm] = np.nanmean(str_radi[ids_storms[storms_sIndian_solo[strm]]])
	month_sIndian_solo[strm] = stats.mode(str_month[ids_storms[storms_sIndian_solo[strm]]])[0]
	year_sIndian_solo[strm] = stats.mode(str_year[ids_storms[storms_sIndian_solo[strm]]])[0]

pres_All = np.hstack((pres_Atlantic,pres_Pacific,pres_sAtlantic,pres_sPacific,pres_sIndian))
length_All = np.hstack((length_Atlantic,length_Pacific,length_sAtlantic,length_sPacific,length_sIndian))
pres_All_clust = np.hstack((pres_Atlantic_clust,pres_Pacific_clust,pres_sAtlantic_clust,pres_sPacific_clust,pres_sIndian_clust))
pres_All_solo = np.hstack((pres_Atlantic_solo,pres_Pacific_solo,pres_sAtlantic_solo,pres_sPacific_solo,pres_sIndian_solo))

lapl_All = np.hstack((lapl_Atlantic,lapl_Pacific,lapl_sAtlantic,lapl_sPacific,lapl_sIndian))
lapl_All_clust = np.hstack((lapl_Atlantic_clust,lapl_Pacific_clust,lapl_sAtlantic_clust,lapl_sPacific_clust,lapl_sIndian_clust))
lapl_All_solo = np.hstack((lapl_Atlantic_solo,lapl_Pacific_solo,lapl_sAtlantic_solo,lapl_sPacific_solo,lapl_sIndian_solo))

radi_All_clust = np.hstack((radi_Atlantic_clust,radi_Pacific_clust,radi_sAtlantic_clust,radi_sPacific_clust,radi_sIndian_clust))
radi_All_solo = np.hstack((radi_Atlantic_solo,radi_Pacific_solo,radi_sAtlantic_solo,radi_sPacific_solo,radi_sIndian_solo))

month_All = np.hstack(((month_Atlantic+6)%12,(month_Pacific+6)%12,month_sAtlantic,month_sPacific,month_sIndian))
month_All_clust = np.hstack((month_Atlantic_clust,month_Pacific_clust,month_sAtlantic_clust,month_sPacific_clust,month_sIndian_clust))
month_All_solo = np.hstack((month_Atlantic_solo,month_Pacific_solo,month_sAtlantic_solo,month_sPacific_solo,month_sIndian_solo))

year_All_clust = np.hstack((year_Atlantic_clust,year_Pacific_clust,year_sAtlantic_clust,year_sPacific_clust,year_sIndian_clust))
year_All_solo = np.hstack((year_Atlantic_solo,year_Pacific_solo,year_sAtlantic_solo,year_sPacific_solo,year_sIndian_solo))


#################
# PDF with min. pressure, only strongest cyclone per family
#################
laplstep = .5
laplBins_mid = np.arange(2.0,9.1,laplstep)
laplBins = np.arange(1.75,9.26,laplstep)

pdfAll = np.zeros((10,len(laplBins_mid)))
pdfAtlantic = np.zeros((10,len(laplBins_mid)))
pdfPacific = np.zeros((10,len(laplBins_mid)))
pdfsAtlantic = np.zeros((10,len(laplBins_mid)))
pdfsPacific = np.zeros((10,len(laplBins_mid)))
pdfsIndian = np.zeros((10,len(laplBins_mid)))

quantsAll = np.zeros((10,3))
quantsAtlantic = np.zeros((10,3))
quantsPacific = np.zeros((10,3))
quantsSAtlantic = np.zeros((10,3))
quantsSPacific = np.zeros((10,3))
quantsSIndian = np.zeros((10,3))

quantsAll_Expect = np.zeros((10,3))
quantsAtlantic_Expect = np.zeros((10,3))
quantsPacific_Expect = np.zeros((10,3))
quantsSAtlantic_Expect = np.zeros((10,3))
quantsSPacific_Expect = np.zeros((10,3))
quantsSIndian_Expect = np.zeros((10,3))

'''
for l in range(10):
	if(l < 9):
		laplTempAtlantic = [np.nanmax(lapl_Atlantic[np.array([np.where(np.array(storms_Atlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Atlantic if len(sublist) == l + 1]
		laplTempPacific = [np.nanmax(lapl_Pacific[np.array([np.where(np.array(storms_Pacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Pacific if len(sublist) == l + 1]
		laplTempsAtlantic = [np.nanmax(lapl_sAtlantic[np.array([np.where(np.array(storms_sAtlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sAtlantic if len(sublist) == l + 1]
		laplTempsPacific = [np.nanmax(lapl_sPacific[np.array([np.where(np.array(storms_sPacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sPacific if len(sublist) == l + 1]
		laplTempsIndian = [np.nanmax(lapl_sIndian[np.array([np.where(np.array(storms_sIndian) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sIndian if len(sublist) == l + 1]

	else:
		laplTempAtlantic = [np.nanmax(lapl_Atlantic[np.array([np.where(np.array(storms_Atlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Atlantic if len(sublist) >= l + 1]
		laplTempPacific = [np.nanmax(lapl_Pacific[np.array([np.where(np.array(storms_Pacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Pacific if len(sublist) >= l + 1]
		laplTempsAtlantic = [np.nanmax(lapl_sAtlantic[np.array([np.where(np.array(storms_sAtlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sAtlantic if len(sublist) >= l + 1]
		laplTempsPacific = [np.nanmax(lapl_sPacific[np.array([np.where(np.array(storms_sPacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sPacific if len(sublist) >= l + 1]
		laplTempsIndian = [np.nanmax(lapl_sIndian[np.array([np.where(np.array(storms_sIndian) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sIndian if len(sublist) >= l + 1]
	laplTempAll = np.hstack((laplTempAtlantic,laplTempPacific,laplTempsAtlantic,laplTempsPacific,laplTempsIndian))
	pdfAll[l,::] = np.histogram(laplTempAll,bins=laplBins,normed=True)[0]*laplstep
	pdfAtlantic[l,::] = np.histogram(laplTempAtlantic,bins=laplBins,normed=True)[0]*laplstep
	pdfPacific[l,::] = np.histogram(laplTempPacific,bins=laplBins,normed=True)[0]*laplstep
	pdfsAtlantic[l,::] = np.histogram(laplTempsAtlantic,bins=laplBins,normed=True)[0]*laplstep
	pdfsPacific[l,::] = np.histogram(laplTempsPacific,bins=laplBins,normed=True)[0]*laplstep
	pdfsIndian[l,::] = np.histogram(laplTempsIndian,bins=laplBins,normed=True)[0]*laplstep
	quantsAll[l,::] = np.quantile(laplTempAll,[0.1,0.5,0.9])	
	quantsAtlantic[l,::] =  np.quantile(laplTempAtlantic,[0.1,0.5,0.9])
	quantsPacific[l,::] = np.quantile(laplTempPacific,[0.1,0.5,0.9])
	quantsSAtlantic[l,::] = np.quantile(laplTempsAtlantic,[0.1,0.5,0.9])
	quantsSPacific[l,::] = np.quantile(laplTempsPacific,[0.1,0.5,0.9])
	quantsSIndian[l,::] = np.quantile(laplTempsIndian,[0.1,0.5,0.9])


	TempAll = np.zeros(10000)
	TempAtlantic = np.zeros(10000)
	TempPacific = np.zeros(10000)
	TempsAtlantic = np.zeros(10000)
	TempsPacific = np.zeros(10000)
	TempsIndian = np.zeros(10000)

	for i in range(10000):
		TempAll[i] =np.nanmax(random.sample(list(lapl_All),l+1))
		TempAtlantic[i] =np.nanmax(random.sample(list(lapl_Atlantic),l+1))
		TempPacific[i] =np.nanmax(random.sample(list(lapl_Pacific),l+1))
		TempsAtlantic[i] =np.nanmax(random.sample(list(lapl_sAtlantic),l+1))
		TempsPacific[i] =np.nanmax(random.sample(list(lapl_sPacific),l+1))
		TempsIndian[i] =np.nanmax(random.sample(list(lapl_sIndian),l+1))

	quantsAll_Expect[l,::] = np.quantile(TempAll,[0.1,0.5,0.9])	
	quantsAtlantic_Expect[l,::] =  np.quantile(TempAtlantic,[0.1,0.5,0.9])
	quantsPacific_Expect[l,::] = np.quantile(TempPacific,[0.1,0.5,0.9])
	quantsSAtlantic_Expect[l,::] = np.quantile(TempsAtlantic,[0.1,0.5,0.9])
	quantsSPacific_Expect[l,::] = np.quantile(TempsPacific,[0.1,0.5,0.9])
	quantsSIndian_Expect[l,::] = np.quantile(TempsIndian,[0.1,0.5,0.9])





#laplBins = np.arange(920,1025.1,2.5)
plt.figure()
f, axs = plt.subplots(2, 3, sharey=False)
f.set_figheight(4.0)
f.set_figwidth(10.0)
plt.subplots_adjust(hspace=0.4,wspace=.25)
im = axs[0,0].contourf(laplBins_mid,range(1,11),pdfAll,cmap="RdBu_r")
axs[0,0].plot(quantsAll[:,1],range(1,11),color="k",linewidth=1.5)
axs[0,0].plot(quantsAll_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[0,0].plot(quantsAll[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[0,0].plot(quantsAll[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[0,0].set_title("All")

axs[0,1].contourf(laplBins_mid,range(1,11),pdfAtlantic,cmap="RdBu_r")
axs[0,1].set_title("Atlantic")
axs[0,1].plot(quantsAtlantic[:,1],range(1,11),color="k",linewidth=1.5)
axs[0,1].plot(quantsAtlantic_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[0,1].plot(quantsAtlantic[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[0,1].plot(quantsAtlantic[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
strAtl = "{:2.2f}".format(np.nanmean(lapl_Atlantic_solo))
strAtlclust = "{:2.2f}".format(np.nanmean(lapl_Atlantic_clust))

axs[0,2].contourf(laplBins_mid,range(1,11),pdfPacific,cmap="RdBu_r")
axs[0,2].set_title("Pacific")
axs[0,2].plot(quantsPacific[:,1],range(1,11),color="k",linewidth=1.5)
axs[0,2].plot(quantsPacific_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[0,2].plot(quantsPacific[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[0,2].plot(quantsPacific[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
strPac = "{:2.2f}".format(np.nanmean(lapl_Pacific_solo))
strPacclust = "{:2.2f}".format(np.nanmean(lapl_Pacific_clust))

axs[1,0].contourf(laplBins_mid,range(1,11),pdfsAtlantic,cmap="RdBu_r")
axs[1,0].set_title("South Atlantic")
axs[1,0].plot(quantsSAtlantic[:,1],range(1,11),color="k",linewidth=1.5)
axs[1,0].plot(quantsSAtlantic_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[1,0].plot(quantsSAtlantic[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[1,0].plot(quantsSAtlantic[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
strSAtl = "{:2.2f}".format(np.nanmean(lapl_sAtlantic_solo))
strSAtlclust = "{:2.2f}".format(np.nanmean(lapl_sAtlantic_clust))

axs[1,1].contourf(laplBins_mid,range(1,11),pdfsPacific,cmap="RdBu_r")
axs[1,1].set_title("South Pacific")
axs[1,1].plot(quantsSPacific[:,1],range(1,11),color="k",linewidth=1.5)
axs[1,1].plot(quantsSPacific_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[1,1].plot(quantsSPacific[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[1,1].plot(quantsSPacific[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
strSPac = "{:2.2f}".format(np.nanmean(lapl_sPacific_solo))
strSPacclust = "{:2.2f}".format(np.nanmean(lapl_sPacific_clust))

axs[1,2].contourf(laplBins_mid,range(1,11),pdfsIndian,cmap="RdBu_r")
axs[1,2].set_title("South Indian")
axs[1,2].plot(quantsSIndian[:,1],range(1,11),color="k",linewidth=1.5)
axs[1,2].plot(quantsSIndian_Expect[:,1],range(1,11),color="blue",linewidth=1.5)
axs[1,2].plot(quantsSIndian[:,0],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
axs[1,2].plot(quantsSIndian[:,2],range(1,11),color="k",linewidth=1.5,linestyle="dashed")
strSInd = "{:2.2f}".format(np.nanmean(lapl_sIndian_solo))
strSIndclust = "{:2.2f}".format(np.nanmean(lapl_sIndian_clust))

# Add color bar
f.subplots_adjust(right=0.875)
cbar_ax = f.add_axes([0.9, 0.15, 0.02, 0.7])
f.colorbar(im, cax=cbar_ax)

# Add xlabel and ylabel
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Maximum Laplacian of storm")
plt.ylabel("Count")

plt.savefig("MaxLaplPdfAreas" + reschar + "_StrongestCyclones_"+ Options["whichClusters"] + ".pdf")

'''
### BOXPLOTS ####
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(bbox_to_anchor=(1.1, 0.75))
 


### BOXPLOTS ####
boxplotsAll = []
boxplotsAtlantic = []
boxplotsPacific = []
boxplotsSAtlantic = []
boxplotsSPacific = []
boxplotsSIndian = []

quantsAll_Expect = np.zeros((10,3))
quantsAtlantic_Expect = np.zeros((10,3))
quantsPacific_Expect = np.zeros((10,3))
quantsSAtlantic_Expect = np.zeros((10,3))
quantsSPacific_Expect = np.zeros((10,3))
quantsSIndian_Expect = np.zeros((10,3))

maxstorms = 10
# the list named ticks, summarizes or groups
ticks = ['Solo', '2', '3','4','5','6','7','8','9','10+']

if(Options["whichClusters"] == "nolength"):
    maxstorms = 8
    # the list named ticks, summarizes or groups
    ticks = ['Solo', '2', '3','4','5','6','7','8+','','']


for l in range(maxstorms):
	if(l < maxstorms - 1):
		laplTempAtlantic = [np.nanmax(lapl_Atlantic[np.array([np.where(np.array(storms_Atlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Atlantic if len(sublist) == l + 1]
		laplTempPacific = [np.nanmax(lapl_Pacific[np.array([np.where(np.array(storms_Pacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Pacific if len(sublist) == l + 1]
		laplTempsAtlantic = [np.nanmax(lapl_sAtlantic[np.array([np.where(np.array(storms_sAtlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sAtlantic if len(sublist) == l + 1]
		laplTempsPacific = [np.nanmax(lapl_sPacific[np.array([np.where(np.array(storms_sPacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sPacific if len(sublist) == l + 1]
		laplTempsIndian = [np.nanmax(lapl_sIndian[np.array([np.where(np.array(storms_sIndian) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sIndian if len(sublist) == l + 1]

	else:
		laplTempAtlantic = [np.nanmax(lapl_Atlantic[np.array([np.where(np.array(storms_Atlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Atlantic if len(sublist) >= l + 1]
		laplTempPacific = [np.nanmax(lapl_Pacific[np.array([np.where(np.array(storms_Pacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_Pacific if len(sublist) >= l + 1]
		laplTempsAtlantic = [np.nanmax(lapl_sAtlantic[np.array([np.where(np.array(storms_sAtlantic) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sAtlantic if len(sublist) >= l + 1]
		laplTempsPacific = [np.nanmax(lapl_sPacific[np.array([np.where(np.array(storms_sPacific) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sPacific if len(sublist) >= l + 1]
		laplTempsIndian = [np.nanmax(lapl_sIndian[np.array([np.where(np.array(storms_sIndian) == item)[0] for item in sublist])]) for sublist in sorted_clusters_sIndian if len(sublist) >= l + 1]
	laplTempAll = np.hstack((laplTempAtlantic,laplTempPacific,laplTempsAtlantic,laplTempsPacific,laplTempsIndian))
	boxplotsAll.append(laplTempAll)
	boxplotsAtlantic.append(laplTempAtlantic)
	boxplotsPacific.append(laplTempPacific)
	boxplotsSAtlantic.append(laplTempsAtlantic)
	boxplotsSPacific.append(laplTempsPacific)
	boxplotsSIndian.append(laplTempsIndian)
	
	
	TempAll = np.zeros(10000)
	TempAtlantic = np.zeros(10000)
	TempPacific = np.zeros(10000)
	TempsAtlantic = np.zeros(10000)
	TempsPacific = np.zeros(10000)
	TempsIndian = np.zeros(10000)

	for i in range(10000):
		TempAll[i] =np.nanmax(random.sample(list(lapl_All),l+1))
		TempAtlantic[i] =np.nanmax(random.sample(list(lapl_Atlantic),l+1))
		TempPacific[i] =np.nanmax(random.sample(list(lapl_Pacific),l+1))
		TempsAtlantic[i] =np.nanmax(random.sample(list(lapl_sAtlantic),l+1))
		TempsPacific[i] =np.nanmax(random.sample(list(lapl_sPacific),l+1))
		TempsIndian[i] =np.nanmax(random.sample(list(lapl_sIndian),l+1))

	quantsAll_Expect[l,::] = np.quantile(TempAll,[0.1,0.5,0.9])	
	quantsAtlantic_Expect[l,::] =  np.quantile(TempAtlantic,[0.1,0.5,0.9])
	quantsPacific_Expect[l,::] = np.quantile(TempPacific,[0.1,0.5,0.9])
	quantsSAtlantic_Expect[l,::] = np.quantile(TempsAtlantic,[0.1,0.5,0.9])
	quantsSPacific_Expect[l,::] = np.quantile(TempsPacific,[0.1,0.5,0.9])
	quantsSIndian_Expect[l,::] = np.quantile(TempsIndian,[0.1,0.5,0.9])

if(Options["whichClusters"] == "nolength"):
	for l in range(8,10):
		boxplotsAll.append([])
		boxplotsAtlantic.append([])
		boxplotsPacific.append([])
		boxplotsSAtlantic.append([])
		boxplotsSPacific.append([])
		boxplotsSIndian.append([])
	
		quantsAll_Expect[l,::] = np.nan
		quantsAtlantic_Expect[l,::] =  np.nan
		quantsPacific_Expect[l,::] = np.nan
		quantsSAtlantic_Expect[l,::] = np.nan
		quantsSPacific_Expect[l,::] = np.nan
		quantsSIndian_Expect[l,::] = np.nan
	



### BOX PLOTS
fig = plt.figure(figsize=(12,2.67))
ax = plt.subplot(1, 1, 1)
bxplotAll = plt.boxplot(boxplotsAll, positions=np.array(np.arange(len(boxplotsAll)))*2.0-0.75, widths=0.2, flierprops=dict(markeredgecolor='black',markersize=0.6))
bxplotAtl = plt.boxplot(boxplotsAtlantic, positions=np.array(np.arange(len(boxplotsAll)))*2.0-0.45, widths=0.2, flierprops=dict(markeredgecolor='#9a0200',markersize=0.6))
bxplotPac = plt.boxplot(boxplotsPacific, positions=np.array(np.arange(len(boxplotsAll)))*2.0-0.15, widths=0.2,flierprops=dict(markeredgecolor='darkblue',markersize=0.6))
bxplotSAtl = plt.boxplot(boxplotsSAtlantic, positions=np.array(np.arange(len(boxplotsAll)))*2.0+0.15, widths=0.2,flierprops=dict(markeredgecolor='#e17701',markersize=0.6))
bxplotSPac = plt.boxplot(boxplotsSPacific, positions=np.array(np.arange(len(boxplotsAll)))*2.0+0.45, widths=0.2,flierprops=dict(markeredgecolor='#75bbfd',markersize=0.6))
bxplotSIndi = plt.boxplot(boxplotsSIndian, positions=np.array(np.arange(len(boxplotsAll)))*2.0+0.75, widths=0.2,flierprops=dict(markeredgecolor='#3f9b0b',markersize=0.6))

# setting colors for each groups
# =["black",'xkcd:deep red','darkblue','xkcd:pumpkin','xkcd:sky blue','xkcd:grass green'],
define_box_properties(bxplotAll, 'black', 'All')
define_box_properties(bxplotAtl, '#9a0200', 'N. Atlantic')
define_box_properties(bxplotPac, 'darkblue', 'N. Pacific')
define_box_properties(bxplotSAtl, '#e17701', 'S. Atlantic') #e17701
define_box_properties(bxplotSPac, '#75bbfd', 'S. Pacific')
define_box_properties(bxplotSIndi, '#3f9b0b', 'S. Indian')

# Add results from random sampling
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.75,quantsAll_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.75,quantsAll_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.75,quantsAll_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
#NH
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.45,quantsAtlantic_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.45,quantsAtlantic_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.45,quantsAtlantic_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.15,quantsPacific_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.15,quantsPacific_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0-0.15,quantsPacific_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
#SH
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.15,quantsSAtlantic_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.15,quantsSAtlantic_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.15,quantsSAtlantic_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.45,quantsSPacific_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.45,quantsSPacific_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.45,quantsSPacific_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.75,quantsSIndian_Expect[:,0],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.75,quantsSIndian_Expect[:,1],marker='+',color="dimgrey",s=6,linewidth=1,zorder=2)
plt.scatter(np.array(np.arange(len(boxplotsAll)))*2.0+0.75,quantsSIndian_Expect[:,2],marker='x',color="dimgrey",s=6,linewidth=1,zorder=2)

# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
ax.set_yticks(np.arange(0,10)) #np.arange(0, 5)

#X -Y labels
plt.ylabel("Max. Laplacian")
plt.xlabel("Nr. of storms in cluster")
 
# set the limit for x axis
plt.xlim(-2, len(ticks)*2)

# Hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

# Add gridlines
ax.grid(linewidth=0.5,axis='y',zorder=0)

plt.tight_layout()
plt.savefig("Boxplots_"+ Options["whichClusters"] + ".pdf")


"""
plt.clf()
plt.figure(figsize=(12,4))
data = np.log10(arrayLengths)
cmap = plt.get_cmap('RdBu_r', np.max(data)*2+1) # set limits .5 outside true range
#plt.imshow(data, origin='lower',levels=[0,0.2,0.5,1,1.2,1.5,2.,2.2,2.5,3,3.2,3.5,4.0,4.2,4.5])
mat = plt.matshow(data,cmap=cmap,vmin = -.25, vmax = np.max(data)+.25, fignum=1, ) ## added fignum
plt.xlabel("Length of cluster")
plt.yticks(ticks = [5,4,3,2,1,0], labels = ticks)
plt.colorbar(mat, shrink=0.55) #ticks=np.arange(0,np.max(data)+1), 
plt.savefig("ClustersLength_AreasImage_"+ Options["whichClusters"] + ".pdf")


ticks = ["All", "N. Atlantic", "N. Pacific","S. Atlantic", "S. Pacfic", "S. Indian"]

plt.figure(figsize=(12,4))
data = np.log10(arrayLengths)
cmap = plt.get_cmap('RdBu_r', np.max(data)-+1) # set limits .5 outside true range
#plt.imshow(data, origin='lower')
mat = plt.matshow(data,cmap=cmap,vmin = -.5, vmax = np.max(data)+.5, fignum=1) ## added fignum
plt.xlabel("Length of cluster")
plt.yticks(ticks = [5,4,3,2,1,0], labels = ticks)
plt.colorbar(mat, ticks=np.arange(0,np.max(data)+1), shrink=0.75)
plt.savefig("ClustersLength_AreasImage_"+ Options["whichClusters"] + ".pdf")
"""
