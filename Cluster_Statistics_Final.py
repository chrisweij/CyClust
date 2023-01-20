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
        conn_temp = str_connected[ids_storms[strm]]
        
        
        
        if(np.any(conn_temp !=0)):
            str_first[strm -1] = np.nanmin(dt_temp[conn_temp >0])
         
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
plt.rcParams['font.sans-serif'] = ['Arial'] #,'Arial','Verdana','Helvetica']
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
ax.grid(linestyle = '--')

# Hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

plt.legend(labels=["All","North Atlantic","North Pacific","South Atlantic","South Pacific","South Indian"])
plt.tight_layout()
plt.savefig("ClustersLength_Areas_lines_"+ Options["whichClusters"] + ".pdf")


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
