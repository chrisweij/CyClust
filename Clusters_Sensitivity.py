#!/usr/bin/env python
# -*- encoding: utf-8

#from dynlib.shorthands import dt, td, get_instantaneous, metsave, fig, np
from dynlib.settings import proj

from dynlib.context.erainterim import conf
import dynlib.context.derived
from scipy.ndimage.filters import gaussian_filter
#from geopy.distance import great_circle as great_circle_old

from datetime import datetime as dt, timedelta as td
from dynlib.metio import metopen, metsave, get_instantaneous
import matplotlib.dates as mdates 

import dynlib.diag
import matplotlib 
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import loadtxt
import dynlib.figures as fig

#exec(open("projection.py").read())
from dynlib import sphere, cm
import time
import math
from numpy import loadtxt
conf.datapath.append("/Data/gfi/share/Reanalysis/ERA_INTERIM/TOPO_LANDMASK/")

#########################
# Thresholds
#########################
#1. Distance criterium
distthresh = 1.0 #1000.0

#2. Time criterium
timthresh = 60.0

#3. Length criterium 
#lngthresh = 2000.0
lngthresh = 1.5 #calc_Rossby_radius(lat=45)*2.0 # 1000.0

#Sensitivity ranges for thresholds
#New set of experiments
timthreshs = np.arange(1,4.1,0.25)*24.0
lngthreshs = np.arange(0.6,2.21,0.2)
distthreshs = np.arange(0.5,1.51,0.1)

#########################
# Load results
#########################
nrclust_dist_tim = np.zeros([len(distthreshs),len(timthreshs)])
nrcluststrm_dist_tim = np.zeros([len(distthreshs),len(timthreshs)])
maxdistmean  = np.zeros([len(distthreshs),len(timthreshs)])
maxdistmedian = np.zeros([len(distthreshs),len(timthreshs)])
maxdiststd = np.zeros([len(distthreshs),len(timthreshs)])

nrclust_dist_length = np.zeros([len(distthreshs),len(lngthreshs)])
nrcluststrm_dist_length = np.zeros([len(distthreshs),len(lngthreshs)])

nrclust_tim_length= np.zeros([len(timthreshs),len(lngthreshs)])
nrcluststrm_tim_length = np.zeros([len(timthreshs),len(lngthreshs)])

formatter =  "{:1.1f}"
outfile = "/home/cwe022/Clusters_Sensitivity/Results_dist_" + formatter.format(distthresh) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthresh)

#Varying distance and time difference
#lngthresh = 2000.0
for idx1 in range(len(distthreshs)):
	for idx2 in range(len(timthreshs)):
		infile = "/home/cwe022/Clusters_Sensitivity/Results_DJF_NH_dist_" + formatter.format(distthreshs[idx1]) + "_tim_" + formatter.format(timthreshs[idx2]) + "_length_" + formatter.format(lngthresh) + ".npz" #vary_dist_tim/
		#infile = "/Data/gfi/spengler/cwe022/Clusters_Sensitivity/Results_dist_" + formatter.format(distthreshs[idx1]) + "_tim_" + formatter.format(timthreshs[idx2]) + "_length_" + formatter.format(lngthresh) + ".npz" #vary_dist_tim/
		results = np.load(infile)
		nrclust_dist_tim[idx1,idx2] = np.nanmean(results['nrclst_wint'])
		nrcluststrm_dist_tim[idx1,idx2] = np.nanmean(results['nrstrmclst_wint'])
		maxdistmean[idx1,idx2] = np.nanmean(results['maxdists'])
		maxdistmedian[idx1,idx2] = np.nanmedian(results['maxdists'])
		maxdiststd[idx1,idx2] = np.nanstd(results['maxdists'])

		#if((distthreshs[idx1] == 1000.0) & (timthreshs[idx2] == 60)):
		if((distthreshs[idx1] >= 0.999) & (distthreshs[idx1] <= 1.001) & (timthreshs[idx2] == 60)):
			print("Saving maxdists")
			maxdists = results['maxdists']

#Varying distance and connection length
#timthresh = 60.0
for idx1 in range(len(distthreshs)):
	for idx2 in range(len(lngthreshs)):
		infile = "/home/cwe022/Clusters_Sensitivity/Results_DJF_NH_dist_" + formatter.format(distthreshs[idx1]) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthreshs[idx2]) + ".npz" #/vary_dist_length
		#infile = "/Data/gfi/spengler/cwe022/Clusters_Sensitivity/Results_dist_" + formatter.format(distthreshs[idx1]) + "_tim_" + formatter.format(timthresh) + "_length_" + formatter.format(lngthreshs[idx2]) + ".npz" #/vary_dist_length
		results = np.load(infile)
		nrclust_dist_length[idx1,idx2] = np.nanmean(results['nrclst_wint'])
		nrcluststrm_dist_length[idx1,idx2] = np.nanmean(results['nrstrmclst_wint'])
		#maxdistmean[idx1,idx2] = np.nanmean(results['maxdists'])
		#maxdistmedian[idx1,idx2] = np.nanmedian(results['maxdists'])


#Varying connection length and time difference
#distthresh = 1000.0
for idx1 in range(len(timthreshs)):
	for idx2 in range(len(lngthreshs)):
		infile = "/home/cwe022/Clusters_Sensitivity/Results_DJF_NH_dist_" + formatter.format(distthresh) + "_tim_" + formatter.format(timthreshs[idx1]) + "_length_" + formatter.format(lngthreshs[idx2]) + ".npz" #/vary_tim_length
		#infile = "/Data/gfi/spengler/cwe022/Clusters_Sensitivity/Results_dist_" + formatter.format(distthresh) + "_tim_" + formatter.format(timthreshs[idx1]) + "_length_" + formatter.format(lngthreshs[idx2]) + ".npz" #/vary_tim_length
		results = np.load(infile)
		nrclust_tim_length[idx1,idx2] = np.nanmean(results['nrclst_wint'])
		nrcluststrm_tim_length[idx1,idx2] = np.nanmean(results['nrstrmclst_wint'])
		#maxdistmean[idx1,idx2] = np.nanmean(results['maxdists'])
		#maxdistmedian[idx1,idx2] = np.nanmedian(results['maxdists'])

#########################
# Plotting results
#########################
class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s



def calc_frac_change(array):
	shape_arr = array.shape
	frac_change_x = np.zeros(shape_arr)
	frac_change_y = np.zeros(shape_arr)
	for idx1 in range(shape_arr[0]):
		if(idx1 == 0):
			frac_change_x[idx1,:] = (array[idx1+1,:] - array[idx1,:])/array[idx1,:]
		elif(idx1 == shape_arr[0] - 1):
			frac_change_x[idx1,:] = (array[idx1,:] - array[idx1-1,:])/array[idx1,:]
		else:
			frac_change_x[idx1,:] = 0.5*(array[idx1+1,:] - array[idx1,:])/array[idx1,:] + 0.5*(array[idx1,:] - array[idx1 -1,:])/array[idx1,:]

	for idx2 in range(shape_arr[1]):
		if(idx2 == 0):
			frac_change_y[:,idx2] = (array[:,idx2+1] - array[:,idx2])/array[:,idx2]
		elif(idx2 == shape_arr[1] - 1):
			frac_change_y[:,idx2] = (array[:,idx2] - array[:,idx2-1])/array[:,idx2]
		else:
			frac_change_y[:,idx2] = 0.5*(array[:,idx2+1] - array[:,idx2])/array[:,idx2] + 0.5*(array[:,idx2] - array[:,idx2 -1])/array[:,idx2]		

	return frac_change_x, frac_change_y

def calc_grad_change(x,y,array):
	shape_arr = array.shape
	grad_change_x = np.zeros(shape_arr)
	grad_change_y = np.zeros(shape_arr)
	for idx1 in range(shape_arr[0]):
		if(idx1 == 0):
			grad_change_x[idx1,:] = (array[idx1+1,:] - array[idx1,:])/(x[idx1+1] - x[idx1])
		elif(idx1 == shape_arr[0] - 1):
			grad_change_x[idx1,:] = (array[idx1,:] - array[idx1-1,:])/(x[idx1] - x[idx1-1])
		else:
			grad_change_x[idx1,:] = (array[idx1+1,:] - array[idx1,:])/(x[idx1+1] - x[idx1-1])

	for idx2 in range(shape_arr[1]):
		if(idx2 == 0):
			grad_change_y[:,idx2] = (array[:,idx2+1] - array[:,idx2])/(y[idx2+1] - y[idx2])
		elif(idx2 == shape_arr[1] - 1):
			grad_change_y[:,idx2] = (array[:,idx2] - array[:,idx2-1])/(y[idx2] - y[idx2-1])
		else:
			grad_change_y[:,idx2] = 0.5*(array[:,idx2+1] - array[:,idx2])/(y[idx2+1] - y[idx2-1])

	return grad_change_x, grad_change_y

#################################################
#Varying distance and time difference
#################################################

#### CLUSTERS  ####
fig1, ax = plt.subplots()
plt.contourf(timthreshs,distthreshs,nrclust_dist_tim,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrclust_dist_tim)        
CS1 = ax.contour(timthreshs,distthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(timthreshs,distthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Time difference threshold (hours)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("nrclust_dist_tim_length__" + formatter.format(lngthresh) + ".png")

#### STORMS  ####
fig1, ax = plt.subplots()
plt.contourf(timthreshs,distthreshs,nrcluststrm_dist_tim,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrcluststrm_dist_tim)        
CS1 = ax.contour(timthreshs,distthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(timthreshs,distthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Time difference threshold (hours)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("nrcluststrm_dist_tim_length_" + formatter.format(lngthresh) + ".png")

#### Max and median connection length  ###
plt.figure()
plt.contourf(timthreshs,distthreshs,maxdistmean,cmap="OrRd")
plt.xlabel("Time difference threshold (hours)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("maxdists_mean_dist_tim.png")

plt.figure()
plt.contourf(timthreshs,distthreshs,maxdistmedian,cmap="OrRd")
plt.xlabel("Time difference threshold (hours)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("maxdists_median_dist_tim.png")


#################################################
#Varying distance and connection length
#################################################
#### CLUSTERS  ####
fig1, ax = plt.subplots()
plt.contourf(lngthreshs,distthreshs,nrclust_dist_length,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrclust_dist_length)        
CS1 = ax.contour(lngthreshs,distthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(lngthreshs,distthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Connection length threshold (km)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("nrclust_dist_length.png")

#### STORMS  ####
fig1, ax = plt.subplots()
plt.contourf(lngthreshs,distthreshs,nrcluststrm_dist_length,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrcluststrm_dist_length)        
CS1 = ax.contour(lngthreshs,distthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(lngthreshs,distthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Connection length threshold (km)")
plt.ylabel("Distance threshold (km)")
plt.colorbar()
plt.savefig("nrcluststrm_dist_length.png")

#################################################
# Varying connection length and time difference
#################################################
#### CLUSTERS  ####
fig1, ax = plt.subplots()
plt.contourf(lngthreshs,timthreshs,nrclust_tim_length,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrclust_tim_length)        
CS1 = ax.contour(lngthreshs,timthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(lngthreshs,timthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Connection length threshold (km)")
plt.ylabel("Time difference threshold (hours)")
plt.colorbar()
plt.savefig("nrclust_tim_length.png")

#### STORMS  ####
fig1, ax = plt.subplots()
plt.contourf(lngthreshs,timthreshs,nrcluststrm_tim_length,cmap="OrRd")
frac_change_x, frac_change_y, = calc_frac_change(nrcluststrm_tim_length)        
CS1 = ax.contour(lngthreshs,timthreshs,frac_change_x*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="black")
# Recast levels to new class
CS1.levels = [nf(val) for val in CS1.levels]
CS2 = ax.contour(lngthreshs,timthreshs,frac_change_y*100,levels=[-100,-75,-60,-50,-40,-30,-20,-10,-5,0,5,10,20,30,40,50,60,75,100],linewidths=0.65,colors="dimgrey")
# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r \%%'
else:
    fmt = '%r %%'

ax.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlabel("Connection length threshold (km)")
plt.ylabel("Time difference threshold (hours)")
plt.colorbar()
plt.savefig("nrcluststrm_tim_length.png")
