# CyClust

Cyclone clustering detection 

# Desired input

Set of tracked storms, currently output from the Murray and Simmonds (1991) algorithm. 

Text file with containing different columns with at least
- Storm id (can start/contain arbitrary numbers)
- Longitude
- Latitude
- Time (currently assumes YYYYMMDDHH format, e.g. 2011120112, for the 1 Dec 2011, 12 UTC)

# Input parameters

#Thresholds used in algorithm
distthresh = 1.0 #1. Distance criterium
timthresh = 36.0 #2. Time criterium, in hours
lngthresh = 1.5 #3. Length overlap criterium (in Rossby Radius)
timlngthresh = 6 #4. Time overlap criterium (in time steps)

frameworkSparse = True #If True, uses sparse matrices to save results

# Output
The algorithm saves the output in a nested list (a list of sublist, where the sublist are the detected clusters or solo storms)
