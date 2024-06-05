# CyClust

Cyclone clustering detection, based on  ... Additional analysis notebooks can be found at ...

# Basic idea of algorithm

# Desired input

Set of tracked storms, currently using output from the Fortran version of the Murray and Simmonds (1991) algorithm, however the algorithm also has been tested on the tracks of the Hodges (1995) algorithm, and an adapted python version of the Murray and Simmonds algorithm included in the Dynlib library. 

As an input it uses a text file with containing different columns with at least
- Storm id (can start/contain arbitrary numbers)
- Longitude
- Latitude
- Time (currently assumes YYYYMMDDHH format, e.g. 2011120112, for the 1 Dec 2011, 12 UTC)
See for example the text files in the Tracks directory.

Alternatively numpy arrays of length n can directly be used or read in for each variable seperately, with a numpy array for each variable (Storm id, Lon, Lat, Time) seperately. 

# Input parameters

#Thresholds used in algorithm
- distthresh = 1.0 #1. Distance criterium
- timthresh = 36.0 #2. Time criterium, in hours
- lngthresh = 1.5 #3. Length overlap criterium (in Rossby Radius)
- timlngthresh = 6 #4. Time overlap criterium (in time steps)

frameworkSparse = True #If True, uses sparse matrices to save results

# Output
The algorithm saves the output in a nested list containing the Storm id, a list of sublist, where the sublists are the detected clusters or solo storms. E.g.
[[1] [2 3] [4] [5 7] [6]] ... indicates that two clusters [2 3] and [5 7] detected. 

# References
Hodges, K. I. (1995). Feature tracking on the unit sphere. _Monthly Weather Review_, 123(12), 3458-3465.
Murray, R. J., & Simmonds, I. (1991). A numerical scheme for tracking cyclone centres from digital data. _Australian meteorological magazine_, 39(3), 155-166.
Spensberger, C. (2024). Dynlib: A library of diagnostics, feature detection algorithms, plotting and convenience functions for dynamic meteorology (1.4.0). Zenodo.
