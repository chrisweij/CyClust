# CyClust

Cyclone clustering detection, detection algorithm used in Weijenborg and Spengler (2024). Additional analysis notebooks can be found at ...

# Basic idea of algorithm

The basis idea of the clustering algorithm is that it checks if multiple cyclone tracks follow a similar path, based on the 'cyclone families' described in Bjerknes and Solberg (1922). For details see Weijenborg and Spengler (2024). The algorith further divides cyclone clusters into two different types, a 'Bjerknes type' close to the cyclone families of Bjerkens and Solberg (1922) and a stagnant type. The former type detects cyclones that follow each other over a certain minimum distance, whereas the stagnant type includes cyclones which do not move much in space, but still have a proximity over time.

# Run of algorithm 

The algorithm can be runned using the script Find_clusters3d.py. All options/parameters listed below can be set in Options.yaml, afterwards the script Find_clusters3d.py can be run. This file reads in the parameters set in the Options.yaml file, detects the clusters and the subtypes, and finally saves the output clustered cyclones as a npz file. 

As standard it reads in "Tracks/Selected_tracks_2011_2012_DJF", which includes cyclone tracks of the 2011-2012 winter, tracked with the Melbourne and Simmonds algorithm. 

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
- distthresh = 1.0 #1. Distance criterium (in Rossby Radii)
- timthresh = 36.0 #2. Time criterium (in hours)
- lngthresh = 1.5 #3. Length overlap criterium (in Rossby Radii)
- timlngthresh = 48.0 #4. Time overlap criterium (in hours)

frameworkSparse = True #If True, uses sparse matrices to save results

# Output
_sorted_clusters_ The algorithm saves the output in a nested list containing the Storm id, a list of sublist, where the sublists are the detected clusters or solo storms. E.g.
[[1] [2 3] [4] [5 7] [6]] ... indicates that two clusters [2 3] and [5 7] detected. 

_sorted_clusters_length_ and _sorted_clusters_nolength_ It further outputs a similar list for each of the two subtypes. Names will be changed in the near future to Bjerknes and Stagnant to keep consistent with the published paper. 

_ConnTracks_ NxN array with N the maximum number of str_id, with ConnTracks[i,j] containing a nonzero number if two cyclones are connected to each other (part of the same cluster). 

NB: The output can be loaded with the numpy function load, but to be able to load the clustering list, you have to set the allow_pickle option to True, e.g.:
``` python
Results = np.load("result_file.npz",allow_pickle=True)
```

# References
Bjerknes, J. (1922). Life cycle of cyclones and the polar front theory of atmospheric circulation. Geofys. Publ., 3, 1-18.\
Hodges, K. I. (1995). Feature tracking on the unit sphere. _Monthly Weather Review_, 123(12), 3458-3465.\
Murray, R. J., & Simmonds, I. (1991). A numerical scheme for tracking cyclone centres from digital data. _Australian meteorological magazine_, 39(3), 155-166.\
Spensberger, C. (2024). Dynlib: A library of diagnostics, feature detection algorithms, plotting and convenience functions for dynamic meteorology (1.4.0). Zenodo.\
Weijenborg, C., & Spengler, T. (2024). Detection and global climatology of two types of cyclone clustering.
