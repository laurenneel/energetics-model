

import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl




#######   SET CONSTANTS   #######################################################





#for beta probability distribution
a, b = 1.0,4.0



#for von Mises prob dist
k, mu = 1.0,0.0

#earth essentricity 
e=0.01675 

#So = solar constant = 1360 W m-2
S0=1360 #Wm-2

#boltzman constant, sigma = 5.67e-8
sigma=5.67e-8 # W m-2 k-4

#ordinal day of the year
julian_days= range(365)

# eg = ground emissivity
eg=0.75


time_list=np.arange(0,24,0.1)

julian=4
#latitude=33.4484
#longitude=-112.0740
altitude=350
tau=0.7
rg=0.4 #ground reflectance #when modeled with sears April2019 we used rg=0.4
h=0.1 #SVL of 100mm
d=0.1 #animal diameter in unit meters 

Tave=25.
amp=15.
z=0.
D=0.03
u=0.1 #u=wind in m/s

max_t_yesterday=28.0
min_t_today=10.5
max_t_today=28.9
min_t_tomorrow=10.5


# alpha_s = shortwave rad absorptance of the animal 
alpha_s = 0.8

# alpha_l = longwave rad absorptance of the animal
alpha_l=0.95

#cp = specific heat of air at constant temp 
cp=29.3 #unit = J/(mol*C)

mass=20



##################################################################################
#2D landscapes differ in:
    # Availability of Tpref
    # spatial arrangement of temps (dispersion)
    # thermal variance of temps

#Each landscape is a 64 x 64 tiled grid of cells
    # each cell classified as preferred (Te = 34) or non-preferred
    # the mean for nonpreferred cells ranged from 34° to 50°C in increments of 1°C 
    # thermal heterogeneity was created by increasing the std dev of mean cell temps 
        # the standard deviation was 0.25°, 0.5°, 1.0°, or 2.0°C for preferred cells 
        # sd was 0.5°, 1.0°, 2.0°, or 4.0°C for nonpreferred cells

#For each distribution of Te, values were assigned to cells according to one of three spatial structures: 
    # 1 patch, 4 patches, and 16 patches of preferred temperatures
    
    
# alpha, Operative temperatures were drawn from bimodal distributions and distributed among cells 
    #to create a spatial structure
    # Preferred patches—containing temperatures from the part of the distribution surrounding the lower mode—composed 6.25%, 14.1%, 25.0%, or 39.1% of the environment. 
    # Preferred patches were distributed within 1 large patch or spread evenly among either 4 or 16 smaller patches. 
    # Operative temperatures of the remaining cells were drawn from the part of the distribution surrounding the upper mode, which was hotter than preferred. 
    # To manipulate thermal heterogeneity within environments, we increased the standard deviation of temperature within preferred and nonpreferred cells from 0.5° to 4°C without changing the means (only one of those distributions is shown here)
    
#distance and range of angles searched were drawn from beta and von mises distributions

#beta distribution required 2 parameters: (set by sears and angilletta costs benefit 2015)
    # alpha, a = 1
    # beta, b = 4
    
    # these parameter settings mean locations were sampled more intesnsely at closer proximity 

#max distance sampled
def max_distance_sampled(mass):
    return 8 * (math.log10(mass) + 1)

#distance over which animals sampled habitat modeled as a beta probability distribution 
# a/p and b/q are shape parameters 
    # they assumed a = 1 and b = 4
    
# range of angles searched drawn from von Mises distribution
    #requires 2 parameters:
    # k , determines concentration of search angles 
    # mu , determines the angle with reference to a forward facing direction 
    # k = 0, mu = 1 such that animals oriented straight ahead and concentrated their searches in a forward direction between 1p/2 and 2p/2 radians
    
    
# %%
