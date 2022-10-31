

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

# -*- coding: utf-8 -*-

from time import time as what_time
from random import choice, gauss
import numpy as np
from numpy.random import *
from scipy.stats.mstats import *
from scipy.interpolate import interp1d
from numpy.random import uniform, seed, normal, binomial
from numpy import pi, min, max, percentile, radians, degrees
from numpy.fft import ifft2
import multiprocessing
from time import time
from scipy.interpolate import *
from numpy import percentile as percent
#from matplotlib.mlab import prctile as percent #mlab.prctile (use numpy.percentile instead)
from numpy import flipud, array
from math import exp, log, sin, cos, sqrt, acos, asin, atan, atan2
#from numpy import arctan as atan
#from numpy import arctan2 as atan2
#from numpy import arcsin as asin
#from numpy import arccos as acos
import string
from time import time


seed() #99

J = input("day? ")

def do_sim():

	SIGMA = 5.673 * 10**(-8) # Stephan-Boltmann constant
	SOLAR = 1360. # Solar constant
	#filename = raw_input("site? ")
	
	#t =  input("time of day? ")

	climate_change = 0 #input("Climate change scenario? 0 = no, 1= yes: ")
	if climate_change==1:
		minTemps = np.array([-5., -5.2, -2.7, 1.3, 5.8, 11.5 , 16.2, 18.5, 17.3, 12.7, 6.6, -1.0, -5.0, -5.2]) +3.
		maxTemps = np.array([14.4,15.2, 17.6, 21.4, 26.6, 31.2, 35.6, 36.0, 34.7, 31.2, 25.4, 18.8, 14.4, 15.2]) +3.
		g=3.
		climname = "change"
	else:
		minTemps = np.array([-1.1,-1.1, 1.1, 3.3, 6.7, 11.7, 16.7, 20.6, 20., 15.6, 9.4, 2.8, -1.1, -1.1]) - 10.23 
		maxTemps = np.array([11.7,12.2, 14.4, 18.9, 23.3, 29.4, 35.6, 38.3, 36.7, 32.8, 25.6, 17.8, 11.7, 12.2]) - 10.23
		g=-4.9
		climname = "norm"

	#write_to_file = raw_input("Write thermal map to: ")

	# DATA

	# Create air temperatures and soil temperatures
	# from splined climate records

	# Air temperature data from Zion NP, east entrance

	days = np.array([-16, 15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381])

	maxT = UnivariateSpline(days, maxTemps, k=3)
	minT = UnivariateSpline(days, minTemps, k=3)

	# Ground temperature data from splines of raw data from 24 days <0.5 C from average MIN & MAX for that day
	hours = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,24.])
	janTemps = np.array([-7.01,-7.77,-7.89,-8.28,-8.72,-8.99,-9.32,-9.74,-8.6,-5.61,-1.41,2.74,6.3,8.37,8.65,6.85,3.44,0.28,-2.03,-3.38,-4.68,-5.22,-5.91,-6.82,-7.01]) + g
	febTemps = np.array([-5.15,-6.06,-6.16,-6.59,-6.46,-7.39,-7.75,-7.71,-5.5,-1.36,3.6,7.96,11.43,13.52,13.71,12.09,8.77,4.29,1.23,-0.84,-1.93,-2.98,-3.64,-4.49,-5.11]) + g
	marTemps = np.array([-1.24,-2.36,-2.52,-3.05,-2.81,-3.97,-4.31,-3.5,0.33,5.38,11.2,15.95,19.81,22.14,23.03,21.03,17.54,12.34,8,5.08,3.05,1.76,0.62,-0.76,-1.14]) + g
	aprTemps = np.array([1.7,0.86,0.09,-0.33,-0.84,-1.12,-1.23,1.34,5.94,11.43,17.02,22.06,25.62,27.8,28.09,26.95,23.81,18.94,13.68,9.61,6.88,5.31,3.89,2.94,1.85]) + g
	mayTemps = np.array([7.78,6.76,6.03,5.11,4.91,4.31,5.1,8.89,13.94,19.55,25.38,30.42,34.1,36.79,37,35.81,32.65,28.4,22.89,17.18,14.25,12.02,10.67,8.97,7.84]) + g
	junTemps = np.array([11.87,10.92,10.14,9.26,8.96,8.49,9.92,13.89,19.07,24.5,30.68,35.81,39.59,42.44,42.66,41.56,38.49,34.26,28.76,22.88,19.22,16.69,14.81,12.9,12.08]) + g
	julTemps = np.array([17.73,16.67,15.93,14.97,14.78,14.27,15.26,19.37,24.45,30.41,35.81,40.79,44.37,46.65,47.28,46.63,43.27,38.9,33.85,28.06,24.45,21.91,20.33,18.91,17.92]) + g
	augTemps = np.array([16.17,15.02,14.3,14.42,13.15,12.65,13.28,16.21,21.04,26.26,32.55,37.76,41.68,43.98,44.6,43.14,40.12,35.4,29.63,24.99,21.99,19.85,18.38,17.11,15.59]) + g
	sepTemps = np.array([9.99,9.23,8.59,7.83,7.53,7.07,6.87,8.59,12.79,18.13,23.99,29.24,33.2,36.07,36.17,34.71,31.08,26.37,20.91,17.34,14.67,13.43,12.11,11.29,10.11]) + g
	octTemps = np.array( [1.92,1.03,0.67,0.16,-0.75,-0.73,-1.13,-0.97,2.13,6.87,12.26,17.56,21.76,24.31,25.28,23.18,19.49,14.15,10.33,7.82,6.02,4.76,3.68,2.3,1.99]) + g
	novTemps = np.array( [-4.09,-3.99,-5.08,-5.51,-5.54,-6.28,-6.62,-7.01,-5.96,-1.85,2.78,7.13,11.2,13.47,13.88,12.1,8.58,4.35,1.83,0.2,-0.61,-1.98,-2.79,-3.79,-4.07]) + g
	decTemps = np.array([-9.46,-9.49,-10.35,-10.75,-10.47,-11.46,-11.79,-12.02,-11.42,-8.67,-4.58,0.16,3.44,5.67,6.01,4.15,0.66,-2.1,-4.52,-5.83,-6.52,-7.66,-8.35,-9.23,-9.47]) + g

	janSpline = UnivariateSpline(hours,janTemps,k=3)
	febSpline = UnivariateSpline(hours,febTemps,k=3)
	marSpline = UnivariateSpline(hours,marTemps,k=3)
	aprSpline = UnivariateSpline(hours,aprTemps,k=3)
	maySpline = UnivariateSpline(hours,mayTemps,k=3)
	junSpline = UnivariateSpline(hours,junTemps,k=3)
	julSpline = UnivariateSpline(hours,julTemps,k=3)
	augSpline = UnivariateSpline(hours,augTemps,k=3)
	sepSpline = UnivariateSpline(hours,sepTemps,k=3)
	octSpline = UnivariateSpline(hours,octTemps,k=3)
	novSpline = UnivariateSpline(hours,novTemps,k=3)
	decSpline = UnivariateSpline(hours,decTemps,k=3)


	class T_e():
		def __init__(self, emissivity = 0.95, phi = 37.2025, longitude = 67.01, elev = 0., h = 0.07, d = 0.2, emmiss_animal = 0.95, sky = 0.7, r_soil = 0.3, emmiss_soil = 0.95, abs_animal_long = 0.97, abs_animal_short = 0.7, Ta_min0 = 20., Ta_min1 = 20., Ta_min2 = 20., Ta_max0 = 35., Ta_max1 = 35., Ta_max2 = 35., ampl = 15., wind = 0.1):
			self.id = id
			self.SIGMA =  5.673e-8
			self.SOLAR = 1360.
			self.emissivity = emissivity
			self.phi = phi
			self.longitude = longitude
			self.elev = elev
			self.h = h
			self.d = d
			self.emmiss_animal = emmiss_animal
			self.char_dim = d
			self.sky = sky
			self.r_soil = r_soil
			self.emmiss_soil = emmiss_soil
			self.abs_animal_long = abs_animal_long
			self.abs_animal_short = abs_animal_short
			self.Ta_min0 = Ta_min0
			self.Ta_min1 = Ta_min1
			self.Ta_min2 = Ta_min2
			self.Ta_max0 = Ta_max0
			self.Ta_max1 = Ta_max1
			self.Ta_max2 = Ta_max2
			self.ampl = ampl
			self.wind = wind
			self.size = 6.
			self.h_t = exp(log(self.size) * 0.36 + 0.72)
			self.c_t = exp(log(0.42 + 0.44 * self.size))
			self.Z=0.
			self.TA= 0.
		
			self.x_min, self.y_min = 0., 0.
			self.x_max, self.y_max = 908,728 #x_max, y_max
			self.position = {'x':rand() * self.x_max, 'y': rand() * self.y_max}
			#if veget[int(self.position['y'])][int(self.position['x'])] ==1:
			#	while veget[int(self.position['y'])][int(self.position['x'])] ==1:
			#		self.position = {'x':rand() * self.x_max, 'y': rand() * self.y_max}
			# This line makes sure that lizard falls in plot
			#if lizplot[int(self.position['y'])][int(self.position['x'])] != 1:
			while lizplot[int(self.position['y'])][int(self.position['x'])] != 1:
				self.position = {'x':rand() * self.x_max, 'y': rand() * self.y_max}
			
			self.thigh =  35.
			self.tlow = 29.
			self.topt = 32.
			self.mu = 0.
			self.kappa = 0.25
			self.beta_alpha = 1.
			self.beta_beta = 3.
			self.d_max = 5.*(np.log10(self.size)+1.)
			self.p1 = 0.5
			self.p2 = 0.0
			self.p3 = 0.0
			self.decisions = 6 #12
			self.tb = 20.
			self.te = 20.
			self.orientation = vonmises(self.mu, self.kappa)
			self.dist = 0.
			self.ctmax = 39.
			self.moved = 0
			self.cost = 0
			self.totenergy = 0.
			self.dist = 0.
			self.active = 0
			self.t_target = 32.
			self.db = 0.
			self.de = 0.
			self.energy_balance = 0.
			self.moved = 0.
			self.tot_dist_moved = 0.
			self.total_activity = 0.
			self.ate = 0.
			self.id = 0
	
		def tprops(self): #thermal preoperties
			self.h_t = exp(log(self.size) * 0.36 + 0.72)
			self.c_t = exp(0.42 + 0.44 * log(self.size))

		def greybody(self, K):
			return self.emissivity * self.SIGMA * (K + 273)**4
	
		def zenith(self ,t, J):	# phi = lattitde, delta = solar declination, t0 = solar noon
			self.Z = degrees(acos(sin(radians(self.phi)) * sin(radians(self.declination(J))) + cos(radians(self.phi)) * cos(radians(self.declination(J))) * cos(radians(15. * (t - self.noon(J))))))
			#if self.Z <0.:
			#	self.Z = 0.
			if self.Z >90:
				self.Z = 90.
			return
	
		def altitude(self, t, J):
			return degrees(asin(sin(radians(self.phi)) * sin(radians(self.declination(J))) + cos(radians(self.phi)) * cos(radians(self.declination(J))) * cos(radians(15. * (t - self.noon(J))))))
		
		def azimuth(self, t, J):
			#return degrees(asin(-cos(radians(self.declination(J))) * sin(radians(15. * (t - self.noon(J))))/cos(radians(self.altitude(t,J)))))
			self.zenith(t,J)
			#print self.Z, self.phi
			if t<self.noon(J):
				#return 180.- degrees(acos(-(sin(radians(self.declination(J)))-cos(radians(self.zenith(t,J)))*sin(radians(self.phi)))/(cos(radians(self.phi))*sin(radians(self.zenith(t,J))))))
				return 180.- degrees(acos(-(sin(radians(self.declination(J)))-cos(radians(self.Z))*sin(radians(self.phi)))/(cos(radians(self.phi))*sin(radians(self.Z)))))
			else:
				#return 180.+ degrees(acos(-(sin(radians(self.declination(J)))-cos(radians(self.zenith(t,J)))*sin(radians(self.phi)))/(cos(radians(self.phi))*sin(radians(self.zenith(t,J))))))
				return 180.+ degrees(acos(-(sin(radians(self.declination(J)))-cos(radians(self.Z))*sin(radians(self.phi)))/(cos(radians(self.phi))*sin(radians(self.Z)))))
			
		def t_adj(self, t, J):
			return t -12. + self.noon(J)
	
		def declination(self, J):	# delta
			return degrees(asin(0.39785* sin(radians(278.97 + 0.9856 * J + 1.9165 * sin(radians(356.6 + 0.9856 * J))))))
	
		def LC(self):
			return self.longitude % 15
	
		def f(self, J):
			return 279.575 + 0.9856 * J
	
		def ET(self, J):
			return (-104.7 * sin(radians(self.f(J))) + 596.2 * sin(radians(2 * self.f(J))) + 4.3 * sin(radians(3 * self.f(J))) - 12.7 * sin(radians(4 * self.f(J))) -429.3 * cos(radians(self.f(J)))- 2.0 * cos(radians(2 * self.f(J))) + 19.3 * cos(radians(3 * self. f(J))))/3600.
	
		def noon(self ,J):
			return 12 + self.LC() / 15. - self.ET(J)
	
		def bardd2(self, J):
			return 1 + 2 * 0.01675 * cos(2 * pi / 365 * J)
	
		def sol_noatm(self, t, J, a_s, alpha_s):
			self.zenith(t,J)
			if self.Z<90.:
				s = self.SOLAR * self.bardd2(J) * cos(self.a_o_i(t, J, a_s, alpha_s))##*cos(self.a_o_i(t, J, a_s, alpha_s))#(radians(self.zenith(t, J)))#cos(radians(self.zenith(t, J)))#
				if s >= 0.: return s
				else: return 0.
			else: return 0.
	
		def p_a(self):
			return 101.3 * exp(-self.elev / 8200.)
	
		def t_m(self, t, J):
			#return self.sky**(self.p_a() / (101.3 * cos(radians(self.zenith(t, J)))))
			return self.sky**(self.p_a() / (101.3 * cos(radians(self.Z))))
		
		def sol_atm(self, t, J, a_s=0, alpha_s=0):
			return self.sol_noatm(t, J, a_s, alpha_s) * self.t_m(t, J)
	
		def sol_diffuse(self, t, J, a_s, alpha_s):
			return self.sol_noatm(t, J, a_s, alpha_s) * 0.3 * (1 - self.t_m(t, J))
	
		def h_g(self, t, J, a_s=0, alpha_s=0):
			return self.sol_diffuse(t, J, a_s, alpha_s) + self.sol_atm(t, J, a_s=0, alpha_s=0)
	
		def short_ground(self, t, J): 
			return self.h_g(t, J) * self.r_soil
		
		def gamma_t(self, t):
			return 0.44 - 0.46 * sin(t * pi / 12. + 0.9) + 0.11 * sin(2. * t * pi / 12. + 0.9)

		def t_air(self, t, J):
			if (t >= 0.) and (t <= 5.):
				self.TA = maxT(J - 1) * self.gamma_t(self.t_adj(t, J)) + minT(J) * (1 - self.gamma_t(self.t_adj(t, J)))
				return self.TA
			if (t > 5.) and (t <= 14.):
				self.TA = maxT(J) * self.gamma_t(self.t_adj(t, J)) + minT(J) * (1 - self.gamma_t(self.t_adj(t, J)))
				return self.TA
			if (t > 14.) and (t <= 24.):
				self.TA = maxT(J) * self.gamma_t(self.t_adj(t, J)) + minT(J + 1) * (1 - self.gamma_t(self.t_adj(t, J)))
				return self.TA


		def t_ground(self, t, J, sun = 1.):
					
			if 15 > J >= 0: # for days between jan 1 and jan 15
				return sun * ((J - 0.)/(14. - -17.) * (janSpline(t)[0] - decSpline(t)[0]) + decSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 46 > J >= 15: # for days between jan 15 and feb 15
				return sun *((J - 15.)/(45. - 15.) * (febSpline(t)[0] - janSpline(t)[0]) + janSpline(t)[0])+ (1.-sun)*self.t_air(t,J)
			if 75 > J >= 46:
				return sun *((J - 46.)/(74. - 46.) * (marSpline(t)[0] - febSpline(t)[0]) + febSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 106 > J >= 75:
				return sun *((J - 75.)/(105. - 75.) * (aprSpline(t) - marSpline(t)) + marSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 136 > J >= 106:
				return sun *((J - 106.)/(135. - 106.) * (maySpline(t) - aprSpline(t)) + aprSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 167 > J >= 136:
				return sun *((J - 136.)/(166. - 136.) * (junSpline(t) - maySpline(t)) + maySpline(t)) + (1.-sun)*self.t_air(t,J)
			if 197 > J >= 167:
				return sun *((J - 167.)/(196. - 167.) * (julSpline(t) - junSpline(t)) + junSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 228 > J >= 197:
				return sun *((J - 197.)/(227. - 197.) * (augSpline(t) - julSpline(t)) + julSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 259 > J >= 228:
				return sun *((J - 228.)/(258. - 228.) * (sepSpline(t) - augSpline(t)) + augSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 289 > J >= 259:
				return sun *((J - 259.)/(288. - 259.) * (octSpline(t) - sepSpline(t)) + sepSpline(t)) + (1.-sun)*self.t_air(t,J)
			if 320 > J >= 289:
				return sun *((J - 289.)/(319. - 289.) * (novSpline(t)[0] - octSpline(t)[0]) + octSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 350 > J >= 320:
				return sun *((J - 320.)/(349. - 320.) * (decSpline(t)[0] - novSpline(t)[0]) + novSpline(t)[0])+ (1.-sun)*self.t_air(t,J)
			if 367 > J >= 350:
				return sun *((J - 350.)/(380. - 350.) * (janSpline(t)[0] - decSpline(t)[0]) + decSpline(t)[0]) + (1.-sun)*self.t_air(t,J)


		def long_atmos(self, t, J):
			return 53.1e-14 * (self.t_air(t, J) + 273.)**6
	
		def t_ave(self):
			temp=[]
			for i in range(24):
				temp.append(self.t_air(i, J))
			return mean(temp)
	
		#def t_ground(self, t, sun):
		#	return sun*(self.t_ave() + self.ampl * sin((pi / 12.) * (t - 8.))) + (1.- sun) * self.t_air(t)
	
		def long_ground(self, t, J, sun):
			return self.emmiss_soil * self.SIGMA * (self.t_ground(t, J, sun) + 273)**4
	
		def Ap_over_A(self, t,J,a_s, alpha_s):
			Ap = 1.+ 4. * self.h * sin(self.a_o_i(t,J,a_s,alpha_s)) / (pi * self.d)
			A= 4.+ 4. * self.h / self.d
			return Ap / A
	
		def R_abs(self, t, J, sun, a_s, alpha_s):
			return sun * self.abs_animal_short * (self.Ap_over_A(t,J,a_s,alpha_s) * self.sol_atm(t, J, a_s, alpha_s) + 0.5 *  self.sol_diffuse(t, J, a_s, alpha_s) + 0.5 * self.short_ground(t, J)) + 0.5 * self.abs_animal_long * (self.long_atmos(t, J) + self.long_ground(t, J, sun))

		def Q_rad(self, t, J):
			#return self.emmiss_animal * self.SIGMA * (self.t_air(t,J) + 273.)**4
			return self.emmiss_animal * self.SIGMA * (self.TA + 273.)**4
	
		def g_Ha(self, wind):
			#return 1.4 * 0.135 * sqrt(self.wind / self.char_dim)
			return 1.4 * 0.135 * sqrt(wind / self.char_dim)
	
		def g_r(self, t, J): #, sun):
			return 4.* self.SIGMA * ((self.t_air(t,J) + 273.)**3) / 29.3
			#return 4.* self.SIGMA * ((self.TA + 273.)**3) / 29.3
	
		def a_o_i(self, t, J, a_s, alpha_s): #angle of incidence...NEED INPUTS for a_s = slope, alpha_s = aspect  with N=0 E=90
			self.zenith(t,J)
			if 0.<self.Z<=90.:
				tmp=acos(cos(radians(a_s)) * cos(radians(self.altitude(t,J))) * cos(radians(self.azimuth(t, J) - alpha_s)) + sin(radians(a_s)) * sin(radians(self.altitude(t,J))))
				#print tmp
				if tmp<=pi/2:
					return tmp
				if tmp > pi/2.:
					return pi/2.
				else: return pi/2
			else: return pi/2
		
		def t_e(self, t, J, sun, a_s, alpha_s,veg):
			#_
			#wind = 1.0 #note, this is to mimic convection shadow effect
			#if veg == 1.: #was or not and <-THIS NEEDS TO BE ADDRESSED IN CODE!!!!
			sun = 1. - veget[int(lizard.position['y'])][int(lizard.position['x'])]
			wind = 0.1
			return self.t_ground(t, J, sun) + (self.R_abs(t, J, sun, a_s, alpha_s) - self.Q_rad(t, J)) / (29.3 * (self.g_Ha(wind) + self.g_r(t, J))) #changing tground to tair
		
		def mass(self):
			return ((self.h * 100.) * pi * (0.5 * d * 100.)**2.) / 1000.

		def update_tb(self):
			if self.tb <= self.te: 
				return self.tb + (1. - exp(-1. / self.h_t)) * (self.te - self.tb) #heating
			if self.tb  > self.te: 
				return self.tb + (1. - exp(-1. / self.c_t)) * (self.te - self.tb) #cooling

		def mei(self, x):
			#mass = 10.
			max_con=-328.7+22.82*x-0.32*x**2
			act_con= max_con * self.size /24.
			asin_de= 85.34-0.5*x
			de = (sin((3.14159/180)*asin_de))**2
			return (act_con * 0.25*de/1000.*25600.)/60. #per minute
	
		def smr(self):
			act_scope = 1. + 4. * self.moved/self.d_max #* 0.5
			return (act_scope*20.2*10**(0.038*self.tb-1.771)*self.size**0.82)/60.
	
		def net(self,x, dist=10.):
			return self.mei(x) - self.smr(x,dist)
			
			
		def mei(self):
			max_con=-328.7+22.82*self.tb-0.32*self.tb**2
			act_con= max_con * self.size /24.
			asin_de= 85.34-0.5*self.tb
			de = (sin((3.14159/180)*asin_de))**2
			return (act_con * 0.25*de/1000.*25600.)/60.
			
		def net(self):
			#E=mei(self.tb)- (dist/10.)*smr(self.tb,1)-(1.-dist/10.)*smr(self.tb,0)
			return (self.moved/self.d_max)*self.smr()+(1.-self.dist/self.d_max)*self.smr()
			#return self.mei() - self.smr()
			#return E	

		def getDisp(self):
			incre = vonmises(self.mu, self.kappa)
			c = beta(self.beta_alpha, self.beta_beta) * self.d_max
			a, b = c * sin(self.orientation + incre), c * cos(self.orientation + incre)
			return a, b, c, incre
		
		def move_reg(self, t, J, elev, slope, aspect, veget):
			incre, x1, y1 = 0., self.position['x'], self.position['y'] #, 0.
			#self.active = 1
			lenx = len(elev[0])
			leny = len(elev)
			az=radians(self.azimuth(t, J))
			al=radians(self.altitude(t, J))
			self.zenith(t,J)
			#elevation = radians(self.altitude(t,J))#90.-lizard.Z #elev[int(lizard.position['y'])][int(lizard.position['x'])]
			#self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(self.position['y'])][int(self.position['x'])], x=self.position['x'], y=self.position['y']), a_s=90.- slope[int(self.position['y'])][int(self.position['x'])], alpha_s=aspect[int(self.position['y'])][int(self.position['x'])], veg=veg) # at the moment this is just a holder from ast code
			diff_tb = 50.
			temp_c = 32.
			if self.thigh >= self.tb >= self.tlow:
				if binomial(1, self.p1): # remain still
					#self.active = 0 # x, y = x, y ... 
					self.moved = 0.
				else: # move
					for l in range(self.decisions): # needs to be one per second 
						a, b, c, incre = self.getDisp()
						while(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							#while lizplot[int(self.position['y'])][int(self.position['x'])] == -9999: #(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							a, b, c, incre = self.getDisp()
						x1 = self.position['x'] + a
						y1 = self.position['y'] + b
						az=radians(self.azimuth(t, J))
						al=radians(self.altitude(t, J))
						self.te = self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
						up_tb =  self.update_tb()
						diff_tb_new = abs(self.t_target - up_tb)
						if diff_tb_new <= diff_tb: # and c <= temp_c:
							diff_tb = diff_tb_new
							temp_c = c
							temp_x1, temp_y1 = x1, y1
						else:
							pass	
						if l == (self.decisions -1):
							self.position['x'], self.position['y'], c = temp_x1, temp_y1, temp_c
							self.dist += temp_c
							self.orientation += incre
							self.moved = c
						
			elif self.tb > self.thigh:
				if self.tb > self.thigh > self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
					if binomial(1, self.p2):
						self.active = 0 #remain still
						self.moved = 0.
					else: # move
						for l in range(self.decisions): # needs to be one per second 
							a, b, c, incre = self.getDisp()
							#if lizplot[int(self.position['y'])][int(self.position['x'])] == 0:#(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
								a, b, c, incre = self.getDisp()
							x1 = self.position['x'] + a
							y1 = self.position['y'] + b
							az=radians(self.azimuth(t, J))
							al=radians(self.altitude(t, J))
							self.te = self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
							up_tb =  self.update_tb()
							diff_tb_new = abs(self.t_target - up_tb)
							if diff_tb_new <= diff_tb: # and c <= temp_c:
								diff_tb = diff_tb_new
								temp_c = c
								temp_x1, temp_y1 = x1, y1
							else:
								pass			
							if l == (self.decisions -1):
								self.position['x'], self.position['y'], c = temp_x1, temp_y1, temp_c
								self.dist += temp_c
								self.orientation += incre
								self.moved = c
				else: # move
					for l in range(self.decisions): # needs to be one per second 
						a, b, c, incre = self.getDisp()
						#if lizplot[int(self.position['y'])][int(self.position['x'])] == 0:#(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
						while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							a, b, c, incre = self.getDisp()
						x1 = self.position['x'] + a
						y1 = self.position['y'] + b
						az=radians(self.azimuth(t, J))
						al=radians(self.altitude(t, J))
						self.te = self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
						up_tb =  self.update_tb()
						diff_tb_new = abs(self.t_target - up_tb)
						if diff_tb_new <= diff_tb:# and c <= temp_c:
							diff_tb = diff_tb_new
							temp_c = c
							temp_x1, temp_y1 = x1, y1
						else:
							pass	
						if l == (self.decisions -1):
							self.position['x'], self.position['y'], c = temp_x1, temp_y1, temp_c
							self.dist += temp_c
							self.orientation += incre
							self.moved = c

			else: #self.tb < self.tlow:
				if self.tb < self.tlow < self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[y1][x1], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
					if binomial(1, self.p3):
						self.active = 0 #remain still
						self.moved = 0.
					else: # move
						for l in range(self.decisions): # needs to be one per second 
							a, b, c, incre = self.getDisp()
							#if lizplot[int(self.position['y'])][int(self.position['x'])] == 0:#(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
								a, b, c, incre = self.getDisp()
							x1 = self.position['x'] + a
							y1 = self.position['y'] + b
							az=radians(self.azimuth(t, J))
							al=radians(self.altitude(t, J))
							self.te = self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
							up_tb =  self.update_tb()	
							diff_tb_new = abs(self.t_target - up_tb)
							if diff_tb_new <= diff_tb: # and c <= temp_c:
								diff_tb = diff_tb_new
								temp_c = c
								temp_x1, temp_y1 = x1, y1
							else:
								pass	
							if l == (self.decisions - 1):
								self.position['x'], self.position['y'], c = temp_x1, temp_y1, temp_c
								self.dist += temp_c
								self.orientation += incre
								self.moved = c
				else:
					for l in range(self.decisions): # needs to be one per second 
						a, b, c, incre = self.getDisp()
						#if lizplot[int(self.position['y'])][int(self.position['x'])] == 0:#(not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
						while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
							a, b, c, incre = self.getDisp()
						x1 = self.position['x'] + a
						y1 = self.position['y'] + b
						az=radians(self.azimuth(t, J))
						al=radians(self.altitude(t, J))
						self.te = self.t_e(t,J, sun=horizon_master(az, al, x=x1, y=y1), a_s=90.-slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
						up_tb =  self.update_tb()
						diff_tb_new = abs(self.t_target - up_tb)
						if diff_tb_new <= diff_tb: # and c <= temp_c:
							diff_tb = diff_tb_new
							temp_c = c
							temp_x1, temp_y1 = x1, y1
						else:
							pass			
						if l == (self.decisions - 1):
							self.position['x'], self.position['y'], c = temp_x1, temp_y1, temp_c
							self.dist += temp_c
							self.orientation += incre
							self.moved = c	


	# Make Maps
	#elev, slope, aspect, veget, e_range, e_fract, v_range, v_fract = make_maps(h,  high,  N,  MU,  SD)
	
	def read_it():
		x=[]
		while 1:
			line = finput.readline()
			if not line: break
			x.append(line.split())
		finput.close()
		return x

	def char2int(y):
		i=0
		while i < len(y):
			j = 0
			while j < len(y[i]):
				y[i][j] = float(y[i][j])
				j = j + 1
			i = i + 1
		return y


	site = input("site? ")
	#print "name of elev file: ",
	#name_of_file = raw_input()
	finput = open(site + "elev", "r")
	ELEV = []
	ELEV = read_it()
	HEADER1 = ELEV[:6]          #creates raster header
	ELEV = ELEV[6:]
	ELEV=char2int(ELEV)
	elev = array(ELEV, dtype=float)
	del ELEV

	#print "name of slope file: ",
	#name_of_file = raw_input()
	finput = open(site +"slope", "r")
	SLOPE = []
	SLOPE= read_it()
	HEADER2 = SLOPE[:6]          #creates raster header
	SLOPE = SLOPE[6:]
	SLOPE = char2int(SLOPE)
	slope = array(SLOPE, dtype=float)
	del SLOPE

	#print "name of aspect file: ",
	#name_of_file = raw_input()
	finput = open(site+"aspect", "r")
	ASPECT = []
	ASPECT= read_it()
	HEADER3 = ASPECT[:6]          #creates raster header
	ASPECT = ASPECT[6:]
	ASPECT = char2int(ASPECT)
	aspect = array(ASPECT, dtype=float)
	del ASPECT

	#print "name of veg class file: ",
	#name_of_file = raw_input()
	finput = open(site+"veg", "r")
	VEG = []
	VEG= read_it()
	HEADER4 = VEG[:6]          #creates raster header
	VEG = VEG[6:]
	VEG = char2int(VEG)
	veget = array(VEG, dtype=float)
	del VEG

	#print "name of plot class file: ",
	#name_of_file = raw_input()
	finput = open(site+"plot", "r")
	SHAD= []
	SHAD= read_it()
	HEADER4 = SHAD[:6]          #creates raster header
	SHAD = SHAD[6:]
	SHAD = char2int(SHAD)
	lizplot = array(SHAD, dtype=int)
	del SHAD
	
	
	#limit = N
	# Need to read in files
	
	
	
	data = np.array(elev[:])
	"""
	def horizon_master(azimuth, elevation, x, y):
		if find_horizon(azimuth,x,y) > elevation: #if true then  shaded
			#print True
			return 0. #True is shaded, so we want '0' for full sun, THOUGH IF FULL SHADE THEN WE WANT THIS TO EQUAL 1 IF RETURNING TO SUN
		else:
			#print False
			return 1.
	"""

	cellsize = 1.	
	#def find_horizon(azimuth, x, y):
	def horizon_master(azimuth, elevation, x, y):
		z_data = [data[y][x]]
		hyp = 1. #0.5 #0.25# float(elev_header[4][1])/2. # 0.25#0.25# 0.25 
		while 0 < x+ hyp*sin(azimuth) < len(data[0]) and 0 < y + hyp*cos(azimuth) < len(data):
			# print x+ hyp*sin(azimuth), y + hyp*cos(azimuth)
			x = x + hyp*sin(azimuth)/cellsize #added int
			y = y + hyp*cos(azimuth)/cellsize
			#if data[y][x] not in z_data:
			#	z_data.append(data[int(y)][int(x)])
			#if x<909 and y<729:
			#	z_data.append(data[int(y)][int(x)])
			z_data.append(data[int(y)][int(x)])
		h_max = -pi/2.
		for i in range(1, len(z_data)):
			h_angle = atan((z_data[i]-z_data[0])/(i*hyp *cellsize)) #*cellsize
			if h_angle > h_max:
				h_max = h_angle
				if h_max > elevation:
					return 0
		return 1
		#return h_max
	
		
	# Run activity sims
	"""shadows=np.copy(elev[:])
	temperatures = np.copy(elev[:])
	activity_temps = np.copy(elev[:])
	critical_temps = np.copy(elev[:])
	d_temps = np.copy(elev[:])
	performance = np.copy(elev[:])
	act_sum=[]
	crit_sum=[]
	temp_temp = []
	d_temp = []
	act_temp =[]
	crit_temp = []
	p_temp = []
	temp_percents, d_percents, p_percents=[],[],[]"""

	times = np.arange(7.,21.,1./60.).tolist()
	pop_size = 100
	population = [T_e() for i in range(pop_size)]
	for ind in range(len(population)):
		population[ind].id = ind
		

	
	output = open(name_of_sim+".csv", 'a')
	output2 = open(name_of_sim2+".csv", 'a')
	
	def sim_act(t,J):
		lizard.zenith(t,J)
		lizard.active=0
		#az=radians(lizard.azimuth(t, J))
		#al=radians(lizard.altitude(t, J))
		if lizard.Z > 0.:
			lizard.te = lizard.t_e(t,J, sun=horizon_master(radians(lizard.azimuth(t, J)), radians(lizard.altitude(t,J)), x=lizard.position['x'], y=lizard.position['y']), a_s=90.-slope[lizard.position['y']][lizard.position['x']], alpha_s=aspect[lizard.position['y']][lizard.position['x']], veg=veget[lizard.position['y']][lizard.position['x']])
			#print lizard.zenith(t,J)
			if lizard.tb < lizard.tlow:#lizard.tb < lizard.te < lizard.tlow:
				lizard.moved = 0.
				lizard.active = 0.
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				#lizard.tot_dist_moved += lizard.moved
		
			elif lizard.tlow <= lizard.tb <=lizard.thigh: #<=lizard.ctmax:
				lizard.move_reg(t,J,elev, slope, aspect, veget)
				lizard.active = 1.
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				lizard.tot_dist_moved += lizard.moved
				lizard.ate += binomial(1, 0.05)
				lizard.total_activity += 1.
				
			else:
				if lizard.te > lizard.thigh:
					lizard.moved = 0.
					lizard.active = 0.
					lizard.energy_balance += lizard.smr()
					#lizard.tot_dist_moved += lizard.moved
				else: 
					lizard.active = 0.
					lizard.tb = lizard.update_tb()
					lizard.energy_balance += lizard.smr()
					#lizard.tot_dist_moved += lizard.moved
					#lizard.ate += binomial(1, 0.05)
					#lizard.total_activity += 1.
			#if lizard.tlow <= lizard.tb <=lizard.thigh:
			#	lizard.active = 1
			#else:
			#	lizard.active = 0	
			output.write(str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\t"+str(lizard.id)+"\n")
	
	#for trial in [0, 1]: # trial 0 is for 'burn in'
	for t in times:
		print(J, t)
		[sim_act(t,J) for lizard in population]
		


		tot_act, tot_smr, tot_move, tot_ave_ate =0.,0.,0., 0.
		for lizard in population:
			tot_act += float(lizard.total_activity)
			tot_smr += lizard.energy_balance
			tot_move += lizard.tot_dist_moved
			tot_ave_ate += float(lizard.ate)
		output2.write(str(t) + "\t" +str(tot_act/len(population)) +"\t" + str(tot_smr/len(population))+"\t" + str(tot_move/len(population))+"\t"+str(tot_ave_ate/len(population))+"\n")

		for lizard in population:
			lizard.energy_balance = 0.
			lizard.moved = 0.
			#lizard.tb = lizard.t_e(7.,J, sun=horizon_master(radians(lizard.azimuth(t, J)), radians(lizard.altitude(t,J)), x=lizard.position['x'], y=lizard.position['y']), a_s=90.-slope[lizard.position['y']][lizard.position['x']], alpha_s=aspect[lizard.position['y']][lizard.position['x']], veg=veget[lizard.position['y']][lizard.position['x']])
			lizard.tot_dist_moved = 0.
			lizard.ate = 0.
			lizard.total_activity = 0.

st1= time()
name_of_sim="details"#raw_input("name of file? ")
output = open(name_of_sim+".csv", 'w')
name_of_sim2="summary"#raw_input("name of file? ")
output2 = open(name_of_sim2+".csv", 'w')
output2.write("t\active\tsmr\tmoved\tate\n")
output2.close()
output.write("time\tX\tY\tte\ttb\tactive\tmei\tsmr\tnet\tmoved\tate\tid\n")
output.close()

"""
jobs = []
for x in range(256):
	p=multiprocessing.Process(target=do_sim, args=())
	jobs.append(p)
	p.start()
for x in range(10):
	jobs[x].join()
"""
"""
from multiprocessing import Pool
pool = Pool(32)
for _ in range(256): #was 1000 == number of maps
	pool.apply_async(do_sim, ())
pool.close()

pool.join()
"""



do_sim()
output.close()
output2.close()
st2 = time()
print(st2-st1)

