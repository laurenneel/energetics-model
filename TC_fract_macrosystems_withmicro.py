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
#from matplotlib.mlab import prctile as percent
from numpy import flipud
from math import exp, log, sin, cos, sqrt, acos, asin, atan, atan2
#from numpy import arctan as atan
#from numpy import arctan2 as atan2
#from numpy import arcsin as asin
#from numpy import arccos as acos
import pandas as pd ## added by LN
import math
import csv

seed() #99

h = 0. #input("elevational range? ") #10.
high = 1
#D = 3 #input("fractal dimension (between 2 and 3)? ") #1 # D = 3 - H, for 0 < H < 1
#cdef double H #= 3. - D
N = 100 #input("spatial extent? ") #128
MU= 0.
SD = 1.
#J =  150 #input("day? ")
from time import time

## read in elev, slope, aspect, veget files
## LR2
#elev = pd.DataFrame(pd.read_csv('/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/LR2/LR2_elevation'))
#elev = elev.iloc[8:]
#elev.reset_index(drop=True, inplace=True)
#
#slope = pd.DataFrame(pd.read_csv('/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/LR2/LR2_slope'))
#slope=slope.iloc[8:]
#slope.reset_index(drop=True, inplace=True)
#
#aspect = pd.DataFrame(pd.read_csv('/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/LR2/LR2_aspect'))
#aspect = aspect.iloc[8:]
#aspect.reset_index(drop=True, inplace=True)
#
#
#veget = pd.DataFrame(pd.read_csv('/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/LR2/LR2_VARI'))
#veget=veget.iloc[8:]
#veget.reset_index(drop=True, inplace=True)


def read_it():
		x=[]
		while True:
			line=finput.readline()
			if not line: break
			x.append(line.split())
		finput.close()
		return(x)


#i=0
def char2int(y):
		i=0
		while i < len(y):
			j = 0
			while j < len(y[i]):
				y[i][j] = float(y[i][j])
				j = j + 1
			i = i + 1
		return y

path_txt_input_folder='/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/TC2/'
finput = open(path_txt_input_folder + "TC2_elev_1m", "r")
ELEV = []
ELEV = read_it()
HEADER1 = ELEV[:8]          #creates raster header
ELEV = ELEV[8:]
ELEV=char2int(ELEV)
elev = np.array(ELEV, dtype=float)
del ELEV

finput = open(path_txt_input_folder + "TC2_slope_1m", "r")
SLOPE = []
SLOPE= read_it()
HEADER2 = SLOPE[:8]          #creates raster header
SLOPE = SLOPE[8:]
SLOPE = char2int(SLOPE)
slope = np.array(SLOPE, dtype=float)
del SLOPE

finput = open(path_txt_input_folder + "TC2_VARI_1m", "r")
VEG = []
VEG= read_it()
HEADER4 = VEG[:8]          #creates raster header
VEG = VEG[8:]
VEG = char2int(VEG)
veget = np.array(VEG, dtype=float)
del VEG

finput = open(path_txt_input_folder + "TC2_aspect_1m", "r")
ASPECT = []
ASPECT= read_it()
HEADER4 = ASPECT[:8]          #creates raster header
ASPECT = ASPECT[8:]
ASPECT = char2int(ASPECT)
aspect = np.array(ASPECT, dtype=float)
del ASPECT


#### need to remove all -9999 values. Then print the length of the new shape. Take the square root of that number and round down to get the number of rows and columns to use
#### TC1 - modifying the shape of input drone files
#elev = elev[elev != -9999]
#shape = elev.shape[0]
#
#if shape > (144 * 144):
#  shape = (144, 144)
#  elev = np.reshape(elev[:shape[0]*shape[1]], shape)
#else:
#  elev = np.reshape(elev, (int(np.sqrt(shape)), int(np.sqrt(shape))))
# 
#### aspect  
#aspect = aspect[aspect != -9999]
#shape1 = aspect.shape[0]
#
#if shape1 > (144 * 144):
#  shape1 = (144, 144)
#  aspect = np.reshape(aspect[:shape1[0]*shape1[1]], shape1)
#else:
#  aspect = np.reshape(aspect, (int(np.sqrt(shape1)), int(np.sqrt(shape1))))
#
##### slope  
#slope = slope[slope != -9999]
#shape2 = slope.shape[0]
#
#if shape2 > (144 * 144):
#  shape2 = (144, 144)
#  slope = np.reshape(slope[:shape2[0]*shape2[1]], shape2)
#else:
#  slope = np.reshape(slope, (int(np.sqrt(shape2)), int(np.sqrt(shape2))))
#  
#  
#  #### veget  
#veget = veget[veget != -9999]
#shape3 = veget.shape[0]
#
#if shape3 > (144 * 144):
#  shape3 = (144, 144)
#  veget = np.reshape(veget[:shape3[0]*shape3[1]], shape3)
#else:
#  veget = np.reshape(veget, (int(np.sqrt(shape3)), int(np.sqrt(shape3))))
#  
  
  
## TC2 input data
elev = elev[elev != -9999]
shape = elev.shape[0]
print(shape)

if shape > (200 * 200):
  shape = (200, 200)
  elev = np.reshape(elev[:shape[0]*shape[1]], shape)
else:
  elev = np.reshape(elev, (int(np.sqrt(shape)), int(np.sqrt(shape))))
 
### aspect  
aspect = aspect[aspect != -9999]
shape1 = aspect.shape[0]

if shape1 > (200 * 200):
  shape1 = (200, 200)
  aspect = np.reshape(aspect[:shape1[0]*shape1[1]], shape1)
else:
  aspect = np.reshape(aspect, (int(np.sqrt(shape1)), int(np.sqrt(shape1))))

#### slope  
slope = slope[slope != -9999]
shape2 = slope.shape[0]

if shape2 > (200 * 200):
  shape2 = (200, 200)
  slope = np.reshape(slope[:shape2[0]*shape2[1]], shape2)
else:
  slope = np.reshape(slope, (int(np.sqrt(shape2)), int(np.sqrt(shape2))))
  
  
  #### veget  
veget = veget[veget != -9999]
shape3 = veget.shape[0]

if shape3 > (200 * 200):
  shape3 = (200, 200)
  veget = np.reshape(veget[:shape3[0]*shape3[1]], shape3)
else:
  veget = np.reshape(veget, (int(np.sqrt(shape3)), int(np.sqrt(shape3))))
  
  


  
  
##### BRING in MICROCLIMATE DATA
micro_df = pd.read_csv("/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/microclim_input_era5/hourly_df_TC_2021.csv")
micro_df['datetime']= pd.to_datetime(micro_df['datetime'])

#micro_df_minute_res = micro_df.set_index('datetime')
#micro_df_minute_res = micro_df_minute_res.asfreq('T')
#micro_df_minute_res = micro_df_minute_res.interpolate(method='time')
#micro_df_minute_res['new_hour'] = micro_df_minute_res.index.hour
#micro_df_minute_res['ordinal_day'] = micro_df_minute_res.index.dayofyear


micro_df_minute_res = micro_df.set_index('datetime')
micro_df_minute_res = micro_df_minute_res.asfreq('T')
micro_df_minute_res = micro_df_minute_res.interpolate(method='time')
micro_df_minute_res['new_hour'] = micro_df_minute_res.index.hour
micro_df_minute_res['day_of_year'] = micro_df_minute_res.index.dayofyear       
# create an array that goes from 0 to 24 in increments of 0.0166666667 and repeat it 525601 times
time_array = np.linspace(0, 24, int(24 / 0.0166666667), endpoint=False)
micro_df_minute_res['hour_float'] = np.tile(time_array, len(micro_df_minute_res) // len(time_array) + 1)[:len(micro_df_minute_res)]

              



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
		minTemps = np.array([-5.,-5.2, -2.7, 1.3, 5.8, 11.5, 16.2, 18.5, 17.3, 12.7, 6.6, -1.0, -5.0, -5.2]) 
		maxTemps = np.array([14.4,15.2, 17.6, 21.4, 26.6, 31.2, 35.6, 36.0, 34.7, 31.2, 25.4, 18.8, 14.4, 15.2])
		g=0.
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

	#def make_elevation(h, high, D, N, MU, SD):
	#	seed(100) # just to get same fractal dimensions <-normally this changes
	#	Arand = high
	#	i0, j0 = 0., 0.
	#	H = 3. - D
	#	A = np.zeros((N,N),dtype = complex) #complex number plane
	#	for i in range(N//2):
	#		for j in range(N//2):
	#			phase = 2. * pi  * uniform(0,high)/Arand
	#			if i != 0 or j != 0:
	#				rad = pow(i*i + j*j,-(H + 1.)/2.) * normal(MU, SD)
	#			else:
	#				rad = 0.
	#			A[i][j] = complex(rad * cos(phase), rad * sin(phase))
	#			if  i == 0.:
	#				i0 = 0.
	#			else:
	#				i0 = N-i #i added int(N-i)
	#			if j0 == 0.:
	#				j0 = 0.
	#			else:
	#				j0 = N-j #i added int(N-i)
	#			if A[int(i0)][int(j0)] == complex(rad * cos(phase), -rad*sin(phase)):
	#				break
	#		
	#	A[N//2][0]= complex(A[N//2][0].real, 0)
	#	A[0][N//2]= complex(A[N//2][0].real, 0)
	#	A[N//2][N//2]= complex(A[N//2][0].real, 0)

	#	for i in range(1, int(N/2 -1)):
	#		for j in range(1, int(N/2 -1)):
	#			phase = 2.* pi* uniform(0,high)/Arand
	#			rad = pow(i*i + j*j, -(H+1)/2.) * normal(MU,SD)
	#			A[i][N-j] = complex(rad * cos(phase), rad * sin(phase))
	#			A[N-i][j] = complex(rad * cos(phase), -rad * sin(phase))
	#	X = ifft2(A).real
	#	X= h * (X - min(X.real)) / (max(X.real) - min(X.real))
	#	z = X.view(float)
	#	return z

	#def make_temp_vegetation(h, high, D, N, MU, SD):
	#	print(D)
	#	H = 3. - D
	#	temp_v = make_elevation(h, high, D, N, MU, SD)
	#	return temp_v

	#def make_vegetation(temp_v, pcnt):
	#	return np.array(temp_v > percentile(temp_v, 100.-pcnt))#, int)




	# e is the center cell to a moving grid.
	# Because we want to preserve the fractal nature of the surface,
	# we must move as if over a tesselation.
	#
	#	a	b	c
	#
	#	d	e	f
	#
	#	g	h	i



	class T_e():
		def __init__(self, emissivity = 0.95, phi = 34., longitude = 109., elev = 0., h = 0.07, d = 0.2, emmiss_animal = 0.95, sky = 0.7, r_soil = 0.3, emmiss_soil = 0.94, abs_animal_long = 0.97, abs_animal_short = 0.7, Ta_min0 = 20., Ta_min1 = 20., Ta_min2 = 20., Ta_max0 = 35., Ta_max1 = 35., Ta_max2 = 35., ampl = 15., wind = 0.1):
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
			self.size = 10.
			self.h_t = exp(log(self.size) * 0.36 + 0.72)
			self.c_t = exp(log(0.42 + 0.44 * self.size))
			self.Z=0.
			self.TA= 0.
		
			self.x_min, self.y_min = 0., 0.
			self.x_max, self.y_max = 200., 200. #x_max = ncols, y_max = nrows #### CHANGED FROM SEARS 99. for each 8feb!
			self.position = {'x':rand() * self.x_max, 'y': rand() * self.y_max}
		
			self.thigh =  39. #sears had 35.#Upper voluntary max
			self.tlow = 29. #sears had 29.
			self.topt = 33. #sears had 32.
			self.mu = 0.
			self.kappa = 0.25
			self.beta_alpha = 1.
			self.beta_beta = 3.
			self.d_max = 5.*(np.log10(self.size)+1.)
			self.p1 = 0.5
			self.p2 = 0.0
			self.p3 = 0.0
			self.decisions = 6
			self.tb = 20.
			self.te = 20.
			#self.te_min_list =[] #LN added
			#self.te_max_list =[] #LN added
			self.orientation = vonmises(self.mu, self.kappa)
			self.dist = 0.
			self.ctmax = 40.
			self.moved = 0
			self.cost = 0
			self.totenergy = 0.
			self.dist = 0.
			self.active = 0
			self.t_target = 35. #previously 32
			self.db = 0.
			self.de = 0.
			self.energy_balance = 0.
			self.moved = 0.
			self.tot_dist_moved = 0.
			self.total_activity = 0.
			self.ate = 0.
	
		def tprops(self): #thermal preoperties
			self.h_t = exp(log(self.size) * 0.36 + 0.72)
			self.c_t = exp(0.42 + 0.44 * log(self.size))

		def greybody(self, K):
			return self.emissivity * self.SIGMA * (K + 273)**4
	
        ## original MS fxn
		#def zenith(self ,t, J):	# phi = lattitde, delta = solar declination, t0 = solar noon
		#	self.Z = degrees(acos(sin(radians(self.phi)) * sin(radians(self.declination(J))) + cos(radians(self.phi)) * cos(radians(self.declination(J))) * cos(radians(15. * (t - self.noon(J))))))
		#	if self.Z <0.: #or self.Z is None:  ##LN modified 
		#		self.Z = 0.
		#	if self.Z >90: # or self.Z is None:  ##LN modified 
		#		self.Z = 90.
		#	return

		def zenith(self ,t, J):	# phi = lattitde, delta = solar declination, t0 = solar noon
			self.Z = degrees(acos(sin(radians(self.phi)) * sin(radians(self.declination(J))) + cos(radians(self.phi)) * cos(radians(self.declination(J))) * cos(radians(15. * (t - self.noon(J))))))
			if self.Z <0. or self.Z is None:  ##LN modified 
				self.Z = 0.
			if self.Z >90 or self.Z is None:  ##LN modified 
				self.Z = 90.
			return self.Z # adding return statement so it always returns a value preventing 'NoneType" error
	
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
			if self.zenith(t,J)<90.:
				s = self.SOLAR * self.bardd2(J) * cos(self.a_o_i(t, J, a_s, alpha_s))##*cos(self.a_o_i(t, J, a_s, alpha_s))#(radians(self.zenith(t, J)))#cos(radians(self.zenith(t, J)))#
				if s >= 0: return s
				else: return 0. ##LN added .
			else: return 0. ##LN added .
	
		def p_a(self):
			return 101.3 * exp(-self.elev / 8200.)
	
		def t_m(self, t, J):
			return self.sky**(self.p_a() / (101.3 * cos(radians(self.zenith(t, J)))))
			#return self.sky**(self.p_a() / (101.3 * cos(radians(self.Z))))
		
		def sol_atm(self, t, J, a_s, alpha_s): #def sol_atm(self, t, J, a_s=0, alpha_s=0):
			return self.sol_noatm(t, J, a_s, alpha_s) * self.t_m(t, J)
	
		def sol_diffuse(self, t, J, a_s, alpha_s):
			return self.sol_noatm(t, J, a_s, alpha_s) * 0.3 * (1 - self.t_m(t, J))
	
		def h_g(self, t, J, a_s, alpha_s): ##LN changed args from def h_g(self, t, J, a_s=0, alpha_s=0):
			return self.sol_diffuse(t, J, a_s, alpha_s) + self.sol_atm(t, J, a_s, alpha_s) #and self.sol_atm(t, J, a_s=0, alpha_s=0)
	
		def short_ground(self, t, J, a_s, alpha_s): 
			return self.h_g(t, J, a_s, alpha_s) * self.r_soil
		
		def gamma_t(self, t):
			return 0.44 - 0.46 * sin(t * pi / 12. + 0.9) + 0.11 * sin(2. * t * pi / 12. + 0.9)

		#def t_air(self, t, J):
		#	if (t >= 0.) and (t <= 5.):
		#		self.TA = maxT(J - 1) * self.gamma_t(self.t_adj(t, J)) + minT(J) * (1 - self.gamma_t(self.t_adj(t, J)))
		#		return self.TA
		#	if (t > 5.) and (t <= 14.):
		#		self.TA = maxT(J) * self.gamma_t(self.t_adj(t, J)) + minT(J) * (1 - self.gamma_t(self.t_adj(t, J)))
		#		return self.TA
		#	if (t > 14.) and (t <= 24.):
		#		self.TA = maxT(J) * self.gamma_t(self.t_adj(t, J)) + minT(J + 1) * (1 - self.gamma_t(self.t_adj(t, J)))
		#		return self.TA

		def t_ground(self, sun, Tg_shade, Tg_sun):
			if Tg_shade is None:
				Tg_shade = micro_df_minute_res.loc[J, "Tg_shade"].loc[t]
			if Tg_sun is None:
				Tg_sun= micro_df_minute_res.loc[J, "Tg_sun"].loc[t]
			if sun >= 0.5: # in shade
				return Tg_shade
			if sun < 0.5: # in sun
				return Tg_sun

		def long_atmos(self, Ta_2m):## UPDATE FOR MICROCLIM INPUT DATA...SUN Ta at 2m input!!
			if Ta_2m is None:
				Ta_2m = micro_df_minute_res.loc[J, "Ta_2m"].loc[t]
			#Ta_2m= micro_df.Ta_2m
			return 53.1e-14 * (Ta_2m + 273.)**6
        
		#def t_ave(self):
		#	temp=[]
		#	for i in range(24):
		#		temp.append(self.t_air(i, J))
		#	return np.mean(temp)
	
		#def t_ground(self, t, sun):
		#	return sun*(self.t_ave() + self.ampl * sin((pi / 12.) * (t - 8.))) + (1.- sun) * self.t_air(t)
	
		def long_ground(self, sun, Tg_shade, Tg_sun): #### UPDATE FOR MICROCLIM INPUT DATA... sun/shade GROUND input
			if Tg_shade is None:
				Tg_shade = micro_df_minute_res.loc[J, "Tg_shade"].loc[t]
			if Tg_sun is None:
				Tg_sun= micro_df_minute_res.loc[J, "Tg_sun"].loc[t]
			if sun >= 0.5:
				return self.emmiss_soil * self.SIGMA * (Tg_shade + 273)**4
			if sun < 0.5:
				return self.emmiss_soil * self.SIGMA * (Tg_sun + 273)**4
			#return self.emmiss_soil * self.SIGMA * (self.t_ground(sun, micro_df) + 273)**4
	
		def Ap_over_A(self, t,J,a_s, alpha_s):
			Ap = 1.+ 4. * self.h * sin(self.a_o_i(t,J,a_s,alpha_s)) / (pi * self.d)
			A= 4.+ 4. * self.h / self.d
			return Ap / A
	
		def R_abs(self, t, J, sun, a_s, alpha_s, Ta_2m, Tg_shade, Tg_sun): #### UPDATE FOR MICROCLIM INPUT DATA..... needs to call the right ground and air temp
			if Ta_2m is None:
				Ta_2m = micro_df_minute_res.loc[J, "Ta_2m"].loc[t]
			if Tg_shade is None:
				Tg_shade = micro_df_minute_res.loc[J, "Tg_shade"].loc[t]
			if Tg_sun is None:
				Tg_sun = micro_df_minute_res.loc[J, "Tg_sun"].loc[t]
			absorbed_radiation = sun * self.abs_animal_short * (self.Ap_over_A(t,J,a_s,alpha_s) * self.sol_atm(t, J, a_s, alpha_s) + 0.5 *  self.sol_diffuse(t, J, a_s, alpha_s) + 0.5 * self.short_ground(t, J, a_s, alpha_s)) + 0.5 * self.abs_animal_long * (self.long_atmos(Ta_2m) + self.long_ground(sun, Tg_shade, Tg_sun))
			return absorbed_radiation
			#return sun * self.abs_animal_short * (self.Ap_over_A(t,J,a_s,alpha_s) * self.sol_atm(t, J, a_s, alpha_s) + 0.5 *  self.sol_diffuse(t, J, a_s, alpha_s) + 0.5 * self.short_ground(t, J, a_s, alpha_s)) + 0.5 * self.abs_animal_long * (self.long_atmos(Ta_2m) + self.long_ground(sun, Tg_shade, Tg_sun))

		def Q_rad(self, Ta_shade, Ta_sun, sun):
			if Ta_shade is None: 
				Ta_shade = micro_df_minute_res.loc[J, "Ta_shade"].loc[t]
			if Ta_sun is None:
				Ta_sun = micro_df_minute_res.loc[J, "Ta_sun"].loc[t]
			if sun >= 0.5:
				return self.emmiss_animal * self.SIGMA * (Ta_shade + 273.)**4
			if sun < 0.5:
				return self.emmiss_animal * self.SIGMA * (Ta_sun + 273.)**4

			#return self.emmiss_animal * self.SIGMA * (self.t_air(t,J) + 273.)**4
			#return self.emmiss_animal * self.SIGMA * (self.TA + 273.)**4
	
		def g_Ha(self, wind):
			#return 1.4 * 0.135 * sqrt(self.wind / self.char_dim)
			return 1.4 * 0.135 * sqrt(wind / self.char_dim)
	
		def g_r(self, sun, Ta_shade, Ta_sun): #### input ground data in sun/shade here!!!
			if Ta_shade is None:
				Ta_shade = micro_df_minute_res.loc[J, "Ta_shade"].loc[t]
			if Ta_sun is None:
				Ta_sun = micro_df_minute_res.loc[J, "Ta_sun"].loc[t]
			if sun >= 0.5:
				return 4.* self.SIGMA * ((Ta_shade + 273.)**3) / 29.3
			if sun < 0.5:
				return 4.* self.SIGMA * ((Ta_sun + 273.)**3) / 29.3
	
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
		
		def t_e(self, t, J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun, a_s, alpha_s,veg): #UPDATE FOR MICROCLIM INPUT DATA .... t_ground should be replaced with Ta at 2cm... need to set it up for different sun/shade 2cm input
			#_wind = 0. #note, this is to mimic convection shadow effect
			if sun ==0 and veg == 0: #was or not and
				sun = 0
			else:
				sun = 0.375
			if veg == 1:
				wind = 0.1
			else:
				wind = 2.0
			operative_temp = self.t_ground(sun, Tg_shade, Tg_sun) + (self.R_abs(t, J, sun, a_s, alpha_s, Ta_2m, Tg_shade, Tg_sun) - self.Q_rad(Ta_shade, Ta_sun, sun)) / (29.3 * (self.g_Ha(wind) + self.g_r(sun, Ta_shade, Ta_sun)))

			return operative_temp

			return self.t_ground(sun, Tg_shade, Tg_sun) + (self.R_abs(t, J, sun, a_s, alpha_s, Ta_2m, Tg_shade, Tg_sun) - self.Q_rad(Ta_shade, Ta_sun, sun)) / (29.3 * (self.g_Ha(wind) + self.g_r(sun, Ta_shade, Ta_sun)))
		
		def mass(self):
			return ((self.h * 100.) * pi * (0.5 * self.d * 100.)**2.) / 1000.

	
		def update_tb(self):
			if self.tb <= self.te: 
				new_tb = self.tb + (1. - exp(-1. / self.h_t)) * (self.te - self.tb) #heating
				return new_tb
				#return self.tb + (1. - exp(-1. / self.h_t)) * (self.te - self.tb) #heating
			if self.tb  > self.te: 
				new_tb = self.tb + (1. - exp(-1. / self.c_t)) * (self.te - self.tb) #cooling
				return new_tb
				#return self.tb + (1. - exp(-1. / self.c_t)) * (self.te - self.tb) #cooling

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
			
		def  net(self):
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
			#self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(self.position['y'])][int(self.position['x'])], x=self.position['x'], y=self.position['y']), a_s=90.- slope[int(self.position['y'])][int(self.position['x'])], alpha_s=aspect[int(self.position['y'])][int(self.position['x'])], veg=veg) # at the moment this is just a holder from ast code
			diff_tb = 50.
			temp_c = 32.
			while elev[int(y1)][int(x1)] == -9999 or slope[int(y1)][int(x1)] == -9999 or aspect[int(y1)][int(x1)] == -9999 or veget[int(y1)][int(x1)] == -9999:
				a, b, c, incre = self.getDisp()
				if self.thigh >= self.tb >= self.tlow:
					if binomial(1, self.p1): # remain still
						#self.active = 0 # x, y = x, y ... 
						self.moved = 0.
					else: # move
						for l in range(self.decisions): # needs to be one per second 
							a, b, c, incre = self.getDisp()
							if (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
								while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
									a, b, c, incre = self.getDisp()
							x1 = self.position['x'] + a
							y1 = self.position['y'] + b
							az=radians(self.azimuth(t, J))
							al=radians(self.altitude(t, J))
							self.te = self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
					if self.tb > self.thigh > self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
						if binomial(1, self.p2):
							self.active = 0 #remain still
							self.moved = 0.
						else: # move
							for l in range(self.decisions): # needs to be one per second 
								a, b, c, incre = self.getDisp()
								if (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
									while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
										a, b, c, incre = self.getDisp()
								x1 = self.position['x'] + a
								y1 = self.position['y'] + b
								az=radians(self.azimuth(t, J))
								al=radians(self.altitude(t, J))
								self.te = self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
							if (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
								while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
									a, b, c, incre = self.getDisp()
							x1 = self.position['x'] + a
							y1 = self.position['y'] + b
							az=radians(self.azimuth(t, J))
							al=radians(self.altitude(t, J))
							self.te = self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
	#
				else: #self.tb < self.tlow:
					if self.tb < self.tlow < self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[y1][x1], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
						if binomial(1, self.p3):
							self.active = 0 #remain still
							self.moved = 0.
						else: # move
							for l in range(self.decisions): # needs to be one per second 
								a, b, c, incre = self.getDisp()
								if (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
									while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
										a, b, c, incre = self.getDisp()
								x1 = self.position['x'] + a
								y1 = self.position['y'] + b
								az=radians(self.azimuth(t, J))
								al=radians(self.altitude(t, J))
								self.te = self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
							if (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
								while (not (0.<=self.position['x']+a<=lenx-1)) or  (not (0.<=self.position['y']+b<=leny-1)):
									a, b, c, incre = self.getDisp()
							x1 = self.position['x'] + a
							y1 = self.position['y'] + b
							az=radians(self.azimuth(t, J))
							al=radians(self.altitude(t, J))
							self.te = self.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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


    ### MAKE MAPS
	#elev, slope, aspect, veget, e_range, e_fract, v_range, v_fract = make_maps(h,  high,  N,  MU,  SD)  ###only need these first 4 variables -- assign 


	
	# Save Maps
	#write_file = "maps/elev" +  "_" + '{:.4}'.format(e_fract) + "_" + '{:.3}'.format(e_range)+ "_" + '{:.3}'.format(v_fract+3.) + "_" + '{:.3}'.format(v_range)+ ".asc"
	#output=open(write_file, 'w')
	#for i in range(N):
	#	for j in range(N):
	#		output.write(str(elev[i][j])), output.write(' ')
	#	output.write("\n")
	#output.close()
	#
	#write_file = "maps/slope" +  "_" +'{:.4}'.format(e_fract) + "_" + '{:.3}'.format(e_range)+ "_" + '{:.3}'.format(v_fract+3.) + "_" + '{:.3}'.format(v_range)+ ".asc"
	#output=open(write_file, 'w')
	#for i in range(N):
	#	for j in range(N):
	#		output.write(str(slope[i][j])), output.write(' ')
	#	output.write("\n")
	#output.close()
	#
	#write_file = "maps/aspect" + "_" + '{:.4}'.format(e_fract) + "_" + '{:.3}'.format(e_range)+ "_" + '{:.3}'.format(v_fract+3.) + "_" + '{:.3}'.format(v_range)+ ".asc"
	#output=open(write_file, 'w')
	#for i in range(N):
	#	for j in range(N):
	#		output.write(str(aspect[i][j])), output.write(' ')
	#	output.write("\n")
	#output.close()
	#
	#write_file = "maps/veg" + "_" + '{:.4}'.format(e_fract) + "_" + '{:.3}'.format(e_range)+ "_" + '{:.3}'.format(v_fract+3.) + "_" + '{:.3}'.format(v_range)+ ".asc"
	#output=open(write_file, 'w')
	#for i in range(N):
	#	for j in range(N):
	#		if veget[i][j]:
	#			output.write(str(1)), output.write(' ')
	#		else:
	#			output.write(str(0)), output.write(' ')				
	#	output.write("\n")
	#output.close()




	limit = N
	data = np.array(elev[:])
	def horizon_master(azimuth, elevation, x, y):
		if find_horizon(azimuth,x,y) > elevation:
			#print True
			return 0 #True is shaded, so we want '0' for full sun
		else:
			#print False
			return 1

	cellsize = 1.	
	def find_horizon(azimuth, x, y):
		x, y=math.floor(x), math.floor(y)
		z_data = [data[y][x]]
		hyp = 0.25# float(elev_header[4][1])/2. # 0.25#0.25# 0.25 
		while 0 < x+ hyp*sin(azimuth) < len(data[0]) and 0 < y + hyp*cos(azimuth) < len(data):
			#print x+ hyp*sin(azimuth), y + hyp*cos(azimuth)
			x = x + hyp*sin(azimuth)*cellsize #added int # check whether this should be mulitplied by cell size
			y = y + hyp*cos(azimuth)*cellsize
			if data[math.floor(y)][math.floor(x)] not in z_data:
				z_data.append(data[math.floor(y)][math.floor(x)])
		h_max = -pi/2.
		for i in range(1, len(z_data)):
			h_angle = atan((z_data[i]-z_data[0])/(i*cellsize))
			if h_angle > h_max:
				h_max = h_angle
		return h_max
	
		
	# Run activity sims
	#shadows=np.copy(elev[:])
	#temperatures = np.copy(elev[:])
	#activity_temps = np.copy(elev[:])
	#critical_temps = np.copy(elev[:])
	#d_temps = np.copy(elev[:])
	#performance = np.copy(elev[:])
	#act_sum=[]
	#crit_sum=[]
	#temp_temp = []
	#d_temp = []
	#act_temp =[]
	#crit_temp = []
	#p_temp = []
	#temp_percents, d_percents, p_percents=[],[],[]

	#times = np.arange(6.,20.,1./60.).tolist() 
#	times = 
	#pop_size = 20
	pop_size = 1
	population = [T_e() for i in range(pop_size)] #population is a list, where each element is an instance of the "T_e" class that's assigned to variable "lizard" each iteration
    #ppulation is defined and instantiated with a list of T_e class objects before sim_act function is called
    # sim_act

	#output = open(name_of_sim+".csv", 'a')
	#output2 = open(name_of_sim2+".csv", 'a')


	#times = np.arange(6.,20.,1./60.).tolist() 

 
#	def sim_act(lizard, t, J):
#		lizard.zenith(t,J=J) #J=J
#		print("zenith angle: ", lizard.Z)
#		if lizard.Z > 0.:
#			Ta_sun = micro_df_minute_res.loc[index, "Ta_sun"]
#			Ta_shade = micro_df_minute_res.loc[index, "Ta_shade"]
#			Tg_sun = micro_df_minute_res.loc[index, "Tg_sun"]
#			Tg_shade = micro_df_minute_res.loc[index, "Tg_shade"]
#			Ta_2m = micro_df_minute_res.loc[index, "Ta_2m"]
#			print("Before floor y:", lizard.position['y'])
#			print("Before floor x:", lizard.position['x'])
#			lizard.te = lizard.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(radians(lizard.azimuth(t, J)), elev[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], x=lizard.position['x'], y=lizard.position['y']), a_s=90.- slope[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], alpha_s=aspect[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], veg=veget[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])])
#			print("Lizard operative temp: ", lizard.te)
#			print(lizard.zenith(t,J))
#			if lizard.tb < lizard.te < lizard.tlow:
#				print("case 1: lizard is not active and has not moved")
#				lizard.moved = 0.
#				lizard.active = 0.
#				lizard.tb = lizard.update_tb()
#				lizard.energy_balance += lizard.smr()
#				#lizard.tot_dist_moved += lizard.moved
#			elif lizard.tlow <= lizard.tb <=lizard.ctmax:
#				print("case 2: lizard is active and moves")
#				lizard.move_reg(t,J,elev, slope, aspect, veget)
#				lizard.tb = lizard.update_tb()
#				lizard.energy_balance += lizard.smr()
#				lizard.tot_dist_moved += lizard.moved
#				lizard.ate += binomial(1, 0.05)
#				lizard.total_activity += 1.
#				lizard.active = 1.
#			else:
#				if lizard.te > lizard.ctmax:
#					print("case 3: lizard is not active and has not moved")
#					lizard.moved = 0.
#					lizard.active = 0.
#					lizard.energy_balance += lizard.smr()
#					#lizard.tot_dist_moved += lizard.moved
#				else: 
#					print("case 4: lizard is not active and not moving")
#					lizard.active = 0.
#					lizard.tb = lizard.update_tb()
#					lizard.energy_balance += lizard.smr()
#					#lizard.tot_dist_moved += lizard.moved
#					#lizard.ate += binomial(1, 0.05)
#					#lizard.total_activity += 1.
#			print(str(lizard.position['x']), str(lizard.position['y']))
#			output.write(str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")
#           #output.write(str(e_range) + "\t" + str(e_fract) + "\t" + str(v_range) + "\t" + str(v_fract) +"\t" + str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")

	def sim_act(lizard, t, J):
		lizard.zenith(t,J=J) #J=J
		print("zenith angle: ", lizard.Z)
		print("Julian day is: ", J)
		if lizard.Z > 0.:
			Ta_sun = micro_df_minute_res.loc[index, "Ta_sun"]
			Ta_shade = micro_df_minute_res.loc[index, "Ta_shade"]
			Tg_sun = micro_df_minute_res.loc[index, "Tg_sun"]
			Tg_shade = micro_df_minute_res.loc[index, "Tg_shade"]
			Ta_2m = micro_df_minute_res.loc[index, "Ta_2m"]
			print("Before floor y:", lizard.position['y'])
			print("Before floor x:", lizard.position['x'])
			lizard.te = lizard.t_e(t,J, Tg_shade, Tg_sun, Ta_2m, Ta_shade, Ta_sun, sun=horizon_master(radians(lizard.azimuth(t, J)), elev[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], x=lizard.position['x'], y=lizard.position['y']), a_s=90.- slope[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], alpha_s=aspect[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], veg=veget[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])])
			print("Lizard operative temp: ", lizard.te)
			print(lizard.zenith(t,J))
			if lizard.tb < lizard.te < lizard.tlow:
				print("case 1: lizard is not active and has not moved")
				lizard.moved = 0.
				lizard.active = 0.
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				#lizard.tot_dist_moved += lizard.moved
			elif lizard.tlow <= lizard.tb <=lizard.ctmax:
				print("case 2: lizard is active and moves")
				lizard.move_reg(t,J,elev, slope, aspect, veget)
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				lizard.tot_dist_moved += lizard.moved
				lizard.ate += binomial(1, 0.05)
				lizard.total_activity += 1.
				lizard.active = 1.
			else:
				if lizard.te > lizard.ctmax:
					print("case 3: lizard is not active and has not moved")
					lizard.moved = 0.
					lizard.active = 0.
					lizard.energy_balance += lizard.smr()
					#lizard.tot_dist_moved += lizard.moved
				else: 
					print("case 4: lizard is not active and not moving")
					lizard.active = 0.
					lizard.tb = lizard.update_tb()
					lizard.energy_balance += lizard.smr()
     
	with open('TC_output.csv', mode='w', newline='') as output_file: ## ALL DATA
		fieldnames = ['time', 'J', 'x', 'y', 'te', 'tb', 'total_activity', 'mei', 'net', 'moved', 'smr', 'energy_balance']
		output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
		output_writer.writeheader()
		for index, row in micro_df_minute_res.iterrows(): 
			t = row['hour_float']
			J = row['day_of_year']
			if t >= 6 and t < 20:
				for lizard in population:
					sim_act(lizard, t, J)
					output_writer.writerow({
                    'time': t,
                    'J': J,
                    'x': lizard.position['x'],
                    'y': lizard.position['y'],
                    'te': lizard.te,
                    'tb': lizard.tb,
                    'total_activity': lizard.total_activity,
                    'mei': lizard.mei(),
                    'net': lizard.net(),
                    'moved': lizard.moved,
                    'smr': lizard.smr(),
                    'energy_balance': lizard.energy_balance
                })
     
     
	with open('TC_overall_averages.csv', mode='w', newline='') as output_file: 
		fieldnames = ['te', 'tb', 'total_activity', 'mei', 'net', 'moved', 'smr', 'energy_balance']
		output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
		output_writer.writeheader()

    # calculate overall averages
		te_avg = sum([lizard.te for lizard in population]) / len(population)
		tb_avg = sum([lizard.tb for lizard in population]) / len(population)
		total_activity_avg = sum([lizard.total_activity for lizard in population]) / len(population)
		mei_avg = sum([lizard.mei() for lizard in population]) / len(population)
		net_avg = sum([lizard.net() for lizard in population]) / len(population)
		moved_avg = sum([lizard.moved for lizard in population]) / len(population)
		smr_avg = sum([lizard.smr() for lizard in population]) / len(population)
		energy_balance_avg = sum([lizard.energy_balance for lizard in population]) / len(population)
  
		output_writer.writerow({
        'te': te_avg,
        'tb': tb_avg,
        'total_activity': total_activity_avg,
        'mei': mei_avg,
        'net': net_avg,
        'moved': moved_avg,
        'smr': smr_avg,
        'energy_balance': energy_balance_avg
    })

		with open('TC_daily_averages.csv', mode='w', newline='') as output_file:
			fieldnames = ['day_of_year', 'te_avg', 'tb_avg', 'total_activity_avg', 'mei_avg', 'net_avg', 'moved_avg', 'smr_avg', 'energy_balance_avg']
			output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
			output_writer.writeheader()

	    # iterate over days and calculate daily averages
			for day in range(1, 367):
				lizards_on_day = [lizard for lizard in population if lizard.day_of_year == day]
				print(f"Day {day}: {len(lizards_on_day)} lizards")
				if lizards_on_day:
					te_avg = sum([lizard.te for lizard in lizards_on_day]) / len(lizards_on_day)
					tb_avg = sum([lizard.tb for lizard in lizards_on_day]) / len(lizards_on_day)
					total_activity_avg = sum([lizard.total_activity for lizard in lizards_on_day]) / len(lizards_on_day)
					mei_avg = sum([lizard.mei() for lizard in lizards_on_day]) / len(lizards_on_day)
					net_avg = sum([lizard.net() for lizard in lizards_on_day]) / len(lizards_on_day)
					moved_avg = sum([lizard.moved for lizard in lizards_on_day]) / len(lizards_on_day)
					smr_avg = sum([lizard.smr() for lizard in lizards_on_day]) / len(lizards_on_day)
					energy_balance_avg = sum([lizard.energy_balance() for lizard in lizards_on_day]) / len(lizards_on_day)
					output_writer.writerow({'day_of_year': day, 'te_avg': te_avg, 'tb_avg': tb_avg, 'total_activity_avg': total_activity_avg, 'mei_avg': mei_avg, 'net_avg': net_avg, 'moved_avg': moved_avg, 'smr_avg': smr_avg, 'energy_balance_avg': energy_balance_avg})


#	for index, row in micro_df_minute_res.iterrows(): 
#		t = row['hour_float']
#		J = row['day_of_year']
#		if t >= 6 and t < 20:
#			print(J, t)
#			[sim_act(lizard, t, J) for lizard in population]


	tot_act, tot_smr, tot_move, tot_ave_ate =0.,0.,0., 0.
	for lizard in population:
		tot_act += float(lizard.total_activity)
		tot_smr += lizard.energy_balance
		tot_move += lizard.tot_dist_moved
		tot_ave_ate += float(lizard.ate)
	output2.write(str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")
			
st1= time()
name_of_sim="details"#raw_input("name of file? ")
output = open(name_of_sim+".csv", 'w')
name_of_sim2="summary"#raw_input("name of file? ")
output2 = open(name_of_sim2+".csv", 'w')
#output2.write("elev\tefract\tveg\tvfract\tactive\tsmr\tmoved\tate\n") #MS original
output2.write("elev\tactive\tsmr\tmoved\tate\n")
output2.close()
#output.write("elev\tefract\tveg\tvfract\ttime\tX\tY\tte\ttb\tactive\tmei\tsmr\tnet\tmoved\tate\n") #MS original
output.write("elev\ttime\tX\tY\tte\ttb\tactive\tmei\tsmr\tnet\tmoved\tate\n")
output.close()


do_sim() #to run it one 
output.close()
output2.close()