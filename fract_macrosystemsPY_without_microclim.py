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

seed() #99

h = 0. #input("elevational range? ") #10.
high = 1
#D = 3 #input("fractal dimension (between 2 and 3)? ") #1 # D = 3 - H, for 0 < H < 1
#cdef double H #= 3. - D
N = 100 #input("spatial extent? ") #128
MU= 0.
SD = 1.
J =  150 #input("day? ")
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

path_txt_input_folder='/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/LR2/'
finput = open(path_txt_input_folder + "LR2_elev_1m", "r")
ELEV = []
ELEV = read_it()
HEADER1 = ELEV[:8]          #creates raster header
ELEV = ELEV[8:]
ELEV=char2int(ELEV)
elev = np.array(ELEV, dtype=float)
del ELEV

finput = open(path_txt_input_folder + "LR2_slope_1m", "r")
SLOPE = []
SLOPE= read_it()
HEADER2 = SLOPE[:8]          #creates raster header
SLOPE = SLOPE[8:]
SLOPE = char2int(SLOPE)
slope = np.array(SLOPE, dtype=float)
del SLOPE

finput = open(path_txt_input_folder + "LR2_VARI_1m", "r")
VEG = []
VEG= read_it()
HEADER4 = VEG[:8]          #creates raster header
VEG = VEG[8:]
VEG = char2int(VEG)
veget = np.array(VEG, dtype=float)
del VEG

finput = open(path_txt_input_folder + "LR2_aspect_1m", "r")
ASPECT = []
ASPECT= read_it()
HEADER4 = ASPECT[:8]          #creates raster header
ASPECT = ASPECT[8:]
ASPECT = char2int(ASPECT)
aspect = np.array(ASPECT, dtype=float)
del ASPECT


#### modifying the shape of input drone files
elev = elev[elev != -9999]
shape = elev.shape[0]

if shape > (132 * 132):
  shape = (132, 132)
  elev = np.reshape(elev[:shape[0]*shape[1]], shape)
else:
  elev = np.reshape(elev, (int(np.sqrt(shape)), int(np.sqrt(shape))))
  
#### aspect  
aspect = aspect[aspect != -9999]
shape = aspect.shape[0]

if shape > (132 * 132):
  shape = (132, 132)
  aspect = np.reshape(elev[:shape[0]*shape[1]], shape)
else:
  aspect = np.reshape(aspect, (int(np.sqrt(shape)), int(np.sqrt(shape))))

#### slope  
slope = slope[slope != -9999]
shape = slope.shape[0]

if shape > (132 * 132):
  shape = (132, 132)
  slope = np.reshape(elev[:shape[0]*shape[1]], shape)
else:
  slope = np.reshape(slope, (int(np.sqrt(shape)), int(np.sqrt(shape))))
  
  
  #### veget  
veget = veget[veget != -9999]
shape = veget.shape[0]

if shape > (132 * 132):
  shape = (132, 132)
  veget = np.reshape(elev[:shape[0]*shape[1]], shape)
else:
  veget = np.reshape(veget, (int(np.sqrt(shape)), int(np.sqrt(shape))))
  
  
  
  
  


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

	days = np.array([-16, 15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381]) ## you need 14 numbers to get jan after and dec before the year you want

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

	def make_elevation(h, high, D, N, MU, SD):
		seed(100) # just to get same fractal dimensions <-normally this changes
		Arand = high
		i0, j0 = 0., 0.
		H = 3. - D
		A = np.zeros((N,N),dtype = complex) #complex number plane
		for i in range(N//2):
			for j in range(N//2):
				phase = 2. * pi  * uniform(0,high)/Arand
				if i != 0 or j != 0:
					rad = pow(i*i + j*j,-(H + 1.)/2.) * normal(MU, SD)
				else:
					rad = 0.
				A[i][j] = complex(rad * cos(phase), rad * sin(phase))
				if  i == 0.:
					i0 = 0.
				else:
					i0 = N-i #i added int(N-i)
				if j0 == 0.:
					j0 = 0.
				else:
					j0 = N-j #i added int(N-i)
				if A[int(i0)][int(j0)] == complex(rad * cos(phase), -rad*sin(phase)):
					break
			
		A[N//2][0]= complex(A[N//2][0].real, 0)
		A[0][N//2]= complex(A[N//2][0].real, 0)
		A[N//2][N//2]= complex(A[N//2][0].real, 0)

		for i in range(1, int(N/2 -1)):
			for j in range(1, int(N/2 -1)):
				phase = 2.* pi* uniform(0,high)/Arand
				rad = pow(i*i + j*j, -(H+1)/2.) * normal(MU,SD)
				A[i][N-j] = complex(rad * cos(phase), rad * sin(phase))
				A[N-i][j] = complex(rad * cos(phase), -rad * sin(phase))
		X = ifft2(A).real
		X= h * (X - min(X.real)) / (max(X.real) - min(X.real))
		z = X.view(float)
		return z

	def make_temp_vegetation(h, high, D, N, MU, SD):
		print(D)
		H = 3. - D
		temp_v = make_elevation(h, high, D, N, MU, SD)
		return temp_v

	def make_vegetation(temp_v, pcnt):
		return np.array(temp_v > percentile(temp_v, 100.-pcnt))#, int)




	# e is the center cell to a moving grid.
	# Because we want to preserve the fractal nature of the surface,
	# we must move as if over a tesselation.
	#
	#	a	b	c
	#
	#	d	e	f
	#
	#	g	h	i

	def calc_slope(elev_map, x, y, N):
		if (0<x<N-1) and (0 < y < N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == 0) and (0 < y < N-1):
			a = elev_map[y-1][N-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][N-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == 0) and (y ==  N-1):
			a = elev_map[y-1][N-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[0][N-1]
			h = elev_map[0][x]
			i = elev_map[0][x+1]
		if (x == 0) and (y == 0):
			a = elev_map[N-1][N-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][N-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == N-1) and (0 < y < N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][0]
		if (x == N-1) and (y ==  N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[0][x-1]
			h = elev_map[0][x]
			i = elev_map[0][0]
		if (x == N-1) and (y == 0):
			a = elev_map[N-1][x-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][0]
		if (0 < x < N-1) and (y == 0):
			a = elev_map[N-1][x-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (0 < x < N-1) and (y == N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[0][x-1]
			h = elev_map[0][x]
			i = elev_map[0][x+1]
		
		x_cell_size, y_cell_size = 1,1
		dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * x_cell_size)
		dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * y_cell_size)
		rise_run = sqrt(dzdx**2 +dzdy**2)
		slope_degrees = atan(rise_run) * 57.29578
		return slope_degrees

	def calc_aspect(elev_map, x, y, N):
		if (0<x<N-1) and (0 < y < N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == 0) and (0 < y < N-1):
			a = elev_map[y-1][N-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][N-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == 0) and (y ==  N-1):
			a = elev_map[y-1][N-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[0][N-1]
			h = elev_map[0][x]
			i = elev_map[0][x+1]
		if (x == 0) and (y == 0):
			a = elev_map[N-1][N-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][x+1]
			d = elev_map[y][N-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][N-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (x == N-1) and (0 < y < N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][0]
		if (x == N-1) and (y ==  N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[0][x-1]
			h = elev_map[0][x]
			i = elev_map[0][0]
		if (x == N-1) and (y == 0):
			a = elev_map[N-1][x-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][0]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][0]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][0]
		if (0 < x < N-1) and (y == 0):
			a = elev_map[N-1][x-1]
			b = elev_map[N-1][x]
			c = elev_map[N-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[y+1][x-1]
			h = elev_map[y+1][x]
			i = elev_map[y+1][x+1]
		if (0 < x < N-1) and (y == N-1):
			a = elev_map[y-1][x-1]
			b = elev_map[y-1][x]
			c = elev_map[y-1][x+1]
			d = elev_map[y][x-1]
			e = elev_map[y][x]
			f = elev_map[y][x+1]
			g = elev_map[0][x-1]
			h = elev_map[0][x]
			i = elev_map[0][x+1]
				
		dzdx = ((c + 2*f + i) - (a + 2*d + g)) / 8
		dzdy = ((g + 2*h + i) - (a + 2*b + c))  / 8
		aspect = 57.29578 * atan2(dzdy, -dzdx)
	
		if aspect < 0:
			cell = 90.0 - aspect
		elif aspect > 90.0:
			cell = 360.0 - aspect + 90.0
		else:
			cell = 90.0 - aspect
		return cell

	def make_maps(h,  high,  N,  MU,  SD):
		seed()
		elev_range = 4. #uniform(0, h)
		elev_fract = 2.2 #input("elev fract? ") #uniform(2,3)
		veg_range = uniform(0,100) #float(input("veg %? ")) #uniform(0,100)
		veg_fract = uniform(2,3) #float(input("veg fract? ")) #uniform(2,3)
		#H_elev = 3. - elev_fract
		#H_veg = 3. - veg_fract
		print(elev_range, elev_fract, veg_range, veg_fract)
		#X = make_elevation(h, high, D, H, N, MU, SD)
		X = make_elevation(elev_range, high, elev_fract, N, MU, SD)
		_slope = np.zeros((N,N),dtype = float) #[range(len(X)) for i in range(len(X[0]))]
		_aspect = np.zeros((N,N),dtype = float) #[range(len(X)) for i in range(len(X[0]))]
	
		for y in range(len(X)):
			for x in range(len(X[y])):
				_slope[y][x] = calc_slope(X, x, y, N)
				_aspect[y][x] = calc_aspect(X, x, y, N)
	
		S = _slope #possibly read-in slope here and 
		A = _aspect
		tV = make_temp_vegetation(1., high, veg_fract, N, MU, SD) # h needs to be veg specific
		V = make_vegetation(tV, veg_range)
	
		return X, S, A, V, elev_range, elev_fract, veg_range, veg_fract

	#get_topo_data(name,limit)
	#print elev_header[4][1]


	class T_e():
		def __init__(self, emissivity = 0.95, phi = 34., longitude = 74., elev = 0., h = 0.07, d = 0.2, emmiss_animal = 0.95, sky = 0.7, r_soil = 0.3, emmiss_soil = 0.95, abs_animal_long = 0.97, abs_animal_short = 0.7, Ta_min0 = 20., Ta_min1 = 20., Ta_min2 = 20., Ta_max0 = 35., Ta_max1 = 35., Ta_max2 = 35., ampl = 15., wind = 0.1):
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
			self.x_max, self.y_max = 132., 132. #x_max = ncols, y_max = nrows #### CHANGED FROM SEARS 99. for each 8feb!
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
			self.decisions = 6
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
				return sun *((J - 228.)/(258. - 228.) * (sepSpline(t)[0] - augSpline(t)[0]) + augSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 289 > J >= 259:
				return sun *((J - 259.)/(288. - 259.) * (octSpline(t)[0] - sepSpline(t)[0]) + sepSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 320 > J >= 289:
				return sun *((J - 289.)/(319. - 289.) * (novSpline(t)[0] - octSpline(t)[0]) + octSpline(t)[0]) + (1.-sun)*self.t_air(t,J)
			if 350 > J >= 320:
				return sun *((J - 320.)/(349. - 320.) * (decSpline(t)[0] - novSpline(t)[0]) + novSpline(t)[0])+ (1.-sun)*self.t_air(t,J)
			if 367 > J >= 350:
				return sun *((J - 350.)/(380. - 350.) * (janSpline(t)[0] - decSpline(t)[0]) + decSpline(t)[0]) + (1.-sun)*self.t_air(t,J)

		def long_atmos(self, t, J): #### UPDATE FOR MICROCLIM INPUT DATA...SUN Ta at 2m input!!
			return 53.1e-14 * (self.t_air(t, J) + 273.)**6
	
		def t_ave(self):
			temp=[]
			for i in range(24):
				temp.append(self.t_air(i, J))
			return np.mean(temp)
	
		#def t_ground(self, t, sun):
		#	return sun*(self.t_ave() + self.ampl * sin((pi / 12.) * (t - 8.))) + (1.- sun) * self.t_air(t)
	
		def long_ground(self, t, J, sun): #### UPDATE FOR MICROCLIM INPUT DATA... sun/shade GROUND input
			#micro = pd.read_csv('/Users/laurenneel/Desktop/Grass_RGB_DSM_Oct22/microclim_input_era5/hourly_df_TC_2021.csv', usecols=["Tg_avg"])
			#Tg_avg = micro.iloc[t, J, sun]["Tg_avg"]
			#return self.emmiss_soil * self.SIGMA * (Tg_avg + 273)**4
			#print(Tg_avg)
			return self.emmiss_soil * self.SIGMA * (self.t_ground(t, J, sun) + 273)**4
	
		def Ap_over_A(self, t,J,a_s, alpha_s):
			Ap = 1.+ 4. * self.h * sin(self.a_o_i(t,J,a_s,alpha_s)) / (pi * self.d)
			A= 4.+ 4. * self.h / self.d
			return Ap / A
	
		def R_abs(self, t, J, sun, a_s, alpha_s): #### UPDATE FOR MICROCLIM INPUT DATA..... needs to call the right ground and air temp
			return sun * self.abs_animal_short * (self.Ap_over_A(t,J,a_s,alpha_s) * self.sol_atm(t, J, a_s, alpha_s) + 0.5 *  self.sol_diffuse(t, J, a_s, alpha_s) + 0.5 * self.short_ground(t, J)) + 0.5 * self.abs_animal_long * (self.long_atmos(t, J) + self.long_ground(t, J, sun))

		def Q_rad(self, t, J):
			#return self.emmiss_animal * self.SIGMA * (self.t_air(t,J) + 273.)**4
			return self.emmiss_animal * self.SIGMA * (self.TA + 273.)**4
	
		def g_Ha(self, wind):
			#return 1.4 * 0.135 * sqrt(self.wind / self.char_dim)
			return 1.4 * 0.135 * sqrt(wind / self.char_dim)
	
		def g_r(self, t, J, sun): #### input ground data in sun/shade here!!!
			#return 4.* self.SIGMA * ((self.t_air(t,J) + 273.)**3) / 29.3
			return 4.* self.SIGMA * ((self.TA + 273.)**3) / 29.3
	
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
		
		def t_e(self, t, J, sun, a_s, alpha_s,veg): #UPDATE FOR MICROCLIM INPUT DATA .... t_ground should be replaced with Ta at 2cm... need to set it up for different sun/shade 2cm input
			#_wind = 0. #note, this is to mimic convection shadow effect
			if sun ==1 and veg == 0: #was or not and
				sun = 1
			else:
				sun = 0.375
			if veg == 1:
				wind = 0.1
			else:
				wind = 2.0
			return self.t_ground(t, J, sun) + (self.R_abs(t, J, sun, a_s, alpha_s) - self.Q_rad(t, J)) / (29.3 * (self.g_Ha(wind) + self.g_r(t, J, sun)))
		
		def mass(self):
			return ((self.h * 100.) * pi * (0.5 * self.d * 100.)**2.) / 1000.

	
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
							self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
					if self.tb > self.thigh > self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
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
								self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
							self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
					if self.tb < self.tlow < self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[y1][x1], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)]):
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
								self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
							self.te = self.t_e(t,J, sun=horizon_master(az, elev[int(y1)][int(x1)], x=x1, y=y1), a_s=90.- slope[int(y1)][int(x1)], alpha_s=aspect[int(y1)][int(x1)], veg=veget[int(y1)][int(x1)])
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
	shadows=np.copy(elev[:])
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
	temp_percents, d_percents, p_percents=[],[],[]

	times = np.arange(6.,20.,1./60.).tolist()
	#pop_size = 20
	pop_size = 1
	population = [T_e() for i in range(pop_size)] #population is a list, where each element is an instance of the "T_e" class that's assigned to variable "lizard" each iteration
    #ppulation is defined and instantiated with a list of T_e class objects before sim_act function is called
    # sim_act

	output = open(name_of_sim+".csv", 'a')
	output2 = open(name_of_sim2+".csv", 'a')


	times = np.arange(6.,20.,1./60.).tolist()

	def sim_act(lizard, t,J=150.):
		lizard.zenith(t,J)
		if lizard.Z > 0.:
			lizard.te = lizard.t_e(t,J, sun=horizon_master(radians(lizard.azimuth(t, J)), elev[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], x=lizard.position['x'], y=lizard.position['y']), a_s=90.- slope[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], alpha_s=aspect[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])], veg=veget[math.floor(lizard.position['y'])][math.floor(lizard.position['x'])])
			print(lizard.zenith(t,J))
			if lizard.tb < lizard.te < lizard.tlow:
				lizard.moved = 0.
				lizard.active = 0.
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				#lizard.tot_dist_moved += lizard.moved
		
			elif lizard.tlow <= lizard.tb <=lizard.ctmax:
				lizard.move_reg(t,J,elev, slope, aspect, veget)
				lizard.tb = lizard.update_tb()
				lizard.energy_balance += lizard.smr()
				lizard.tot_dist_moved += lizard.moved
				lizard.ate += binomial(1, 0.05)
				lizard.total_activity += 1.
				lizard.active = 1.
			else:
				if lizard.te > lizard.ctmax:
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
			print(str(lizard.position['x']), str(lizard.position['y']))
			output.write(str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")
            #output.write(str(e_range) + "\t" + str(e_fract) + "\t" + str(v_range) + "\t" + str(v_fract) +"\t" + str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")
	for t in times:
		print(J, t)
		[sim_act(lizard,t,J) for lizard in population]

#	tot_act, tot_smr, tot_move, tot_ave_ate =0.,0.,0., 0.
#	for lizard in population:
#		tot_act += float(lizard.total_activity)
#		tot_smr += lizard.energy_balance
#		tot_move += lizard.tot_dist_moved
#		tot_ave_ate += float(lizard.ate)
#	output2.write(str(t)+ "\t" + str(lizard.position['x']) + "\t" +str(lizard.position['y']) + "\t" + str(lizard.te) + "\t" +str(lizard.tb) + "\t" + str(lizard.active) +"\t" + str(lizard.mei())+"\t" + str(lizard.smr()) + "\t" + str(lizard.net())  + "\t"+ str(lizard.moved)+ "\t"+ str(lizard.ate)+"\n")
			
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

	#for t in times:
    #    print(J, t)
    #    for lizard in population:
    #        print("lizard x position:", lizard.position['x'])
    #        print("lizard y position:", lizard.position['y'])
    #        print("elev array size:", len(elev))
    #        print("slope array size:", len(slope))
    #        print("aspect array size:", len(aspect))
    #        print("veget array size:", len(veget))
    #        sim_act(lizard,t,J)
    

"""
jobs = []
for x in range(1000):
	p=multiprocessing.Process(target=do_sim, args=())
	jobs.append(p)
	p.start()
for x in range(10):
	jobs[x].join()
"""




"""
from multiprocessing import Pool
pool = Pool(8)
for _ in range(256): #was 1000 == number of maps , or the simulation number
	pool.apply_async(do_sim, ())
pool.close()

pool.join()

st2 = time()
print(st2-st1)

"""


do_sim() #to run it one 
output.close()
output2.close()