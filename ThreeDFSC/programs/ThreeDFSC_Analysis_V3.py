#!/usr/bin/env python
# -*- coding: UTF-8 -*-
### Require Anaconda3

### ============================
### 3D FSC Software Package
### Analysis Section
### Written by Yong Zi Tan and Dmitry Lyumkis
### Downloaded from https://github.com/nysbc/Anisotropy
### 
### See Paper:
### Addressing preferred specimen orientation in single-particle cryo-EM through tilting
### 10.1038/nmeth.4347
###
### Credits:
### 1) UCSF Chimera, especially Tom Goddard
### 2) mrcfile 1.0.0 by Colin Palmer (https://github.com/ccpem/mrcfile)
###
### Version 3.0 (23 July 2017)
### 
### Revisions 
### 1.0 - Created analysis program
### 2.0 - Combined with plotting, thresholding and sphericity
### 3.0 - New thresholding algorithm that scales better with large box sizes
### ============================
version = "3.0"

#pythonlib
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import mrcfile
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scipy.ndimage
import scipy.misc
from math import sqrt, acos, atan2, cos, sin, pi
from skimage import measure
from scipy.ndimage.filters import gaussian_filter

## Progress bar, adapted from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

def print_progress(iteration, total, prefix='', suffix='', decimals=1):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration	- Required	: current iteration (Int)
		total		- Required	: total iterations (Int)
		prefix		- Optional	: prefix string (Str)
		suffix		- Optional	: suffix string (Str)
		decimals	- Optional	: positive number of decimals in percent complete (Int)
		bar_length	- Optional	: character length of bar (Int)
	"""
	
	rows, columns = os.popen('stty size', 'r').read().split()
	bar_length = int(float(columns)/2)
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total))) ## adjusted base on window size
	bar = '█' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\x1b[2K\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

	#if iteration == total:
	#	sys.stdout.write('\n')
	sys.stdout.flush()

def cartesian_to_spherical(vector):
	"""Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

	The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


	@param vector:  The Cartesian vector [x, y, z].
	@type vector:   numpy rank-1, 3D array
	@return:        The spherical coordinate vector [r, theta, phi].
	@rtype:         numpy rank-1, 3D array
	"""

	# The radial distance.
	r = np.linalg.norm(vector)

	# Unit vector.
	unit = vector / r

	# The polar angle.
	theta = acos(unit[2])

	# The azimuth.
	phi = atan2(unit[1], unit[0])

	# Return the spherical coordinate vector.
	return np.array([r, theta, phi], np.float64)

def spherical_to_cartesian(spherical_vect, cart_vect):
	"""Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

	The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


	@param spherical_vect:  The spherical coordinate vector [r, theta, phi].
	@type spherical_vect:   3D array or list
	@param cart_vect:       The Cartesian vector [x, y, z].
	@type cart_vect:        3D array or list
	"""

	# Trig alias.
	sin_theta = sin(spherical_vect[1])

	# The vector.
	cart_vect[0] = spherical_vect[0] * cos(spherical_vect[2]) * sin_theta
	cart_vect[1] = spherical_vect[0] * sin(spherical_vect[2]) * sin_theta
	cart_vect[2] = spherical_vect[0] * cos(spherical_vect[1])

	return cart_vect

def StandardDeviation(input):
	input_num = [float(c) for c in input]
	mean = sum(input_num) / len(input_num)
	diff = [a - mean for a in input_num]
	sq_diff = [b ** 2 for b in diff]
	ssd = sum(sq_diff)
	variance = ssd / (len(input_num) - 1)
	sd = sqrt(variance)
	return sd
	
def Mean(input):
	input_num = [float(c) for c in input]
	mean = sum(input_num) / len(input_num)
	return mean
	
def convert_highpassfilter_to_Fourier_Shells(ThreeDFSC,apix,highpassfilter):

	a = open("Results_" + ThreeDFSC + "/ResEM" + ThreeDFSC + "OutglobalFSC.csv","r")
	b = a.readlines()
	b.pop(0)
	
	globalspatialfrequency = []
	globalfsc = []

	for i in b:
		k = (i.strip()).split(",")
		globalspatialfrequency.append(float(k[0]))
		globalfsc.append(float(k[2]))

	for i in range(len(globalspatialfrequency)):
		if ((1.0/globalspatialfrequency[i])*apix) <= highpassfilter:
			highpassfilter_fouriershell = i-1
			break
	
	if highpassfilter_fouriershell < 0:
		highpassfilter_fouriershell = 0

	return highpassfilter_fouriershell

def calculate_distance(p1,p2):
    return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)	

def threshold_binarize(inmrc, thresholded, thresholdedbinarized, FSCCutoff, ThresholdForSphericity, highpassfilter, apix):

	# Thresholds
	cutoff_fsc = float(FSCCutoff)
	cutoff_binarize = float(ThresholdForSphericity)
	min_cutoff = min(cutoff_fsc,cutoff_binarize)

	# Read MRC
	inputmrc = (mrcfile.open(inmrc)).data
	
	# Coordinates
	center = (inputmrc.shape[0]/2,inputmrc.shape[1]/2,inputmrc.shape[2]/2)
	radius = int(inputmrc.shape[0]/2 + 0.5)
		
	# Fill up new np array
	boxsize = inputmrc.shape[0]
	outarraythresholded = np.zeros((boxsize,)*3)
	outarraythresholdedbinarized = np.zeros((boxsize,)*3)
	
	# Find distance of all points to center
	points_array = []
	for i in range(boxsize):
		for j in range(boxsize):
			for k in range(boxsize):
				dist = calculate_distance((i,j,k),center)
				points_array.append([dist,i,j,k])
				
		print_progress(int(i)+1,int(boxsize))
	sys.stdout.write('\n') # Flush progress bar
		
	# Sort array
	points_array.sort()
	
	# Threshold each point locally
	counter = 0
	total_iterations = len(points_array)
	number_of_progress_bar_updates = 200
	iterations_per_progress_bar_update = int(total_iterations/number_of_progress_bar_updates)
	
	memory_inmrc_thresholded = np.copy(inputmrc)
	memory_inmrc_thresholdedbinarized = np.copy(inputmrc)
	
	for i in points_array:
		x = i[1]
		y = i[2]
		z = i[3]
		
		if i[0] < highpassfilter: # Implement high pass filter
			outarraythresholded[x][y][z] = inputmrc[x][y][z]
			outarraythresholdedbinarized[x][y][z] = 1
			memory_inmrc_thresholded[x][y][z] = 1
			memory_inmrc_thresholdedbinarized[x][y][z] = 1			
		
		elif memory_inmrc_thresholded[x][y][z] < min_cutoff: # If value is smaller than the lowest cutoff, skip the calculations below to speed things up
			outarraythresholded[x][y][z] = 0
			outarraythresholdedbinarized[x][y][z] = 0
			memory_inmrc_thresholded[x][y][z] = 0
			memory_inmrc_thresholdedbinarized[x][y][z] = 0
		
		else:
			twentysix_neighboring_points = [[calculate_distance((x-1,y,z),center),x-1,y,z]]
			twentysix_neighboring_points.append([calculate_distance((x,y-1,z),center),x,y-1,z])
			twentysix_neighboring_points.append([calculate_distance((x,y,z-1),center),x,y,z-1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y-1,z),center),x-1,y-1,z])
			twentysix_neighboring_points.append([calculate_distance((x-1,y,z-1),center),x-1,y,z-1])
			twentysix_neighboring_points.append([calculate_distance((x,y-1,z-1),center),x,y-1,z-1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y-1,z-1),center),x-1,y-1,z-1])
			
			twentysix_neighboring_points.append([calculate_distance((x+1,y,z),center),x+1,y,z])
			twentysix_neighboring_points.append([calculate_distance((x,y+1,z),center),x,y+1,z])
			twentysix_neighboring_points.append([calculate_distance((x,y,z+1),center),x,y,z+1])
			twentysix_neighboring_points.append([calculate_distance((x+1,y+1,z),center),x+1,y+1,z])
			twentysix_neighboring_points.append([calculate_distance((x+1,y,z+1),center),x+1,y,z+1])
			twentysix_neighboring_points.append([calculate_distance((x,y+1,z+1),center),x,y+1,z+1])
			twentysix_neighboring_points.append([calculate_distance((x+1,y+1,z+1),center),x+1,y+1,z+1])
			
			twentysix_neighboring_points.append([calculate_distance((x+1,y-1,z),center),x+1,y-1,z])
			twentysix_neighboring_points.append([calculate_distance((x+1,y,z-1),center),x+1,y,z-1])
			twentysix_neighboring_points.append([calculate_distance((x+1,y-1,z-1),center),x+1,y-1,z-1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y+1,z),center),x-1,y+1,z])
			twentysix_neighboring_points.append([calculate_distance((x,y+1,z-1),center),x,y+1,z-1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y+1,z-1),center),x-1,y+1,z-1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y,z+1),center),x-1,y,z+1])
			twentysix_neighboring_points.append([calculate_distance((x,y-1,z+1),center),x,y-1,z+1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y-1,z+1),center),x-1,y-1,z+1])
			
			twentysix_neighboring_points.append([calculate_distance((x+1,y+1,z-1),center),x+1,y+1,z-1])
			twentysix_neighboring_points.append([calculate_distance((x+1,y-1,z+1),center),x+1,y-1,z+1])
			twentysix_neighboring_points.append([calculate_distance((x-1,y+1,z+1),center),x-1,y+1,z+1])
			
			twentysix_neighboring_points.sort()
			
			#Closest point to center
			closest_x = twentysix_neighboring_points[0][1]
			closest_y = twentysix_neighboring_points[0][2]
			closest_z = twentysix_neighboring_points[0][3]
			
			if memory_inmrc_thresholded[x][y][z] < cutoff_fsc:
				outarraythresholded[x][y][z] = 0
				memory_inmrc_thresholded[x][y][z] = 0			
			elif memory_inmrc_thresholded[closest_x][closest_y][closest_z] < cutoff_fsc:
				outarraythresholded[x][y][z] = 0
				memory_inmrc_thresholded[x][y][z] = 0
			else:
				outarraythresholded[x][y][z] = inputmrc[x][y][z]
			
			if memory_inmrc_thresholdedbinarized[x][y][z] < cutoff_binarize:
				outarraythresholdedbinarized[x][y][z] = 0
				memory_inmrc_thresholdedbinarized[x][y][z] = 0			
			elif memory_inmrc_thresholdedbinarized[closest_x][closest_y][closest_z] < cutoff_binarize:
				outarraythresholdedbinarized[x][y][z] = 0
				memory_inmrc_thresholdedbinarized[x][y][z] = 0
			else:
				outarraythresholdedbinarized[x][y][z] = 1			
			

		counter += 1
		if counter % iterations_per_progress_bar_update == 0:
			print_progress(counter+1,int(total_iterations))
	
	sys.stdout.write('\n') # Flush progress bar
	
	mrc_write = mrcfile.new(thresholded,overwrite=True)
	mrc_write.set_data(outarraythresholded.astype('<f4'))
	mrc_write.voxel_size = (float(apix),float(apix),float(apix))
	mrc_write.update_header_from_data()
	mrc_write.close()
			
	mrc_write = mrcfile.new(thresholdedbinarized,overwrite=True)
	mrc_write.set_data(outarraythresholdedbinarized.astype('<f4'))
	mrc_write.voxel_size = (float(apix),float(apix),float(apix))
	mrc_write.update_header_from_data()
	mrc_write.close()

def calculate_sphericity(inmrc):
	# read MRC
	inputmrc = (mrcfile.open(inmrc)).data
	inputmrc_copy = copy.deepcopy(inputmrc)
	extended_inputmrc = np.zeros((inputmrc.shape[0]+10,inputmrc.shape[1]+10,inputmrc.shape[2]+10),dtype=np.float) ## Had to extend it before Gaussian filter, else you might edge effects
	extended_inputmrc[6:6+inputmrc.shape[0], 6:6+inputmrc.shape[1], 6:6+inputmrc.shape[2]] = inputmrc_copy
	
	# Gaussian filtering
	# Sigma=1 works well
	blurred = gaussian_filter(extended_inputmrc,sigma=1)

	# Find surfaces using marching cube algorithm
	verts, faces, normals, values = measure.marching_cubes_lewiner(blurred,level=0.5) ## Fixed thresholded due to Gaussian blurring
	
	# Find surface area
	surface_area = measure.mesh_surface_area(verts,faces)

	# Find volume
	blurred[blurred >= 0.5] = 1
	blurred[blurred < 0.5] = 0
	volume = np.sum(blurred)
	
	# Calculate sphericity
	sphericity = (((pi)**(1/3))*((6*volume)**(2/3)))/(surface_area)
	return sphericity
	
def histogram_sample(inmrc, highpassfilter):
	
	# read MRC
	inputmrc = (mrcfile.open(inmrc)).data
	
	# coordinates
	center = [inputmrc.shape[0]/2,inputmrc.shape[1]/2,inputmrc.shape[2]/2]
	radius = int(inputmrc.shape[0]/2 + 0.5)
		
	# fill up new numpy array
	boxsize = inputmrc.shape[0]
	outarray = np.zeros((boxsize,)*3)
	outarray2 = np.zeros((boxsize,)*3)
	
	histogram_sampling = np.empty([radius,10*10]) # write out the histogram 1D FSCs
	counter = 0
	
	for theta in np.arange(1,360,36):
		#print("theta: %d" % (theta))
		for phi in np.arange(1,360,36):
			setrest = False
			for r in range(radius):
				
				# convert polar to cartesian and read mrc
				spherical_vect = [r, theta, phi]
				cart_vect = spherical_to_cartesian(spherical_vect, [0,0,0])
				cart_vect_new = np.add(cart_vect, center)

				x = int(cart_vect_new[0])
				y = int(cart_vect_new[1])
				z = int(cart_vect_new[2])
				
				# binarize
				# if setrest is True, everything beyond this radius value should be 0
				if (r > int(highpassfilter)):
					histogram_sampling[r][counter] = inputmrc[x][y][z]
				else:
					histogram_sampling[r][counter] = 1
			counter += 1
			
	return histogram_sampling
	
def HistogramCreation(histogram_sampling,histogram,ThreeDFSC,apix,cutoff,sphericity,global_resolution):
	
	stddev = []
	mean = []
	for i in histogram_sampling:
		stddev.append(StandardDeviation(i))
		mean.append(Mean(i))
	#print (stddev)
	#print (mean)
	
	stdplusone = [mean[a] + stddev[a] for a in range(len(mean))]
	stdminusone = [mean[a] - stddev[a] for a in range(len(mean))]
	
	## Open Global FSC
	
	a = open("Results_" + ThreeDFSC + "/ResEM" + ThreeDFSC + "OutglobalFSC.csv","r")
	b = a.readlines()
	b.pop(0)
	
	globalspatialfrequency = []
	globalfsc = []
	
	for i in b:
		k = (i.strip()).split(",")
		globalspatialfrequency.append(float(k[0])/apix)
		globalfsc.append(float(k[2]))
	#print (len(globalspatialfrequency))
	maxrange = max(globalspatialfrequency)
	minrange = min(globalspatialfrequency)
	
	## Calculate Sum of Standard Deviation
	## http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
	
	sumofvar = 0
	for a in stddev:
		sumofvar += a ** 2
	sumofstd = sqrt(sumofvar)
	
	#print ("\n\n")
	#print ("Sum of Standard Deviation is %s" % sumofstd)
	#print ("\n\n")
	
	## Histogram
	
	histogramlist = []
	
	for i in range(len(histogram_sampling[0])):
		for j in range(len(histogram_sampling)):
			if float(histogram_sampling[j][i]) < cutoff: ##Changed to 0.5
				break
			else:
				output = globalspatialfrequency[j]
		histogramlist.append(float(output))
	
	#print (histogramlist)
	
	## Plotting
	plt.title("Histogram and Directional FSC Plot for %s \n Sphericity = %0.3f out of 1. Global resolution = %0.2f $\AA$.\n \n \n \n" % (str(ThreeDFSC),sphericity,global_resolution))
	ax1 = plt.subplot(111)
	ax1.set_xlim([minrange,maxrange])
	n, bins, patches = plt.hist(histogramlist, bins=10, range=(minrange,maxrange))
	ax1.set_ylabel("Percentage of Per Angle FSC (%)", color="#0343df")
	for tl in ax1.get_yticklabels():
		tl.set_color("#0343df")

	ax2 = ax1.twinx()
	ax2.set_ylim([0,1])
	ax2.set_xlim([minrange,maxrange])
	ax2.plot(globalspatialfrequency, globalfsc, linewidth=3, color="#e50000")
	ax2.plot(globalspatialfrequency, stdplusone, linewidth=1, linestyle="--", color="#15b01a")
	ax2.plot(globalspatialfrequency, stdminusone, linewidth=1, linestyle="--", color="#15b01a")
	ax2.plot((minrange,maxrange), (cutoff, cutoff), linestyle="--", color="#929591")
	ax2.set_ylabel("Directional Fourier Shell Correlation", color='#e50000')
	for tl in ax2.get_yticklabels():
		tl.set_color("r")
		
	blue_patch = mpatches.Patch(color="#0343df", label="Histogram of Directional FSC")
	red_solid_line = mlines.Line2D([], [], color="#e50000", linewidth=3, label="Global FSC")
	green_dotted_line = mlines.Line2D([], [], color="#15b01a", linestyle="--", label="$\pm$1 S.D. from Mean of Directional FSC")
	#box = ax1.get_position()
	#ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	#ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	ax1.legend(handles=[blue_patch, green_dotted_line, red_solid_line],loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2)
	xlabel = ax1.set_xlabel("Spatial Frequency ($\AA^{-1}$)")

	#plt.show()
	plt.savefig("Results_" + ThreeDFSC + "/" + histogram + ".pdf", bbox_extra_artists=[xlabel], bbox_inches="tight")
	
	#Flush out plots
	plt.clf()
	plt.cla()
	plt.close()
	
	## Return useful values for ChimeraOutputCreate
	# Max Res, Min Res, global spatial frequency list, global FSC list
	return(1/float(max(histogramlist)),1/float(min(histogramlist)),globalspatialfrequency,globalfsc)
	
def ChimeraOutputCreate(ThreeDFSC,apix,maxRes,minRes,fullmap,globalspatialfrequency,globalfsc,global_resolution):
	
	## Generate Lineplot.py File		

	with open(os.path.realpath(__file__)[:-24] + "Chimera/lineplot_template.py") as f:
		replaced1 = f.read().replace("#==apix==#", str(apix))
		replaced2 = replaced1.replace("#==maxres==#", str(maxRes))
		replaced3 = replaced2.replace("#==minres==#", str(minRes))
		replaced4 = replaced3.replace("#==global_x==#", str(globalspatialfrequency))
		replaced5 = replaced4.replace("#==global_y==#", str(globalfsc))
		replaced6 = replaced5.replace("#==global_res==#",str(global_resolution))

	with open("Results_" + str(ThreeDFSC) + "/Chimera/lineplot.py", "w") as f:
		f.write(replaced6)
	
	## Obtain origins for maps

	# read MRCS
	input3DFSC = (mrcfile.open("Results_" + str(ThreeDFSC) + "/Chimera/" + ThreeDFSC + ".mrc")).data
	inputFullMap = (mrcfile.open(fullmap)).data #Full maps can be anywhere
	
	# coordinates
	center3DFSC = str(int(input3DFSC.shape[0]/2)) + "," + str(int(input3DFSC.shape[1]/2)) + "," + str(int(input3DFSC.shape[2]/2))
	centerFullMap = str(int(inputFullMap.shape[0]/2)) + "," + str(int(inputFullMap.shape[1]/2)) + "," + str(int(inputFullMap.shape[2]/2))
	
	## 3DFSCPlot_Chimera.cmd File

	with open(os.path.realpath(__file__)[:-24] + "Chimera/3DFSCPlot_Chimera_Template.cmd") as f:
		replaced1 = f.read().replace("#===3DFSC====#", str(os.path.basename(ThreeDFSC)) + ".mrc")
		replaced2 = replaced1.replace("#==apix==#", str(apix))
		replaced3 = replaced2.replace("#==Origin3DFSC==#", str(center3DFSC))
		replaced4 = replaced3.replace("#==OriginFullMap==#", str(centerFullMap))
		replaced5 = replaced4.replace("#===FullMap====#", str(os.path.basename(fullmap)))

	with open("Results_" + str(ThreeDFSC) + "/Chimera/3DFSCPlot_Chimera.cmd", "w") as f:
		f.write(replaced5)


def check_globalFSC(ThreeDFSC,apix):

    directory = "Results_" + ThreeDFSC
    if not os.path.exists(directory):
        os.makedirs(directory)
    a = open("Results_" + ThreeDFSC + "/ResEM" + ThreeDFSC + "OutglobalFSC.csv","r")
    b = a.readlines()
    b.pop(0)
    
    globalspatialfrequency = []
    globalfsc = []

    for i in b:
            k = (i.strip()).split(",")
            globalspatialfrequency.append(float(k[0]))
            globalfsc.append(float(k[2]))
    
    shells_below_pt143 = 0
    total_shells_past_first_pt143 = 0
    resolution_below_pt143 = []
    
    for i in range(len(globalfsc)):
            if (float(globalfsc[i]) < 0.143):
                    shells_below_pt143 += 1
                    resolution_below_pt143.append((1/(globalspatialfrequency[i]))*apix)
            if (shells_below_pt143 > 0):
                    total_shells_past_first_pt143 += 1
    
    if (shells_below_pt143 == 0):
            print ("\n\033[1;31;40mWarning: Your global half-map FSC does not fall below 0.143. You may have reached the Nyquist sampling limit. Try unbinning your data. \033[0;37;40m")
            resolution_below_pt143.append(apix)

    if (shells_below_pt143 != total_shells_past_first_pt143):
            print ("\n\033[1;31;40mWarning: Your glboal half-map FSC rises above 0.143 after the first crossing. Check your refinement and masking. \033[0;37;40m")
    
    return resolution_below_pt143[0] ## Returns global resolution
    
def main(halfmap1,halfmap2,fullmap,apix,ThreeDFSC,dthetaInDegrees,histogram,FSCCutoff,ThresholdForSphericity,HighPassFilter):
    # Part 00
    # Warnings and checks. Invisible to user unless something is wrong
    global_resolution = check_globalFSC(ThreeDFSC,apix)

    # Part 01
    print ("\n\033[1;34;40mAnalysis Step 01: Generating thresholded and thresholded + binarized maps. \033[0;37;40m")
    print ("These maps can be used to make figures, and are required for calculating sphericity.")
    FourierShellHighPassFilter = convert_highpassfilter_to_Fourier_Shells(ThreeDFSC,apix,HighPassFilter)
    threshold_binarize("Results_" + ThreeDFSC + "/ResEM" + ThreeDFSC + "Out.mrc", "Results_" + ThreeDFSC + "/" + ThreeDFSC + "_Thresholded.mrc", "Results_" + ThreeDFSC + "/" + ThreeDFSC + "_ThresholdedBinarized.mrc", FSCCutoff, ThresholdForSphericity,FourierShellHighPassFilter,apix)
    print ("Results_" + ThreeDFSC + "/" + ThreeDFSC + "_Thresholded.mrc at " + str(FSCCutoff) + " cutoff and Results_" + ThreeDFSC + "/" + ThreeDFSC + "_ThresholdedBinarized.mrc at " + str(ThresholdForSphericity) + " cutoff for sphericity generated.")
    # Part 02
    print ("\n\033[1;34;40mAnalysis Step 02: Calculating sphericity. \033[0;37;40m")
    sphericity = calculate_sphericity("Results_" + ThreeDFSC + "/" + ThreeDFSC + "_ThresholdedBinarized.mrc")
    print ("Sphericity is %f out of 1. 1 represents a perfect sphere." % (sphericity))
    # Part 03
    print ("\n\033[1;34;40mAnalysis Step 03: Generating Histogram. \033[0;37;40m")
    histogram_sampling = histogram_sample("Results_" + ThreeDFSC + "/ResEM" + ThreeDFSC + "Out.mrc",FourierShellHighPassFilter)
    # Part 04
    maxRes, minRes, globalspatialfrequency, globalfsc = HistogramCreation(histogram_sampling,histogram,ThreeDFSC,apix,FSCCutoff,sphericity,global_resolution)
    print ("Results_" + ThreeDFSC + "/" + histogram + ".pdf generated.")
    # Part 05
    print ("\n\033[1;34;40mAnalysis Step 04: Generating Output Files for Chimera Viewing of 3DFSC \033[0;37;40m")
    directory = "Results_" + str(ThreeDFSC)
    if not os.path.exists(directory):
        print("Directory did not exist, creating directory now.")
        os.makedirs(directory)
    else:
        print("Directory exists.")
    os.system("mkdir Results_" + str(ThreeDFSC) + "/Chimera")
    os.system("cp Results_" + str(ThreeDFSC) + "/" + str(ThreeDFSC) + ".mrc " + " Results_" + str(ThreeDFSC) + "/Chimera/")
    os.system("cp "+ fullmap + " Results_" + str(ThreeDFSC) + "/Chimera/")
    ChimeraOutputCreate(ThreeDFSC,apix,maxRes,minRes,fullmap,globalspatialfrequency,globalfsc,global_resolution)
    print ("Results_" + str(ThreeDFSC) + "/Chimera/3DFSCPlot_Chimera.cmd and Results_" + str(ThreeDFSC) + "/Chimera/lineplot.py generated.")
    print ("To view in Chimera, open 3DFSCPlot_Chimera.cmd in Chimera, with lineplot.py and the mrc files in the Chimera folder in the same directory.")
    
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10])
