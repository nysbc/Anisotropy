# -----------------------------------------------------------------------------
# Make a matplotlib plot of density values along a ray from center of a map.
# Use the current view direction and update plot as models are rotated.
# For Yong Zi Tan for 3D FSC plotting.
#
# This script registers command "fscplot" which takes one argument, the density
# map for which the plot is made.  For example,
#
#    fscplot #3D FSC Map #Actual Density Map
#
# Created by Tom Goddard (Thanks!)
# Modified by Yong Zi Tan
#

def ray_values(v, direction):
	d = v.data
	center = [0.5*(s+1) for s in d.size]
	radius = 0.5*min([s*t for s,t in zip(d.size, d.step)])
	steps = max(d.size)
	from Matrix import norm
	dn = norm(direction)
	from numpy import array, arange, float32, outer
	dir = array(direction)/dn
	spacing = radius/dn
	radii = arange(0, steps, dtype = float32)*(radius/steps)
	ray_points = outer(radii, dir)
	values = v.interpolated_values(ray_points)
	return radii, values, radius

# -----------------------------------------------------------------------------
#
def plot(x, y, xlabel, ylabel, title, fig = None):
	import matplotlib.pyplot as plt
	global_x = [0.005301102629346904, 0.010602205258693808, 0.015903307888040712, 0.021204410517387615, 0.026505513146734522, 0.031806615776081425, 0.03710771840542833, 0.04240882103477523, 0.047709923664122134, 0.053011026293469043, 0.058312128922815946, 0.06361323155216285, 0.06891433418150975, 0.07421543681085666, 0.07951653944020357, 0.08481764206955046, 0.09011874469889737, 0.09541984732824427, 0.10072094995759118, 0.10602205258693809, 0.11132315521628498, 0.11662425784563189, 0.12192536047497878, 0.1272264631043257, 0.13252756573367258, 0.1378286683630195, 0.1431297709923664, 0.1484308736217133, 0.15373197625106022, 0.15903307888040713, 0.16433418150975404, 0.16963528413910092, 0.17493638676844783, 0.18023748939779474, 0.18553859202714162, 0.19083969465648853, 0.19614079728583542, 0.20144189991518235, 0.20674300254452924, 0.21204410517387617, 0.21734520780322306, 0.22264631043256997, 0.22794741306191688, 0.23324851569126379, 0.23854961832061067, 0.24385072094995755, 0.2491518235793045, 0.2544529262086514, 0.2597540288379983, 0.26505513146734516, 0.2703562340966921, 0.275657336726039, 0.28095843935538595, 0.2862595419847328, 0.2915606446140797, 0.2968617472434266, 0.30216284987277353, 0.30746395250212044, 0.3127650551314673, 0.31806615776081426, 0.3233672603901611, 0.3286683630195081, 0.33396946564885494, 0.33927056827820185, 0.34457167090754875, 0.34987277353689566, 0.3551738761662426, 0.3604749787955895, 0.3657760814249364, 0.37107718405428325, 0.3763782866836302, 0.38167938931297707]
	global_y = [0.9999979700388414, 0.9999918881182148, 0.9999698435051844, 0.9999092117740581, 0.9999332089033867, 0.9995871811433383, 0.9989058809860922, 0.9980701504353575, 0.9946616211225378, 0.9912383268161188, 0.9895472530158272, 0.9880052432446349, 0.983261963399584, 0.9756362018919897, 0.9670701024434248, 0.9623682932315725, 0.9497993010705432, 0.9406875303954381, 0.9309804984758582, 0.9222794470932163, 0.9110318636114567, 0.8976593110161751, 0.8832679394726399, 0.8605818735602423, 0.83814578136705, 0.7915719474407615, 0.7649715487359858, 0.7328662302587869, 0.700655426524872, 0.6732524037045554, 0.6272436706249318, 0.5630228248349589, 0.5096018365445875, 0.48848743735147104, 0.4975847041800776, 0.4437506063680171, 0.4183987158484405, 0.3598189140780872, 0.3482535554139784, 0.31924164089992835, 0.22097412992762747, 0.16470577465510544, 0.1519369750619482, 0.13857670765717578, 0.08832471064440026, 0.04739219809832197, 0.028020536144335906, 0.030069123345227475, 0.023674465322505008, 0.02401491197749421, 0.021850473940182566, -0.00528712452344785, -0.012233221028535248, 0.02095180666507843, -0.0028307294855871, 0.02146126007145822, 0.033188063100371624, 0.014271512051111285, 0.0030172730477576013, -0.019357951024150846, 0.011642935417640546, 0.019043645704603433, -0.003950983188851543, -0.016749367252330623, -0.01978771337741378, -0.0014217418681741441, 0.01257112113976283, 0.019881431745788745, 0.010941889906292798, 0.018348311521697993, 0.0028780721304715357, 0.0049748953073480906]
	if fig is None:
		fig = plt.figure()
		fig.plot = ax = fig.add_subplot(1,1,1)
	else:
		ax = fig.plot
		ax.clear()
	plt.subplots_adjust(top=0.85)
	ax.plot(x, y, linewidth=2.0)
	ax.plot(global_x, global_y, 'r', linewidth=1.0) # Plot global FSC
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_ylim(ymin = -0.2, ymax = 1.01)
	ax.set_title(title)
	ax.grid(True)
	fig.canvas.manager.show()
	return fig

# -----------------------------------------------------------------------------
#
def update_plot(fsc_map, fig = None):
	xf = fsc_map.openState.xform
	from chimera import Vector
	direction = xf.inverse().apply(Vector(0,0,-1)).data()
	preradii, values, radius = ray_values(fsc_map, direction)
	radii = []
	apix = 1.31
	resolution_list = []
	for i in range(len(preradii)):
		radii.append(preradii[i]/(radius*2*apix))
	for i in range(len(values)):
		if values[i] < 0.143:
			resolution_list.append(1/radii[i-1])
			break
	resolution = resolution_list[0]
	#title = '3D FSC plotted on axis %.3g,%.3g,%.3g.' % direction
	title = '3D FSC Plot.\nZ directional resolution (out-of-plane in blue) is %.2f.\nGlobal resolution (in red) is %.2f.' % (resolution, 4.287272727272727)
	fig = plot(radii, values, xlabel = 'Spatial Resolution', ylabel = 'Correlation', title = title, fig = fig)
	color_map(resolution)
	return fig

# -----------------------------------------------------------------------------
#
def color_map(resolution):
	import chimera
	from chimera import runCommand
	maxres = 3.93
	minres = 6.7371428571428575
	a = (resolution-maxres)/(minres-maxres)
	r, g, b = 1-a, 0.0, a
	runCommand('color %0.2f,%0.2f,%0.2f,1.0 #1' % (r, g, b))

# -----------------------------------------------------------------------------
#
def fsc_plot(fscMap):
	fig = update_plot(fscMap)
	from chimera import triggers
	h = triggers.addHandler('OpenState', motion_cb, (fscMap, fig))

# -----------------------------------------------------------------------------
#

def motion_cb(trigger_name, mf, trigger_data):
	if 'transformation change' in trigger_data.reasons:
		fsc_map, fig = mf
		update_plot(fsc_map, fig)

# -----------------------------------------------------------------------------
#
def fscplot_cmd(cmdname, args):
	from Commands import volume_arg, parse_arguments
	req_args = [('fscMap', volume_arg)]
	kw = parse_arguments(cmdname, args, req_args)
	fsc_plot(**kw)

# -----------------------------------------------------------------------------
#
from Midas.midas_text import addCommand
addCommand('fscplot', fscplot_cmd)
