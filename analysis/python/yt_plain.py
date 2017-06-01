#!/usr/bin/env python

import yt
from yt.funcs import mylog
import yaml
import numpy as np
import psutil

# Wavenumber to load?
number = '0'
# Is it partial?
parital = False


numcpus = psutil.cpu_count(logical=False)
mylog.setLevel(40)

# Load configuration data
with open('wafer.yaml') as config_file:
    config = yaml.safe_load(config_file)

num = config['grid']['size']
dn = config['grid']['dn']
dt = config['grid']['dt']

x = (dn*num['x']-dn)/2
y = (dn*num['y']-dn)/2
z = (dn*num['z']-dn)/2

# Load potential data

potdata = np.transpose(np.loadtxt('potential.csv', delimiter=',', usecols=3).reshape((num['x'], num['y'], num['z'])), (1,2,0))

# Load wavefunction data

wave_file = []
if parital:
    wave_file = 'wavefunction_' + number + '_partial.csv'
else:
    wave_file = 'wavefunction_' + number + '.csv'

wavedata = np.abs(np.transpose(np.loadtxt(wave_file, delimiter=',', usecols=3).reshape((num['x'], num['y'], num['z'])), (1,2,0)))

# Build yt structure
data = dict(potential = (potdata, 'eV'), wavefunction = wavedata)
bbox = np.array([[-x, x], [-y, y], [-z, z]])
ds = yt.load_uniform_grid(data, potdata.shape, bbox=bbox, length_unit=1, nprocs=numcpus)

# Plot slices in y
slc = yt.SlicePlot(ds, 'y', ['potential', 'wavefunction'])
slc.set_log('wavefunction' , False)
slc.set_log('potential' , False)
slc.set_axes_unit('m')
slc.set_cmap('potential', 'Blues')
slc.annotate_grids(cmap=None)
slc.save()

# Volume renders. These are very finiky, so you need to alter the values case by case.
# Documentation can be found here: http://yt-project.org/doc/visualizing/volume_rendering.html
#Find the min and max of the field
mi, ma = ds.all_data().quantities.extrema('wavefunction')
#Reduce the dynamic range
mi = mi.value + 0.004
#ma = ma.value - 0.002
# Choose a vector representing the viewing direction.
L = [0.4, 0.5, 0.15]
# Define the center of the camera to be the domain center
c = ds.domain_center[0]
# Define the width of the image
W = 1.5*max(ds.domain_width)

sc = yt.create_scene(ds, 'wavefunction')
dd = ds.all_data()
source = sc[0]
source.log_field = False
tf = yt.ColorTransferFunction((mi, ma), grey_opacity=False)
tf.map_to_colormap(mi, ma, scale=0.5, colormap='YlGnBu')
source.set_transfer_function(tf)
sc.add_source(source)
cam = sc.add_camera()
sc.annotate_domain(ds, color=[1, 1, 1, 0.1])
sc.annotate_axes(alpha=0.5)
cam.width = W
cam.resolution = 1024
cam.center = c
cam.normal_vector = L
cam.north_vector = [0, 0, 1]
cam.switch_orientation()
sc.save(sigma_clip=4.0)
