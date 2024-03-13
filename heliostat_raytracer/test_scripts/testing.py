from matplotlib import pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heliostat import *
from heliostat_field import *

hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
system_extent = np.array([
    np.array((0.6, 0.6, -0.6)),
    np.array((-0.6, -0.6, 0.6))
])

azimuth = 0
elevation = 45
incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)

tilt_deg = -10
tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

model = align_heliostat_field(hstats, incident_vec, exp.RECEIVER_POSITION.value, exp.HELIOSTAT_WIDTH.value, tilts=tilts)
model = create_geometry(model, exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value, exp.YLIM.value)

raycasts = 5**2
beam_size = 3.0
source_dist = 10

fig = plt.figure()
# ax = fig.add_subplot()
ax = fig.add_subplot(111, projection='3d')

init_rays = generate_source_incidence(source_dist, incident_vec, system_extent, raycasts)

ax.scatter(system_extent[:, 0], system_extent[:, 1], system_extent[:, 2], color='orange')
scale = 1
for ray in init_rays:
    ax.scatter(ray[0, 0], ray[0 ,1], ray[0, 2], color='blue', alpha=1)
    ax.quiver(ray[0, 0], ray[0, 1], ray[0, 2], 
             ray[0, 0] + ray[1, 0]*scale, ray[0, 1] + ray[1, 1]*scale, ray[0, 2] + ray[1, 2]*scale, 
             color='b')

ax.axes.set_xlim3d(left=-0.6, right=0.6) 
ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
ax.axes.set_zlim3d(bottom=-0.6, top=0.6) 
plt.grid()
plt.show()