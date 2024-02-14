from matplotlib import pyplot as plt
import numpy as np

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from heliostat_field import *
from plotting import *

from heliostat_field import experimental_params as exp

hstats = create_heliostat_field(exp.HELIOSTAT_SEPERATION.value, [2, 2])
azimuth = 60
elevation = 45 # Limit seems to be at 32deg
# incident_vec = norm_vector(np.array((-1*np.cos(elevation*np.pi/180), 0, -1)))
incident_vec = -1*vector_from_elevation_azimuth(azimuth, elevation)

"""tilts = [
    -0.02, -0.1,
    -0.1, -0.1,
    -0.1, -0.1,
    -0.1, -0.1
]"""

tilt_deg = -10
tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

model = align_heliostat_field(hstats, incident_vec, exp.RECEIVER_POSITION.value, exp.HELIOSTAT_WIDTH.value, tilts=tilts)
model = create_geometry(model, exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value, exp.YLIM.value)
""" 
fig, ax = heliostat_field_figure(model, scale=0.3)
plt.show()

fig, ax = target_plane_figure(model)
plt.show() """

raycasts = 300**2
beam_size = 2.0
start_height = 0.2

model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)
efficiency = calculate_collection_fraction(model)
print(f"Elevation angle: {elevation}, Azimuth: {azimuth} had collection efficiency {efficiency*100}%")

fig, ax = show_system(model)
ax.axes.set_xlim3d(left=-0.6, right=0.6) 
ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
ax.axes.set_zlim3d(bottom=-0.6, top=0.6) 
plt.show()

fig, ax = show_target_plane(model)
plt.show()
