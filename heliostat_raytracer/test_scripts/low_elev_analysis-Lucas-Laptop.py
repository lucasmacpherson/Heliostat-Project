from matplotlib import pyplot as plt
import numpy as np

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from heliostat_field import *
from plotting import *
from images import *

from experimental_params import experimental_params as exp

hstats = create_heliostat_field(exp.HELIOSTAT_SEPERATION.value, [2, 2])
# hstats = [np.array([0.226, -0.226, 0]), np.array([0.226, 0.226, 0])]
system_extent = np.array([
    np.array((0.3, 0.5, -0.3)),
    np.array((-0.2, -0.5, 0.2))
])

azimuth = 0
elevation = 38
incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)

tilt_deg = -6
tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

# Create aligned model for principle incident vector
model = align_heliostat_field(hstats, incident_vec, exp.RECEIVER_POSITION.value, exp.MIRROR_SEPERATION.value, tilts=tilts)

print(f"Principle incident vector: {incident_vec.round(3)} (alpha: {azimuth}, beta: {elevation})")
receiver_pos = model['receiver_position']
for i, hstat in enumerate(hstats):
    print(f"""Heliostat {i}: (Position: {hstat.round(3)})
        -> Vector to Receiver: {vector_to_receiver(hstat, receiver_pos).round(3)}
        """
    )

mirrors = model['mirror_positions']
mirror_norms = model['mirror_normals']
for i, mirror in enumerate(mirrors):
    print(f"""Mirror {i}: (Position: {mirror.round(3)})
        -> Normal: {mirror_norms[i].round(3)}
        -> Vector to Receiver: {vector_to_receiver(mirror, receiver_pos).round(3)}
        """
    )

# Creating geoemetry for raytacer
model = create_geometry(model, exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value, exp.YLIM.value)

# Uncomment block with Ctrl+/
# Running raytracer for given source and system parameters
model = raytrace_source_incidence(model, 12, incident_vec, system_extent, (100, 800))
# model = raytrace_uniform_incidence(model, incident_vec, beam_size=2.0, start_height=0.2, raycasts=500**2)

fig, ax = show_system(model)
points = np.array([hstats[0], hstats[0] + 5*np.array([-0.652, 0.273, 0.707])])

ax.plot(points[:, 0], points[:, 1], points[:, 2], color='orange', alpha=1)
ax.axes.set_xlim3d(left=-0.6, right=0.6) 
ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
ax.axes.set_zlim3d(bottom=-0.6, top=0.6) 
plt.show()
