from matplotlib import pyplot as plt
import numpy as np

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from heliostat_field import *
from plotting import *
from images import *

from experimental_params import experimental_params as exp

# hstats = create_heliostat_field(exp.HELIOSTAT_SEPERATION.value, [2, 2])
hstats = [np.array([0.226, -0.226, 0]), np.array([0.226, 0.226, 0])]
system_extent = np.array([
    np.array((0.3, 0.5, -0.3)),
    np.array((-0.2, -0.5, 0.2))
])

azimuth = 0
elevation = 30
incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)

tilt_deg = -10
tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

# Create aligned model for apparent incident vector
model = align_heliostat_field(hstats, incident_vec, exp.RECEIVER_POSITION.value, exp.MIRROR_SEPERATION.value, tilts=tilts)

print(f"Central incident vector: {incident_vec.round(3)} (alpha: {azimuth}, beta: {elevation})")
print(f"Apparent incident vector: {incident_vec.round(3)}")
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

# Uncomment block with Ctrl+/
# Creating geoemetry for raytacer
model = create_geometry(model, exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value, exp.YLIM.value)
#model = create_geometry(model, (1, 1), exp.MIRROR_RADIUS.value, exp.YLIM.value)


# Running raytracer for given source and system parameters
model = raytrace_source_incidence(model, 12, incident_vec, system_extent, (200, 1000))
# model = raytrace_uniform_incidence(model, incident_vec, beam_size=2.0, start_height=0.2, raycasts=500**2)
print(f"Collection fraction: {calculate_collection_fraction(model)}")

img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
img.save("data/last_target_image.png")

print("Creating 3D visualisation of model...")
fig, ax = show_system(model)
points = np.array([hstats[0], hstats[0] + 5*np.array([-0.652, 0.273, 0.707])])
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='orange', alpha=1)

origin = np.array((0, 0, 0))
points = np.array([origin, incident_vec])
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='red', alpha=1)

points = np.array([origin, apparent_inc_vec])
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', alpha=1)

ax.scatter(receiver_pos[0], receiver_pos[1], receiver_pos[2], color="red", marker="x")
surf = get_rectangle_surface((receiver_pos, np.array((0, 0, 1)), exp.RECEIVER_SIZE.value))
ax.plot_trisurf(surf[:, 0], surf[:, 1], surf[:, 2], linewidth=0, color='blue', alpha=0.5)

ax.axes.set_xlim3d(left=-0.6, right=0.6) 
ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
ax.axes.set_zlim3d(bottom=-0.6, top=0.6) 
plt.show()

fig, ax = target_plane_figure(model)
plt.show()

# img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
# img.save("data/last_target_image.png")