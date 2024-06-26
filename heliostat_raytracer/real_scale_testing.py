from matplotlib import pyplot as plt
import numpy as np

from hstat import calculate_collection_fraction, raytrace_source_incidence, raytrace_uniform_incidence
from model.heliostat_field import *
from output.plotting import *
from output.images import *

from experimental_params import experimental_params as exp

hstats = create_heliostat_field(exp.HELIOSTAT_SEPERATION.value, [2, 2])
# hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
system_extent = np.array([
    np.array((0.6, 0.6, 0.2)),
    np.array((-0.6, -0.6, -0.2))
])

azimuth = 0
elevation = 45
# incident_vec = norm_vector(np.array((-1*np.cos(elevation*np.pi/180), 0, -1)))
incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)

"""tilts = [
    -0.02, -0.1,
    -0.1, -0.1,
    -0.1, -0.1,
    -0.1, -0.1
]"""

tilt_deg = -10
tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

model = align_heliostat_field(hstats, incident_vec, exp.RECEIVER_POSITION.value, exp.MIRROR_SEPERATION.value, tilts=tilts)
model = create_geometry(model, exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value, exp.YLIM.value)

raycasts = 1_000_000

beam_size = 3.0
start_height = 0.20

# These must remain fixed for subsequent data to be comparable
source_dist = 12
theta_range = 8 * np.pi/180
phi_range = 15 * np.pi/180

# model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)
model = raytrace_source_incidence(model, source_dist, incident_vec, system_extent, raycasts, ranges=(theta_range, phi_range))
efficiency = calculate_collection_fraction(model)
print(f"Elevation angle: {elevation}, Azimuth: {azimuth} had collection efficiency {efficiency*100}%")

fig, ax = show_system(model)
ax.axes.set_xlim3d(left=-0.6, right=0.6) 
ax.axes.set_ylim3d(bottom=-0.6, top=0.6) 
ax.axes.set_zlim3d(bottom=-0.6, top=0.6) 
plt.show()

fig, ax = show_target_plane(model)
plt.show()

img = intensity_image(model, exp.CAMERA_IMAGESIZE.value)
img.save("data/test_intensity_distribution.png")