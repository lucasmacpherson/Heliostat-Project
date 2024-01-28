from matplotlib import pyplot as plt
import numpy as np

from raytracer import *
from heliostat_field import *
from plotting import *

hstat_sep = 0.452 # Heliostat posts seperated by 45.2cm
heliostat_width = 0.57 # Mirrors are seperated by 28.5cm
receiver_pos = np.array([-hstat_sep-0.087, 0, 0.585 - 0.203]) 
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 

hstats = create_heliostat_field(2*hstat_sep, [2, 2])
incident_vec = norm_vector(np.array([-0.45, 0, -1]))

"""tilts = [
    -0.02, -0.1,
    -0.1, -0.1,
    -0.1, -0.1,
    -0.1, -0.1
]"""
tilts = np.array([-0.02]).repeat(2*len(hstats))

model = align_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width=heliostat_width, tilts=tilts)
""" 
fig, ax = heliostat_field_figure(model, scale=0.3)
plt.show()

fig, ax = target_plane_figure(model)
plt.show() """

raycount = 50
beams = raytrace(model, 0.035, 0.5, raycount)

fig, ax = show_raytracer_beams(model, beams, scale=0.3)
plt.show()

fig, ax = show_beam_receiver_intersection(model, beams)
plt.show()