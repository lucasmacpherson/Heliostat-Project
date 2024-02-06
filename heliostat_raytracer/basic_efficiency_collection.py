from matplotlib import pyplot as plt
import numpy as np

import multiprocessing as mp

from heliostat_field import *
from plotting import *

hstat_sep = 0.452 # Heliostat posts seperated by 45.2cm
heliostat_width = 0.57 # Mirrors are seperated by 28.5cm
mirror_size = 0.025
receiver_pos = np.array([-hstat_sep-0.087, 0, 0.585 - 0.203])
receiver_size = np.array((0.2, 0.3))
ylim = (-1, 2)
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 

hstats = create_heliostat_field(2*hstat_sep, [2, 2])

raycasts = 800**2
beam_size = 2.5
start_height = 0.5

incident_x = np.arange(0.05, 1.5, 0.1)
incident_angles = []
efficiencies = []

for i, x in enumerate(incident_x):
    incident_vec = norm_vector(np.array((x, 0, -1)))
    incident_angles.append(np.pi - np.arcsin(incident_vec.dot(np.array((0, 0, 1)))) * 180/np.pi)
    efficiencies.append(mphelper_efficiency(hstats, incident_vec, receiver_pos, heliostat_width, receiver_size, 
                 mirror_size, beam_size, raycasts, start_height, 'ideal', (-1, 2)))

np.savetxt('data/efficiencies1.csv', np.column_stack((np.array(incident_angles), np.array(efficiencies))), delimiter=',')

"""
    args.append([hstats, incident_vec, receiver_pos, heliostat_width, receiver_size, 
                 mirror_size, beam_size, raycasts, start_height, 'ideal', (-1, 2)])
"""
