from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp

from heliostat_field import *
from plotting import *

# System dimensions
hstat_sep = 0.452 # Heliostat posts seperated by 45.2cm
heliostat_width = 0.57 # Mirrors are seperated by 28.5cm
mirror_size = 0.025
receiver_pos = np.array([-hstat_sep-0.087, 0, 0.555])
receiver_size = np.array((0.2, 0.3))
ylim = (-1, 2)
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 

# System parameters
hstat_layout = [2, 2]
tilt_deg = 10

if __name__ == "__main__":
    mp.freeze_support()
    hstats = create_heliostat_field(hstat_sep, hstat_layout)
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

    raycasts = 1000**2
    beam_size = 5.0
    start_height = 0.25

    # Incident ray only in -x direction (-x, 0, -1)
    incident_x = np.arange(0.05, 8.0, 1)
    incident_x_angles = np.arange(10, 90, 5)
    incident_x = np.cos(incident_x_angles*np.pi/180)
    args = []

    for i, x in enumerate(incident_x):
        incident_vec = norm_vector(np.array((x, 0, -1)))
        args.append([hstats, incident_vec, receiver_pos, heliostat_width, receiver_size, 
                    mirror_size, beam_size, raycasts, start_height, 'ideal', (-1, 2)])

    with mp.Pool() as pool:
        efficiency_results = pool.starmap(mphelper_efficiency, args)

    efficiencies = []
    for result in efficiency_results:
        efficiencies.append(result)

    np.savetxt('data/efficiencies_1M_last.csv', np.column_stack((incident_x, np.array(incident_x_angles), np.array(efficiencies))), delimiter=',')

