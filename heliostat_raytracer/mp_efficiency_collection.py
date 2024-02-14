from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp

from heliostat_field import *
from plotting import *
from experimental_params import experimental_params as exp

# System dimensions
hstat_sep = 0.452 # Heliostat posts seperated by 45.2cm
heliostat_width = 0.57 # Mirrors are seperated by 28.5cm
mirror_size = 0.025
receiver_pos = np.array([-0.365, 0, 0.585])
receiver_size = np.array((0.23, 0.28))
ylim = (-1, 2)
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 

# System parameters
hstat_layout = [2, 2]
tilt_deg = -10

worker_threads = 16

if __name__ == "__main__":
    mp.freeze_support()
    # hstats = create_heliostat_field(hstat_sep, hstat_layout)
    hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

    raycasts = 4000**2
    beam_size = 2.5
    start_height = 0.25

    # Incident ray only in -x direction (-x, 0, -1)
    # elevations = np.arange(10, 80, 5)
    elevations = np.array((45, 65))
    azimuths = np.arange(-80, 80, 10)

    for i, elevation in enumerate(elevations):
        args = []

        for i, azimuth in enumerate(azimuths):
            incident_vec = -1*vector_from_elevation_azimuth(azimuth, elevation)
            args.append([hstats, incident_vec, receiver_pos, heliostat_width, receiver_size, 
                        mirror_size, beam_size, raycasts, start_height, tilts, (-1, 2),
                        f"data/images/{elevation}_{azimuth}_16Mrays_points.png"])

        # with mp.Pool(worker_threads) as pool:
        with mp.Pool() as pool:
            efficiency_results = pool.starmap(mphelper_efficiency_imagegen, args)

        efficiencies = []
        for result in efficiency_results:
            efficiencies.append(result)

        np.savetxt(f'data/{elevation}_16M_efficiencies.csv', np.column_stack((azimuths, np.array(efficiencies))), delimiter=',')

