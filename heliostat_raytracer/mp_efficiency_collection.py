from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import pickle as pkl

from heliostat_field import *
from plotting import *
from experimental_params import experimental_params as exp

# System dimensions
hstat_sep = exp.HELIOSTAT_SEPERATION.value # Heliostat posts seperated by 45.2cm
mirror_sep = exp.MIRROR_SEPERATION.value # Mirrors are seperated by 28.5cm
mirror_size = exp.MIRROR_RADIUS.value
receiver_pos = exp.RECEIVER_POSITION.value
receiver_size = exp.RECEIVER_SIZE.value
source_dist = exp.SOURCE_DISTANCE.value
system_extent = np.array([
    np.array((0.3, 0.5, -0.3)),
    np.array((-0.2, -0.5, 0.2))
])
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 

# System parameters
# hstat_layout = [2, 2]
tilt_deg = -10

worker_threads = 8

if __name__ == "__main__":
    mp.freeze_support()
    # hstats = create_heliostat_field(hstat_sep, hstat_layout)
    hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))

    raycasts = (1000, 25000)

    # Incident ray only in -x direction (-x, 0, -1)
    elevations = np.array((15, 30, 45, 55, 60))
    # elevations = np.array((15, 30, 45, 55, 60))
    azimuths = np.array((-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70))

    efficiencies = {}
    for i, elevation in enumerate(elevations):
        args = []

        for i, azimuth in enumerate(azimuths):
            args.append([hstats, elevation, azimuth, receiver_pos, mirror_sep, receiver_size, 
                        mirror_size, source_dist, system_extent, raycasts, tilts, (-1, 2),
                        f"data/images/{elevation}_{azimuth}_25Mrays_intensity.png"])

        with mp.Pool(worker_threads) as pool:
        # with mp.Pool() as pool:
            efficiency_results = pool.starmap(mphelper_efficiency_imagegen, args)

        for i, result in enumerate(efficiency_results):
            # pool.starmap() preserves order of arguments in results returned
            efficiencies[(elevation, azimuths[i])] = result

    with open('data/25Mrays_last.pkl', 'wb') as file:
        pkl.dump(efficiencies, file)