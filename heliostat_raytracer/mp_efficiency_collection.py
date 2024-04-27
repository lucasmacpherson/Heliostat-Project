from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import pickle as pkl

from hstat import mphelper_alldata_imagegen
from model.heliostat_field import create_heliostat_field
from experimental_params import experimental_params as exp

# System dimensions
# Receiver is located 8.7cm back, at a height of 58.5cm. Heliostat receivers are
# located 20.3cm 
hstat_sep = exp.HELIOSTAT_SEPERATION.value # Heliostat posts seperated by 45.2cm
mirror_sep = exp.MIRROR_SEPERATION.value # Mirrors are seperated by 28.5cm
mirror_size = exp.MIRROR_RADIUS.value
receiver_pos = exp.RECEIVER_POSITION.value
receiver_size = exp.RECEIVER_SIZE.value

# System parameters
hstat_layout = [2, 2]
tilt_deg = -10

# Source incidence parameters
system_extent = np.array([
    np.array((0.3, 0.5, -0.3)),
    np.array((-0.2, -0.5, 0.2))
])
source_dist = exp.SOURCE_DISTANCE.value
# 8, 15 for experimental range
theta_range = 45 * np.pi/180
phi_range = 45 * np.pi/180

# Uniform incidence parameters
beam_size = 2.0
start_height = 0.2

# Process parameters
raycasts = 25_000_000
worker_threads = 6
run_name = "4hst_maxrange_idealtilt_25Mrays"

if __name__ == "__main__":
    mp.freeze_support()
    hstats = create_heliostat_field(hstat_sep, hstat_layout)
    # hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
    # tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))
    tilts = 'ideal'

    # elevations = np.array((15, 30, 45, 60))
    azimuths = np.arange(0, 190, 10)
    elevations = np.arange(0, 90, 5)
    # azimuths = np.array((-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70))
    # azimuths = np.array((0, 15, 30, 45, 60, 70))

    data = {
        'collection_fractions': {},
        'ideal_mirror_tilts': {},
        'heliostat_thetas': {},
        'mirror_thetas': {}
    }

    args = []
    for i, elevation in enumerate(elevations):
        for j, azimuth in enumerate(azimuths):
            args.append([hstats, elevation, azimuth, receiver_pos, mirror_sep, receiver_size, 
                        mirror_size, source_dist, system_extent, raycasts, (theta_range, phi_range), 
                        tilts, (-1, 2), f"data/images/{run_name}/{elevation}_{azimuth}_intensity.png"])

    with mp.Pool(worker_threads) as pool:
        result_lst = pool.starmap(mphelper_alldata_imagegen, args)

    for i, elevation in enumerate(elevations):
        for j, azimuth in enumerate(azimuths):
            idx = (i*len(elevations) + j)
            result = result_lst[idx]

            key = (elevation, azimuth)
            data['collection_fractions'][key] = result['collection_fraction']
            data['ideal_mirror_tilts'][key] = result['ideal_mirror_tilts']
            data['heliostat_thetas'][key] = result['heliostat_thetas']
            data['mirror_thetas'][key] = result['mirror_thetas']

    with open(f'data/{run_name}_last.pkl', 'wb') as file:
        pkl.dump(data, file)