import numpy as np
import multiprocessing as mp
import pickle as pkl

from apparent_inc_optimiser import get_optimized_deltas, realscale_objective_func
from heliostat_field import align_heliostat_field, calculate_collection_fraction, create_geometry
from images import intensity_image
from raytracer import generate_source_incidence, run_raytracer
from experimental_params import experimental_params as exp
from vector import vector_from_azimuth_elevation

def raytrace_highres_system(apparent_vec, incident_vec, fname=""):
    # Really need to stop copying this block around !
    hstats = [np.array([0.226, -0.226, 0]), np.array([0.226, 0.226, 0])]
    tilt_deg = -10
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))
    system_extent = np.array([
        np.array((0.3, 0.5, -0.3)),
        np.array((-0.2, -0.5, 0.2))
    ])
    raycasts = (2500, 10000)

    model = align_heliostat_field(hstats, apparent_vec, exp.RECEIVER_POSITION.value, exp.MIRROR_SEPERATION.value, tilts)
    model = create_geometry(model, exp.RECEIVER_POSITION.value, exp.MIRROR_RADIUS.value)
    initial_rays = generate_source_incidence(exp.SOURCE_DISTANCE.value, incident_vec, system_extent, raycasts)
    model['incident_vector'] = incident_vec
    model['rays'] = run_raytracer(model, initial_rays, max_bounces=100)
    collect_frac = calculate_collection_fraction(model)

    if fname != "":
        img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
        img.save(fname)

    return collect_frac

def optimised_result_helper(azim, elev, fname=""):
    print(f"Finding deltas for a={azim}, B={elev}...")
    bounds = [(-45, 45), (-45, 45)]
    deltas = get_optimized_deltas(azim, elev, bounds)
    
    apparent_inc_vec = -1*vector_from_azimuth_elevation(azim + deltas[0], elev + deltas[1])
    incident_vec = -1*vector_from_azimuth_elevation(azim, elev)

    # print(f"Raytracing aligned system for a={azim}, B={elev}...")
    # collect_frac = raytrace_highres_system(apparent_inc_vec, incident_vec, fname)
    # print(f"Image saved for a={azim}, B={elev}...")

    return deltas

def datagen_angle_range(elevations, azimuths, worker_threads=8):
    all_deltas = {}
    collect_fracs = {}

    for i, elev in enumerate(elevations):
        args = []

        for j, azim in enumerate(azimuths):
            args.append([azim, elev, f"data/images/{elev}_{azim}_25Mrays_aligned.png"])

        with mp.Pool(worker_threads) as pool:
            results = pool.starmap(optimised_result_helper, args)

        for result in results:
            all_deltas[(elev, azimuths[j])] = result
            # collect_fracs[(elev, azimuths[j])] = result[1]

    return all_deltas, collect_fracs

def datasave_angle_range(all_deltas, collect_fracs):
    with open('data/deltas_25Mrays_aligned_last.pkl', 'wb') as file:
        pkl.dump(all_deltas, file)

    with open('data/eff_25Mrays_aligned_last.pkl', 'wb') as file:
        pkl.dump(collect_fracs, file)

if __name__ == "__main__":
    elevations = np.array((15, 30, 45, 55, 60))
    azimuths = np.array((-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70))

    # all_deltas, collect_fracs = datagen_angle_range(azimuths, elevations, worker_threads=8)
    all_deltas = datagen_angle_range(elevations, azimuths)
    with open('data/deltas_25Mrays_aligned_last.pkl', 'wb') as file:
        pkl.dump(all_deltas, file)