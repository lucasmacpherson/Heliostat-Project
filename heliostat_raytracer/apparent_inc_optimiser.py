import numpy as np
from scipy import optimize as opt

from images import intensity_image
from vector import cartesian_to_spherical, norm_vector, vector_from_azimuth_elevation

from heliostat_field import align_heliostat_field, calculate_collection_fraction, create_geometry, mphelper_efficiency
from raytracer import generate_source_incidence, prune_rays, run_raytracer
from experimental_params import experimental_params as exp

# increase receiver size and optimise for minimum average distance from centre

def efficiency_helper(deltas, azim, elev, hstats, receiver_pos, mirror_sep, tilts, receiver_size, mirror_size, source_dist, system_extent, raycasts, save_image=False):
    apparent_inc_vec = -1*vector_from_azimuth_elevation(azim + deltas[0], elev + deltas[1])
    incident_vec = -1*vector_from_azimuth_elevation(azim, elev)

    model = align_heliostat_field(hstats, apparent_inc_vec, receiver_pos, mirror_sep, tilts)
    model = create_geometry(model, receiver_size, mirror_size)
    initial_rays = generate_source_incidence(source_dist, incident_vec, system_extent, raycasts)
    model['incident_vector'] = incident_vec
    model['rays'] = run_raytracer(model, initial_rays, max_bounces=100)
    collect_frac = calculate_collection_fraction(model)

    return collect_frac

def realscale_objective_func(deltas, azim, elev, fname=""):
    hstats = [np.array([0.226, -0.226, 0]), np.array([0.226, 0.226, 0])]
    tilt_deg = -10
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))
    system_extent = np.array([
        np.array((0.3, 0.5, -0.3)),
        np.array((-0.2, -0.5, 0.2))
    ])
    raycasts = (100, 4000) # (100, 2000) ~ 44s per iteration -> 6 hours

    collect_frac = efficiency_helper(deltas, azim, elev, hstats, exp.RECEIVER_POSITION.value,
                                     exp.MIRROR_SEPERATION.value, tilts,
                                     exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value,
                                     exp.SOURCE_DISTANCE.value,
                                     system_extent, raycasts, save_image=True)
    
    return -collect_frac

def get_optimized_deltas(azim, elev, bounds):
    result = opt.differential_evolution(lambda x: realscale_objective_func(x, azim, elev), 
                                        bounds=bounds, maxiter=4, popsize=64)
    return result.x
