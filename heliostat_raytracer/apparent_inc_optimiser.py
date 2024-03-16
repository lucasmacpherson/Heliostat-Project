import numpy as np
from scipy import optimize as opt

from vector import vector_from_azimuth_elevation

from heliostat_field import align_heliostat_field, calculate_collection_fraction, create_geometry, mphelper_efficiency
from raytracer import generate_source_incidence, prune_rays, run_raytracer
from experimental_params import experimental_params as exp


# Either optimise for collection fraction (many values will be zero), or
# increase receiver size and optimise for minimum average distance from centre

def efficiency_helper(apparent_inc_vec, hstats, receiver_pos, mirror_sep, tilts, receiver_size, mirror_size, source_dist, incident_vec, system_extent, raycasts):
    model = align_heliostat_field(hstats, apparent_inc_vec, receiver_pos, mirror_sep, tilts)
    model = create_geometry(model, receiver_size, mirror_size)
    initial_rays = generate_source_incidence(source_dist, incident_vec, system_extent, raycasts)
    model['incident_vector'] = incident_vec
    model['rays'] = run_raytracer(model, initial_rays, max_bounces=100)
    collect_frac = calculate_collection_fraction(model)

    return collect_frac

def realscale_objective_func(apparent_inc_vec, incident_vec):
    hstats = [np.array([0.226, -0.226, 0]), np.array([0.226, 0.226, 0])]
    tilt_deg = -10
    tilts = np.array([tilt_deg * np.pi/180]).repeat(2*len(hstats))
    system_extent = np.array([
        np.array((0.3, 0.5, -0.3)),
        np.array((-0.2, -0.5, 0.2))
    ])
    raycasts = (200, 4000)

    collect_frac = efficiency_helper(apparent_inc_vec, hstats, exp.RECEIVER_POSITION.value,
                                     exp.MIRROR_SEPERATION.value, tilts,
                                     exp.RECEIVER_SIZE.value, exp.MIRROR_RADIUS.value,
                                     exp.SOURCE_DISTANCE.value, incident_vec,
                                     system_extent, raycasts)
    
    if collect_frac == 0:
        return 1e16
    
    return 1/collect_frac

def calculate_bounds(incident_vec, angle_deg):
    # Normalizing the incident vector
    norm_incident_vec = incident_vec / np.linalg.norm(incident_vec)
    
    # Calculate maximum deviation
    # For a 90 degree limit, we use the length of the incident vector as max deviation.
    max_deviation = np.linalg.norm(incident_vec) * np.cos(np.radians(angle_deg))

    # Calculate bounds
    lower_bounds = norm_incident_vec * max_deviation - np.abs(norm_incident_vec * max_deviation)
    upper_bounds = norm_incident_vec * max_deviation + np.abs(norm_incident_vec * max_deviation)

    return list(zip(lower_bounds, upper_bounds))

def get_optimized_vec(incident_vec):
    bounds = calculate_bounds(incident_vec, angle_deg=90)
    result = opt.differential_evolution(lambda x: realscale_objective_func(x, incident_vec), bounds)
    return result.x
