from pathlib import Path

from model.heliostat_field import align_heliostat_field, create_geometry
from output.images import intensity_image
from raytracing.raytracer import \
    generate_source_incidence, generate_uniform_incidence, get_rays_at_target, run_raytracer
from raytracing.vector import vector_from_azimuth_elevation
from experimental_params import experimental_params as exp

def raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height):
    model['incident_vector'] = incident_vec
    initial_rays = generate_uniform_incidence(beam_size, raycasts, start_height, incident_vec)
    rays = run_raytracer(model, initial_rays)
    model['rays'] = rays
    return model

def raytrace_source_incidence(model, source_dist, incident_vec, system_extent, raycasts):
    model['incident_vector'] = incident_vec
    initial_rays = generate_source_incidence(source_dist, incident_vec, system_extent, raycasts)
    rays = run_raytracer(model, initial_rays)
    model['rays'] = rays
    return model

def calculate_collection_fraction(model):
    raycasts = len(model['rays'])
    tgt_plane_rays = get_rays_at_target(model)
    return len(tgt_plane_rays) / raycasts

def mphelper_efficiency(hstats, incident_elev, incident_azi, receiver_pos, mirror_sep, receiver_size, mirror_size, beam_size, raycasts, start_height, tilts=None, ylim=(-1, 2)):
    incident_vec = -1*vector_from_azimuth_elevation(incident_azi, incident_elev)
    model = align_heliostat_field(hstats, incident_vec, receiver_pos, mirror_sep, tilts=tilts)
    model = create_geometry(model, receiver_size, mirror_size, ylim)
    model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)

    return calculate_collection_fraction(model)

def mphelper_efficiency_imagegen(hstats, incident_elev, incident_azi, receiver_pos, mirror_sep, receiver_size, mirror_size, beam_size, start_height, raycasts, tilts=None, ylim=(-1, 2), fname=''):
    incident_vec = -1*vector_from_azimuth_elevation(incident_azi, incident_elev)
    model = align_heliostat_field(hstats, incident_vec, receiver_pos, mirror_sep, tilts=tilts)
    model = create_geometry(model, receiver_size, mirror_size, ylim)
    print(f"Raytracing system with elev={incident_elev}, azim={incident_azi}...")
    model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)
    # model = raytrace_source_incidence(model, source_dist, incident_vec, system_extent, raycasts)
    collection_frac = calculate_collection_fraction(model)
    
    # model = prune_rays(model)
    if fname != '':
        img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
        img.save(fname)

    return collection_frac

def mphelper_alldata_imagegen(hstats, incident_elev, incident_azi, receiver_pos, mirror_sep, receiver_size, mirror_size, source_dist, system_extent, raycasts, tilts=None, ylim=(-1, 2), fname=''):
    incident_vec = -1*vector_from_azimuth_elevation(incident_azi, incident_elev)
    model = align_heliostat_field(hstats, incident_vec, receiver_pos, mirror_sep, tilts=tilts)
    model = create_geometry(model, receiver_size, mirror_size, ylim)
    print(f"Raytracing system with elev={incident_elev}, azim={incident_azi}...")
    model = raytrace_source_incidence(model, source_dist, incident_vec, system_extent, raycasts)

    if fname != '':
        img = intensity_image(model, exp.CAMERA_IMAGESIZE.value)
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        img.save(fname)

    result = {
        'collection_fraction': calculate_collection_fraction(model),
        'ideal_mirror_tilts': model['ideal_mirror_tilts'],
        'heliostat_thetas': model['heliostat_thetas'],
        'mirror_thetas': model['mirror_thetas']
    }
    
    return result