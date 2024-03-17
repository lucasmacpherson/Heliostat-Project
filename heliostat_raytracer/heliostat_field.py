import numpy as np
from tqdm import tqdm
from enum import Enum

from heliostat import *
from raytracer import *
from images import intensity_image
from experimental_params import experimental_params as exp
from vector import *

def create_heliostat_field(size, layout):
    """
    Generate the heliostat field as an array of vector positions, based on the provided
    layout. Scale provided is the width of the rectangular array.
    """
    layout = np.array(layout)
    heliostats = []
    try:
        xstep = size / (layout[1] - 1)
    except ZeroDivisionError: xstep = 0
    try:
        ystep = size / (layout[0] - 1)
    except ZeroDivisionError: ystep = 0

    for i in range(layout[0]):
        for j in range(layout[1]):
            xpos = xstep * j - size/2
            ypos = ystep * i - size/2
            zpos = 0
            heliostats.append(np.array([xpos, ypos, zpos]))
    
    heliostats = np.array(heliostats)
    return heliostats

def align_heliostat_field(hstats, apparent_inc_vec, receiver_pos, mirror_sep, tilts=None):
    receiver_pos = np.array(receiver_pos)
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []

    for i, hstat in enumerate(hstats):
        receiver_vec = vector_to_receiver(hstat, receiver_pos)
        init_mirror_norm = calculate_mirror_normal(receiver_vec, apparent_inc_vec)
        mirrors, offset_vecs = calculate_mirror_positions(hstat, init_mirror_norm, receiver_vec, mirror_sep)

        for j in range(2):
            idx = (2*i + j)
            mirror_positions.append(mirrors[j])
            if tilts is None:
                tilt = 0

            elif isinstance(tilts, str) and tilts == 'ideal':
                tilt = calculate_ideal_tilt(mirrors[j], receiver_pos, init_mirror_norm, apparent_inc_vec)
                print(f"mirror {idx} tilted by {tilt * 180/np.pi}")
            
            else:
                tilt = tilts[idx]

            mirror_norm = tilt_mirror_normal(init_mirror_norm, offset_vecs[j], tilt)
            mirror_norms.append(mirror_norm)
            reflected_vecs.append(calculate_reflection(mirror_norm, apparent_inc_vec))
    
    return {'heliostat_positions': hstats,
            'receiver_position': receiver_pos,
            'apparent_incidence': apparent_inc_vec,
            'mirror_positions': np.array(mirror_positions),
            'mirror_normals': np.array(mirror_norms),
            'reflected_vectors': np.array(reflected_vecs)
            }

def create_geometry(model, receiver_size, mirror_size, ylim=(-1, 2)):
    model['receiver_size'] = receiver_size
    receiver_pos = model['receiver_position']
    mirror_norms = model['mirror_normals']
    mirrors = model['mirror_positions']

    # Rectangle represented by (centre pos, normal and (xsize, ysize))
    rects = []
    receiver_norm = np.array((0, 0, -1))
    rects.append((receiver_pos, receiver_norm, receiver_size)) # target plane

    # Circle represented by (center pos, normal and radius)
    circs = []
    for i, mirror in enumerate(mirrors):
        circs.append((mirror, mirror_norms[i], mirror_size))

    model['geometry'] = {
            'rectangles': rects,
            'circles': circs        
    }
    model['ylim'] = ylim
    return model

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