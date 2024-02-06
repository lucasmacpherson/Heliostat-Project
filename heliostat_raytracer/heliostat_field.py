import numpy as np
from tqdm import tqdm

from heliostat import *
from raytracer import *

def create_heliostat_field(size, layout):
    """
    Generate the heliostat field as an array of vector positions, based on the provided
    layout. Scale provided is the width of the rectangular array.
    """
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

def align_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width, tilts=None):
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []

    for i, hstat in enumerate(hstats):
        receiver_vec = vector_to_receiver(hstat, receiver_pos)
        init_mirror_norm = calculate_mirror_normal(receiver_vec, incident_vec)
        mirrors, offset_vecs = calculate_mirror_positions(hstat, init_mirror_norm, receiver_vec, reflecting_width)

        for j in range(2):
            idx = (2*i + j)
            mirror_positions.append(mirrors[j])
            if tilts is None:
                tilt = 0

            elif isinstance(tilts, str) and tilts == 'ideal':
                tilt = calculate_ideal_tilt(mirrors[j], receiver_pos, init_mirror_norm, incident_vec)
                print(f"mirror {idx} tilted by {tilt * 180/np.pi}")
            
            else:
                tilt = tilts[idx]

            mirror_norm = tilt_mirror_normal(init_mirror_norm, offset_vecs[j], tilt)
            mirror_norms.append(mirror_norm)
            reflected_vecs.append(calculate_reflection(mirror_norm, incident_vec))
    
    return {'heliostat_positions': hstats,
            'receiver_position': receiver_pos,
            'incident_vector': incident_vec,
            'mirror_normals': np.array(mirror_norms),
            'reflected_vectors': np.array(reflected_vecs),
            'mirror_positions': np.array(mirror_positions)
            }

def create_geometry(model, receiver_size, mirror_size, ylim=(-1, 2)):
    recevier_pos = model['receiver_position']
    mirror_norms = model['mirror_normals']
    mirrors = model['mirror_positions']

    # Rectangle represented by (centre pos, normal and (xsize, ysize))
    rects = []
    receiver_norm = np.array((0, 0, -1))
    rects.append((recevier_pos, receiver_norm, receiver_size)) # target plane
    # Bounds
    # rects.append((np.array((0, 0, -1.1)), np.array((0, 0, 1)), (np.array((10, 10)))))
    # rects.append((np.array((0, 0, 2.1)), np.array((0, 0, -1)), (np.array((10, 10)))))

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
    initial_rays = generate_uniform_incidence(beam_size, raycasts, start_height, incident_vec)
    rays = run_raytracer(model, initial_rays)
    model['rays'] = rays
    return model