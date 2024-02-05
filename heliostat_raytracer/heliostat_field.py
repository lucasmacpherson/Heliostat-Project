import numpy as np
from tqdm import tqdm

from heliostat import *

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

def create_geometry(model, receiver_size, mirror_size):
    recevier_pos = model['receiver_position']
    mirror_norms = model['mirror_normals']
    mirrors = model['mirror_positions']

    # Rectangle represented by (centre pos, normal and (xsize, ysize))
    rects = []
    receiver_norm = np.array((0, 0, -1))
    rects.append((recevier_pos, receiver_norm, receiver_size)) # target plane

    # Circle represented by (center pos, normal and radius)
    circs = []
    for i, mirror in enumerate(mirrors):
        circs.append((mirror, mirror_norms[i], mirror_size))

    model['geometry'] = {
            'rectangles': rects,
            'circles': circs        
    }
    return model

def generate_uniform_beam(beam_size, raycasts, start_height):
    """
    Generate a square grid of evenly spaced identical initial beam points

    returns np.array(x, y, z, direction_vector)
    """
    x, y =  np.meshgrid(
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True),
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True)
    )

    return np.column_stack((x.ravel(), y.ravel(), np.full(raycasts, start_height)))
