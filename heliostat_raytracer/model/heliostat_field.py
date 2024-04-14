import numpy as np

from model.heliostat import calculate_ideal_tilt, calculate_mirror_normal, calculate_mirror_positions, tilt_mirror_normal, vector_to_receiver
from raytracing.vector import angle_between_deg, calculate_reflection

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

def align_heliostat_field(hstats, incident_vec, receiver_pos, mirror_sep, tilts=None):
    receiver_pos = np.array(receiver_pos)
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []
    ideal_tilts = []
    hstat_thetas = []
    mirror_thetas = []

    for i, hstat in enumerate(hstats):
        receiver_vec = vector_to_receiver(hstat, receiver_pos)
        init_mirror_norm = calculate_mirror_normal(receiver_vec, incident_vec)
        mirrors, offset_vecs = calculate_mirror_positions(hstat, init_mirror_norm, receiver_vec, mirror_sep)
        hstat_thetas.append(angle_between_deg(-incident_vec, receiver_vec))

        for j in range(2):
            idx = (2*i + j)
            mirror_positions.append(mirrors[j])

            tilt = -1
            ideal_tilt = calculate_ideal_tilt(mirrors[j], receiver_pos, init_mirror_norm, incident_vec)
            if tilts is None:
                tilt = 0

            elif isinstance(tilts, str) and tilts == 'ideal':
                tilt = ideal_tilt
            
            else:
                tilt = tilts[idx]

            mirror_norm = tilt_mirror_normal(init_mirror_norm, offset_vecs[j], tilt)
            # print(f"mirror {idx} tilted by {tilt * 180/np.pi}")
            mirror_norms.append(mirror_norm)
            reflected_vecs.append(calculate_reflection(mirror_norm, incident_vec))
            ideal_tilts.append(ideal_tilt)
            mirror_thetas.append(angle_between_deg(-incident_vec, vector_to_receiver(mirrors[j], receiver_pos)))
    
    return {'heliostat_positions': hstats,
            'receiver_position': receiver_pos,
            'incident_vector': incident_vec,
            'mirror_positions': np.array(mirror_positions),
            'mirror_normals': np.array(mirror_norms),
            'reflected_vectors': np.array(reflected_vecs),
            'ideal_mirror_tilts': np.array(ideal_tilts),
            'heliostat_thetas': np.array(hstat_thetas),
            'mirror_thetas': np.array(mirror_thetas)
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
