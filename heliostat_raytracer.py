import numpy as np
from tqdm import tqdm

import raytracer as rt

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

def raytrace_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width, tilts=None):
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []

    for i, hstat in enumerate(tqdm(hstats)):
        receiver_vec = rt.vector_to_receiver(hstat, receiver_pos)
        init_mirror_norm = rt.calculate_mirror_normal(receiver_vec, incident_vec)
        mirrors, offset_vecs = rt.calculate_mirror_positions(hstat, init_mirror_norm, receiver_vec, reflecting_width)

        for j in range(2):
            idx = (2*i + j)
            mirror_positions.append(mirrors[j])
            if tilts is None:
                tilt = 0

            elif isinstance(tilts, str) and tilts == 'ideal':
                tilt = rt.calculate_ideal_tilt(mirrors[j], receiver_pos, init_mirror_norm, incident_vec)
                print(f"mirror {idx} tilted by {tilt * 180/np.pi}")
            
            else:
                tilt = tilts[idx]

            mirror_norm = rt.tilt_mirror_normal(init_mirror_norm, offset_vecs[j], tilt)
            mirror_norms.append(mirror_norm)
            reflected_vecs.append(rt.calculate_reflection(mirror_norm, incident_vec))
    
    return {'heliostat_positions': hstats,
            'receiver_position': receiver_pos,
            'incident_vector': incident_vec,
            'mirror_normals': np.array(mirror_norms),
            'reflected_vectors': np.array(reflected_vecs),
            'mirror_positions': np.array(mirror_positions)
            }