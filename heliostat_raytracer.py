import numpy as np

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

def raytrace_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width):
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []

    for hstat in hstats:
        receiver_vec = rt.vector_to_receiver(hstat, receiver_pos)
        mirror_norm = rt.calculate_mirror_normal(receiver_vec, incident_vec)
        mirror_norms.append(mirror_norm)
        mirror_norms.append(mirror_norm)
        reflected_vecs.append(rt.calculate_reflection(mirror_norm, incident_vec))
        reflected_vecs.append(rt.calculate_reflection(mirror_norm, incident_vec))
        mirror_positions.extend(rt.calculate_mirror_positions(hstat, mirror_norm, receiver_vec, reflecting_width))
    
    return {'mirror_normals': np.array(mirror_norms),
            'reflected_vectors': np.array(reflected_vecs),
            'mirror_positions': np.array(mirror_positions)}