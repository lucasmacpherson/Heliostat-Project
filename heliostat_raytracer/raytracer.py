import numpy as np

from vector import *

def intersect_circle(ray, circ):
    intersect = calculate_plane_intersection(ray[1], ray[0], circ[1], circ[0])
    dist = distance(intersect, circ[0])
    if dist <= circ[2]:
        return intersect
    else:
        return None
    
def intersect_rectangle(ray, rect):
    intersect = calculate_plane_intersection(ray[1], ray[0], rect[1], rect[0])
    R = rotation_matrix_from_vectors(rect[1], np.array((0, 0, 1)))
    point = np.matmul(R, intersect - rect[0])

def raytrace(model, beam_points, geometry):
    recevier_pos = model['receiver_position']
    incident_vec = model['incident_vector']
    mirror_norms = model['mirror_normals']

    rays = [] # Each ray is an array of (point, direction)
    for point in beam_points:
        continue
        
    return np.array(rays)