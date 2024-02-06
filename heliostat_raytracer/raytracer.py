import numpy as np
from tqdm import tqdm

from heliostat import *
from vector import *

def generate_uniform_incidence(beam_size, raycasts, start_height, incident_vec):
    """
    Generate a square grid of evenly spaced identical initial beam points

    returns np.array(x, y, z, direction_vector)
    """
    if raycasts**0.5 % 1 != 0: 
        raise ValueError("Total raycasts must be n^2 for integer n.")
    xx, yy =  np.meshgrid(
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True),
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True)
    )

    x = xx.ravel()
    y = yy.ravel()

    rays = []
    for i in range(raycasts):
        ray_pos = np.hstack((x[i], y[i], start_height))
        rays.append(np.array((ray_pos, incident_vec), dtype=np.ndarray))

    return rays

def intersect_circle(ray, circ):
    intersect = calculate_plane_intersection(ray[1], ray[0], circ[1], circ[0])
    if intersect is None:
        return None
    
    dist = distance(intersect, circ[0])
    if dist <= circ[2]:
        return intersect
    else:
        return None
    
def intersect_rectangle(ray, rect):
    intersect = calculate_plane_intersection(ray[1], ray[0], rect[1], rect[0])
    if intersect is None:
        return None
    
    R = rotation_matrix_from_vectors(rect[1], np.array((0, 0, 1)))
    point = np.matmul(R, intersect - rect[0])
    xsize = rect[2][0]
    ysize = rect[2][1]

    if (point[0] >= -xsize/2 and point[0] <= xsize/2) and (point[1] >= -ysize/2 and point[1] <= ysize/2):
        return intersect
    else:
        return None

def find_closest_intersection(model, ray):
    geometry = model['geometry']
    intersects = []
    distances = []
    objs = []

    for rect in geometry['rectangles']:
        intersect = intersect_rectangle(ray, rect)
        if intersect is None: continue
        intersects.append(intersect)
        distances.append(distance(ray[0], intersect))
        objs.append(rect)

    for circ in geometry['circles']:
        intersect = intersect_circle(ray, circ)
        if intersect is None: continue
        intersects.append(intersect)
        distances.append(distance(ray[0], intersect))
        objs.append(circ)

    if len(intersects) == 0:
        return None, None

    distances = np.array(distances)
    idx_min = min(range(len(distances)), key=distances.__getitem__) # https://stackoverflow.com/a/11825864
    
    return intersects[idx_min], objs[idx_min]
    
def run_raytracer(model, initial_rays, max_bounces=100):
    recevier_pos = model['receiver_position']
    incident_vec = model['incident_vector']
    mirror_norms = model['mirror_normals']

    rays = [] # Each ray is an array of (point, direction)
    for initial_ray in tqdm(initial_rays):
        ray = [np.array(initial_ray)]
        last_ray = initial_ray

        for n in range(max_bounces):
            intersect, obj = find_closest_intersection(model, last_ray)
            if intersect is None: 
                break
            
            is_rect = not isinstance(obj[2], float)
            if is_rect and len(ray) == 1:
                break

            surf_norm = obj[1]
            r = np.array((intersect, calculate_reflection(surf_norm, last_ray[1])))
            last_ray = r
            ray.append(r)

            if is_rect:
                break

        rays.append(np.array(ray))
   
    return rays

def prune_rays(model):
    rays = []
    for ray in model['rays']:
        if len(ray) != 1:
            rays.append(ray)

    model['rays'] = rays
    return model