import numpy as np
from tqdm import tqdm

from heliostat import *
from vector import *

def uniform_2D_grid(beam_size, raycasts):
    if raycasts**0.5 % 1 != 0: 
        raise ValueError("Total raycasts must be n^2 for integer n.")
    xx, yy =  np.meshgrid(
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True),
        np.linspace(-beam_size/2, beam_size/2, int(np.sqrt(raycasts)), endpoint=True)
    )

    x = xx.ravel()
    y = yy.ravel()
    return x, y

def generate_uniform_incidence(beam_size, raycasts, start_height, incident_vec):
    """
    Generate a square grid of evenly spaced identical initial beam points
    returns np.array(position_vector, direction_vector)
    """
    x, y = uniform_2D_grid(beam_size, raycasts)

    rays = []
    for i in range(raycasts):
        ray_pos = np.hstack((x[i], y[i], start_height))
        rays.append(np.array((ray_pos, incident_vec), dtype=np.ndarray))

    return rays

def generate_source_incidence(source_dist, central_vec, system_extent, raycasts):
    source_pos = source_dist * -central_vec
    
    # Transform into coordinates with source at (0, 0)
    l1 = system_extent[0] - source_pos
    l2 = system_extent[1] - source_pos

    r1, theta1, phi1 = cartesian_to_spherical(l1[0], l1[1], l1[2])
    r2, theta2, phi2 = cartesian_to_spherical(l2[0], l2[1], l2[2])
    # Choose correct phi value to take interval between l1, l2
    phi2 = phi2 + 2*np.pi

    tt, pp = np.meshgrid(
        np.linspace(theta2, theta1, raycasts[0], endpoint=True), 
        np.linspace(phi2, phi1, raycasts[1], endpoint=True)
    )

    thetas = tt.ravel()
    phis = pp.ravel()

    rays = []
    for i, phi in enumerate(phis):
        l = np.array(spherical_to_cartesian(source_dist, thetas[i], phi))
        ray_pos = l + source_pos # Transform back to normal coordinates
        incident_vec = norm_vector(l)
        rays.append(np.array((ray_pos - 0.9*incident_vec, incident_vec), dtype=np.ndarray))

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
    for initial_ray in initial_rays:
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

def get_rays_at_target(model):
    rays = []
    receiver_pos = model['receiver_position']
    
    for ray in model['rays']:
        if len(ray) == 1:
            continue

        if ray[-1][0][2] != receiver_pos[2]:
            continue

        rays.append(ray)

    return rays
