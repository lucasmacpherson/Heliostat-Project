from matplotlib import pyplot as plt
import numpy as np

import heliostat as hst
from vector import *
from raytracer import *

def heliostat_field_figure(model, scale=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(model['mirror_positions'][:, 0], model['mirror_positions'][:, 1], model['mirror_positions'][:, 2], 
                model['reflected_vectors'][:, 0]*scale,model['reflected_vectors'][:, 1]*scale, model['reflected_vectors'][:, 2]*scale, 
                color='b', label="Reflected Rays")
    ax.quiver(model['mirror_positions'][:, 0], model['mirror_positions'][:, 1], model['mirror_positions'][:, 2], 
                model['mirror_normals'][:, 0]*scale,model['mirror_normals'][:, 1]*scale, model['mirror_normals'][:, 2]*scale, 
                color='orange', label="Mirror Normals")
    ax.quiver(model['heliostat_positions'][:, 0], model['heliostat_positions'][:, 1], model['heliostat_positions'][:, 2],
                -model['incident_vector'][0]*scale, -model['incident_vector'][1]*scale, -model['incident_vector'][2]*scale, 
                color='red', label="Incident Vector")           
    ax.scatter(model['receiver_position'][0], model['receiver_position'][1], zs=model['receiver_position'][2])
    ax.legend()

    return fig, ax

def target_plane_figure(model):
    intersections = []
    for i, reflected_vec in enumerate(model['reflected_vectors']):
        intersection = hst.calculate_target_intersection(reflected_vec, model['mirror_positions'][i], model['receiver_position'])
        intersections.append([intersection[0], intersection[1]])

    intersections = np.array(intersections)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=intersections[:, 0], y=intersections[:, 1])
    ax.grid()

    return fig, ax

def get_rectangle_surface(rect):
    rect_p0 = rect[0]
    rect_norm = rect[1]
    rect_size = rect[2]
    rect_points = 10

    xrange = np.linspace(-rect_size[0]/2, rect_size[0]/2, rect_points)
    yrange = np.linspace(-rect_size[1]/2, rect_size[1]/2, rect_points)
    p1 = []
    for xi in xrange:
        for yj in yrange:
            p1.append(np.array((xi, yj, 0)))
    p1 = np.array(p1)

    n1 = np.array((0, 0, 1))
    R = rotation_matrix_from_vectors(n1, rect_norm)
    p2 = []
    for point in p1:
        p2.append(np.matmul(R, point) + rect_p0)
    
    return np.array(p2)

# Credit: https://stackoverflow.com/a/72226595
def sunflower(n: int, alpha: float) -> np.ndarray:
    # Number of points respectively on the boundary and inside the cirlce.
    n_exterior = np.round(alpha * np.sqrt(n)).astype(int)
    n_interior = n - n_exterior

    # Ensure there are still some points in the inside...
    if n_interior < 1:
        raise RuntimeError(f"Parameter 'alpha' is too large ({alpha}), all "
                           f"points would end-up on the boundary.")
    # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * n, n)

    # Generate the radii.
    r_interior = np.sqrt(np.linspace(0, 1, n_interior))
    r_exterior = np.ones((n_exterior,))
    r = np.concatenate((r_interior, r_exterior))
    
    # Return Cartesian coordinates from polar ones.
    return r.reshape(n, 1) * np.stack((np.cos(angles), np.sin(angles)), axis=1)

def get_circle_surface(circ):
    circ_p0 = circ[0]
    circ_norm = circ[1]
    circ_r = circ[2]
    circ_points = 250

    xy = circ_r*sunflower(circ_points, 0.3)
    p1 = np.column_stack((xy, np.zeros(circ_points)))

    n1 = np.array((0, 0, 1))
    R = rotation_matrix_from_vectors(n1, circ_norm)
    p2 = []
    for point in p1:
        p2.append(np.matmul(R, point) + circ_p0)
    
    return np.array(p2)

def show_system(model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    model = prune_rays(model)

    for rect in model['geometry']['rectangles']:
        surf = get_rectangle_surface(rect)
        ax.plot_trisurf(surf[:, 0], surf[:, 1], surf[:, 2], linewidth=0, color='red')

    for circ in model['geometry']['circles']:
        surf = get_circle_surface(circ)
        ax.plot_trisurf(surf[:, 0], surf[:, 1], surf[:, 2], linewidth=0, color='blue')

    for ray in model['rays']:
        points = np.array([ray[i][0] for i in range(len(ray))])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='blue')

    return fig, ax

def show_target_plane(model):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    model = prune_rays(model)

    points = []
    for ray in model['rays']:
        points.append(np.array(ray[-1][0]))
    points = np.array(points)

    ax.scatter(points[:, 0], points[:, 1], marker='o', color='blue')
    return fig, ax