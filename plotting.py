from matplotlib import pyplot as plt
import numpy as np

import raytracer as rt

def heliostat_field_figure(model, scale=0.3):
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
        intersection = rt.calculate_target_intersection(reflected_vec, model['mirror_positions'][i], model['receiver_position'])
        intersections.append([intersection[0], intersection[1]])

    intersections = np.array(intersections)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=intersections[:, 0], y=intersections[:, 1])
    ax.grid()

    return fig, ax