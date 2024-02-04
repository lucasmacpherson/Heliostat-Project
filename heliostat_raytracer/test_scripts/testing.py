from matplotlib import pyplot as plt
import numpy as np

from heliostat import *
from heliostat_field import *

hstats = create_heliostat_field(2, [2, 2])
receiver_pos = np.array([0, 0, 1])
incident_vec = norm_vector(np.array([-0.8, 0.1, -1]))

tilts = [
    -0.02, -0.1,
    -0.1, -0.1,
    -0.1, -0.1,
    -0.1, -0.1
]

tilts = np.array([-0.02]).repeat(2*len(hstats))

model = align_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width=0.25, tilts=tilts)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(model['mirror_positions'][:, 0], model['mirror_positions'][:, 1], model['mirror_positions'][:, 2], 
            model['reflected_vectors'][:, 0],model['reflected_vectors'][:, 1], model['reflected_vectors'][:, 2], 
            color='b', label="Reflected Rays")
ax.quiver(model['mirror_positions'][:, 0], model['mirror_positions'][:, 1], model['mirror_positions'][:, 2], 
            model['mirror_normals'][:, 0],model['mirror_normals'][:, 1], model['mirror_normals'][:, 2], 
            color='orange', label="Mirror Normals")
ax.quiver(hstats[:, 0], hstats[:, 1], hstats[:, 2],
            -incident_vec[0], -incident_vec[1], -incident_vec[2], 
            color='red', label="Incident Vector")           
ax.scatter(receiver_pos[0], receiver_pos[1], zs=receiver_pos[2])
ax.legend()
plt.show()

intersections = []
for i, reflected_vec in enumerate(model['reflected_vectors']):
    intersection = calculate_target_intersection(reflected_vec, model['mirror_positions'][i], receiver_pos)
    intersections.append([intersection[0], intersection[1]])

intersections = np.array(intersections)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x=intersections[:, 0], y=intersections[:, 1])
plt.grid()
plt.show()