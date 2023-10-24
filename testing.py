from matplotlib import pyplot as plt
import numpy as np

from raytracer import *
from heliostat_raytracer import *

hstats = create_heliostat_field(2, [2, 2])
receiver_pos = np.array([0, 0, 1])
incident_vec = norm_vector(np.array([0.5, 0, -1]))

model = raytrace_heliostat_field(hstats, incident_vec, receiver_pos, reflecting_width=0.1)

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
