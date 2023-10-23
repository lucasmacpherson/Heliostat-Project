from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from heliostat_raytracer import *

hstats = create_heliostat_field(20, [5, 5])
incident_vec = norm_vector(np.array([0.5, 0, -1]))

receiver_pos = np.array([0, 0, 10])
receiver_vecs = get_directions_to_receiver(hstats, receiver_pos)
mirror_norms = calculate_mirror_normals(receiver_vecs, incident_vec)
reflected_vecs = calculate_reflected_rays(mirror_norms, incident_vec)

mirror_pos = calculate_mirror_positions(hstats, mirror_norms, receiver_vecs, reflecting_width=0.1)
reflected_vecs_m = reflected_vecs.repeat(2, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(mirror_pos[:, 0], mirror_pos[:, 1], mirror_pos[:, 2], 
            reflected_vecs_m[:, 0],reflected_vecs_m[:, 1], reflected_vecs_m[:, 2], 
            color='b', label="Reflected Rays")
ax.quiver(hstats[:, 0], hstats[:, 1], hstats[:, 2], 
            mirror_norms[:, 0],mirror_norms[:, 1], mirror_norms[:, 2], 
            color='orange', label="Mirror Normals")
ax.quiver(hstats[:, 0], hstats[:, 1], hstats[:, 2], 
            -incident_vec[0], -incident_vec[1], -incident_vec[2], 
            color='red', label="Incident Vector")           
ax.scatter(receiver_pos[0], receiver_pos[1], zs=receiver_pos[2])
ax.legend()
plt.show()

intersections = calculate_target_intersections(reflected_vecs_m, mirror_pos, receiver_pos)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(intersections[:, 0], intersections[:, 1], marker='o')
plt.show()
