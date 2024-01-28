import matplotlib.pyplot as plt

from raytracer import *
from heliostat_raytracer import *

points = sunflower(100, 0.7)
xs, ys = (points[:, 0], points[:, 1])
plt.scatter(xs, ys)

ray_count = 100
incident_vec = norm_vector(np.array([0.1, 0, -1]))
start_pos = np.array((1, 1, 4))
beam_points = generate_initial_beam(ray_count, start_pos, incident_vec, 0.15)
points_plane = np.column_stack((points, np.zeros((ray_count))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xticks(np.arange(-5, 6))
ax.set_yticks(np.arange(-5, 6))
ax.set_zticks(np.arange(-5, 6))

ax.axes.set_xlim3d(left=-5, right=5) 
ax.axes.set_ylim3d(bottom=-5, top=5) 
ax.axes.set_zlim3d(bottom=-5, top=5) 

ax.scatter(points_plane[:, 0], points_plane[:, 1], points_plane[:, 2])
ax.scatter(beam_points[:, 0], beam_points[:, 1], beam_points[:, 2])
ax.quiver(0, 0, 0, incident_vec[0], incident_vec[1], incident_vec[2])

plt.show()