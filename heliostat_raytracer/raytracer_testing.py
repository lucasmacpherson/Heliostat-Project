import matplotlib.pyplot as plt

from raytracer import *
from heliostat_raytracer import *

points = sunflower(100, 0.7)
xs, ys = (points[:, 0], points[:, 1])
plt.scatter(xs, ys)

points_plane = np.column_stack((points, np.zeros((100))))
incident_vec = norm_vector(np.array([0.5, 0, -1]))
R = rotation_matrix_from_vectors(np.array((0, 0, 1)), incident_vec)

rotated_points = []
for point in points_plane:
    rotated_points.append(np.array((np.matmul(R, point))))
rotated_points = np.array(rotated_points)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_plane[:, 0], points_plane[:, 1], points_plane[:, 2],
           )
ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
           )
ax.quiver(0, 0, 0, incident_vec[0], incident_vec[1], incident_vec[2])

plt.show()