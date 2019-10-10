import numpy as np
import open3d as o3d
from numpy.linalg import solve
from numpy.linalg import cholesky
from scipy.optimize import minimize
import math

# Flags:
save_image = False  # save the image on the screen to disk
draw_xstar = True
show_normals = True

point_index = 1000  # the point around which we build a neighborhood
K_neighborhood = 200  # how many nearest neighbors
resolution = 20  # how dense is the Xstar 3D grid
test_threshold = -0.15  # how much broader is the Xstar grid from the min and max training values in the focus patch that we choose


# Define the covariance/kernel function
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# Return the information about the posterior distribution
def prediction(X_s, prior_at_xs, X_train, Y_train, prior_y, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = solve(K, np.eye(len(K), dtype=np.float32))

    mu_s = prior_at_xs + K_s.T.dot(K_inv).dot(Y_train-prior_y)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

# Returns the function that needs to be minimized
def loglikelihood_fn(X_train, Y_train, noise):

    def fn(theta):
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise ** 2 * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        K_inv = solve(K, np.eye(len(K), dtype=np.float32))
        return np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(K_inv.dot(Y_train)) + 0.5 * len(
            X_train) * np.log(2 * np.pi)

    return fn

print("Load a pointcloud and paint it gray")
pcd = o3d.read_point_cloud("Pointclouds/left3.pcd")
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# Build the KD Tree
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
print("Paint the focus point red")
pcd.colors[point_index] = [1, 0, 0]
print("Find its nearest neighbors and paint them blue")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[point_index], K_neighborhood)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
pcd_array = np.asarray(pcd.points)[idx[1:], :]

# Compute the normal vectors
normals = o3d.geometry.estimate_normals(pcd)
normals_array = np.asarray(pcd.normals)[idx[1:], :]

points_in = np.zeros(pcd_array.shape)
points_out = np.zeros(pcd_array.shape)
d_pos = 0.2
d_neg = 0.2

for index, point in enumerate(pcd_array):
    points_in[index] = point - d_pos * pcd.normals[index]
    points_out[index] = point + d_neg * pcd.normals[index]

fone = np.ones((1, len(points_in))) * d_pos  # assign y(x) = +1 to the points inside the surface
fminus = -1 * np.ones((1, len(points_out))) * d_neg  # assign y(x) = -1 to the points outside the surface
fzero = np.zeros((1, len(pcd_array)))  # assign y(x)=0 to the points on the surface

# Concatenate the sub-parts to create the training data:
X_train = np.vstack((pcd_array, points_in, points_out))
Y_train = np.vstack((fzero, fone, fminus)).flatten()  # flattening is required to avoid errors

# Create the prior
prior_y_1 = 0 * np.ones_like(fone)
prior_y_2 = 0 * np.ones_like(fminus)
prior_y_3 = 0 * np.ones_like(fzero)

prior_y = np.vstack((prior_y_1, prior_y_2, prior_y_3)).flatten()

# Create the test arguments :


nr_rows = resolution**3
Xstar = 0.01 * np.ones((nr_rows, 3), dtype=np.float32)
prior_at_xs = 0 * np.ones(shape=Xstar.shape[0])

## Create the testing limits:
minx = np.min(X_train[:, 0], axis=0) - test_threshold
maxx = np.max(X_train[:, 0], axis=0) + test_threshold

miny = np.min(X_train[:, 1], axis=0) - test_threshold
maxy = np.max(X_train[:, 1], axis=0) + test_threshold

minz = np.min(X_train[:, 2], axis=0) - test_threshold
maxz = np.max(X_train[:, 2], axis=0) + test_threshold

cube_x = np.linspace(minx, maxx, resolution)

count = 0
for i, x in enumerate(cube_x):
    cube_y = np.linspace(miny, maxy, resolution)
    for j, y in enumerate(cube_y):
        cube_z = np.linspace(minz, maxz, resolution)
        for k, z in enumerate(cube_z):
            Xstar[count] = [x, y, z]
            count = count + 1


# Find optimal parameters:
noise_3D = 0.1
res = minimize(loglikelihood_fn(X_train, Y_train, noise_3D), [1, 1],
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')  # Use the L-BFGS-B algorithm for bound constrained minimization.

mu_s, cov = prediction(Xstar, prior_at_xs, X_train, Y_train, prior_y, *res.x, sigma_y=noise_3D)


print("Minimization results (l and sigma_f) are: ", *res.x)
print("Means:", mu_s)
print("Covariance: ", cov)
print("Means shape: ", mu_s.shape)
print("Covariance shape: ", cov.shape)



# Visualization Code:

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

line_set = o3d.geometry.LineSet()

if show_normals:
    normal_tip = np.zeros(pcd_array.shape)

    for index, point in enumerate(pcd_array):
        normal_tip[index] = np.linalg.norm(point) * normals_array[index]
        #print("Normal Magnitude: ", np.linalg.norm(normals_array[index])) # this proves that the computed normal is actually a unit vector

    points = np.concatenate((pcd_array, normal_tip), axis=0)
    lines = [[x, 200 + x] for x in range(200)]
    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)  # these lines are the normal vectors
    line_set.colors = o3d.utility.Vector3dVector(colors)


# Draw the Xstar points
Xstar_pcd = o3d.PointCloud()
if draw_xstar == True:
    Xstar_pcd.points = o3d.Vector3dVector(Xstar)

# Draw the coordinate system axes
mesh_frame = o3d.geometry.create_mesh_coordinate_frame(
    size=0.2, origin=[0,0,0]
)

# Visualize all means
mean_points = {}
count_mean = 0
count_add = 0
for i in range(resolution**3):
    mean_value = mu_s[count_mean]
    if math.fabs(mean_value) <= 0.001:  # need to also add the covariance condition
        print("Confidence variance of new candidate point being on surface is: ", cov[count_mean, count_mean])
        mean_points[count_add] = Xstar[i]
        count_add = count_add + 1

    count_mean = count_mean + 1


mean_list = 10*np.ones(shape=(count_add, 3))
for key, value in mean_points.items():
    mean_list[key] = value

print("mean_list shape", mean_list.shape)

mean_pcd = o3d.PointCloud()
mean_pcd.points = o3d.Vector3dVector(mean_list)
mean_pcd.paint_uniform_color([1,1,0])

o3d.visualization.draw_geometries([pcd, line_set, Xstar_pcd, mean_pcd, mesh_frame])


# The following loop creates many patches of neighbourhoods:
# for i in range(1, 5):
#     pcd.colors[1000//i] = [1, 0, 0]
#     print("Find its 200 nearest neighbors, paint them")
#     [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1000//i], 200)
#     np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1.0/i]
#
#     vis.update_geometry()
#     vis.poll_events()
#     vis.update_renderer()
#
#     if save_image:
#         vis.capture_screen_image("Images/reconstructed.jpg")
