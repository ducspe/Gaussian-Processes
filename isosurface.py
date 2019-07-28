import numpy as np
from numpy import *
from numpy.linalg import inv   # this is an efficient way to do inversion
from numpy.linalg import cholesky
from scipy.optimize import minimize
from mayavi import mlab


# Define the covariance/kernel function
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# Return the information about the posterior distribution
def prediction(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


# Returns the function that needs to be minimized
def loglikelihood_fn(X_train, Y_train, noise):

    def fn(theta):
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise ** 2 * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + 0.5 * len(
            X_train) * np.log(2 * np.pi)

    return fn


data_frompcd = np.zeros((166, 6), dtype=float) # (X,Y,Z) + (X,Y,Z) of the normal vector at that point
count = 0
with open('Pointclouds/bunny', 'r') as pcdfile:
    for line in pcdfile:
        j = 0
        for word in line.split():
            data_frompcd[count][j] = float(word)
            j = j + 1
        count = count + 1

# print(data_frompcd)

pcd_arr = data_frompcd[:, 0:3]  # the normal vector information is stripped out


# Will be used as step size variables:
d_pos = 0.2
d_neg = 0.2

#######################################################################################
# Follow the normal vector to create training data outside the original surface:
points_out_0 = [data_frompcd[:, 0]+d_neg*data_frompcd[:, 3]]

points_out_1 = [data_frompcd[:, 1]+d_neg*data_frompcd[:, 4]]

points_out_2 = [data_frompcd[:, 2]+d_neg*data_frompcd[:, 5]]

points_out = np.vstack((points_out_0, points_out_1, points_out_2))

points_out = points_out.T

# print(points_out)

#######################################################################################
# Follow the normal vector to create training data inside the original surface:
points_in_0 = [data_frompcd[:, 0]-d_pos*data_frompcd[:, 3]]

points_in_1 = [data_frompcd[:, 1]-d_pos*data_frompcd[:, 4]]

points_in_2 = [data_frompcd[:, 2]-d_pos*data_frompcd[:, 5]]

points_in = np.vstack((points_in_0, points_in_1, points_in_2))

points_in = points_in.T

# print(points_in)

#######################################################################################

fone = np.ones((1, len(points_in))) * d_pos  # assign y(x) = +1 to the points inside the surface
fminus = -1 * np.ones((1, len(points_out))) * d_neg  # assign y(x) = -1 to the points outside the surface
fzero = np.zeros((1, len(pcd_arr)))  # assign y(x)=0 to the points on the surface

# Concatenate the sub-parts to create the training data:
X_train = np.vstack((pcd_arr, points_in, points_out))
Y_train = np.vstack((fzero, fone, fminus)).flatten()  # flattening is required to avoid errors
# print(Y_train)


## Evaluation limits:

minx = np.min(X_train[:, 0], axis=0) - 0.6
maxx = np.max(X_train[:, 0], axis=0) + 0.6

miny = np.min(X_train[:, 1], axis=0) - 0.6
maxy = np.max(X_train[:, 1], axis=0) + 0.6

minz = np.min(X_train[:, 2], axis=0) - 0.6
maxz = np.max(X_train[:, 2], axis=0) + 0.6

# print(minx)
# print(maxx)
# print(miny)
# print(maxy)
# print(minz)
# print(maxz)

resolution = 10  # grid resolution for evaluation (my computer can handle a max of 20 without changing any RAM settings)

nr_rows = resolution**2 + (resolution-1)*resolution**2
Xstar = 0.01 * np.ones((nr_rows, 3), dtype=np.float16)

# print(Xstar.shape)

for j in range(resolution):
    for i in range(resolution):
        d = j*resolution**2
        lower = resolution*i + d + 1
        upper = (resolution*(i+1)) + d
        Xstar[lower:upper, 0] = np.linspace(minx, maxx, resolution - 1)
        Xstar[lower:upper, 1] = (miny + i * ((maxy - miny) / resolution)) * np.ones((1, upper - lower))
        Xstar[lower:upper, 2] = [minz + ((j + 1) * ((maxz - minz) / resolution))]

#print(Xstar)

noise_3D = 0.1

# Find optimal parameters:
res = minimize(loglikelihood_fn(X_train, Y_train, noise_3D), [1, 1],
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')  # Use the L-BFGS-B algorithm for bound constrained minimization.

mu_s, cov = prediction(Xstar, X_train, Y_train, *res.x, sigma_y=noise_3D)

# print("Means Shape for resolution %s:%s " % (resolution, mu_s.shape))
# print("Covariance Shape for resolution %s:%s: " % (resolution, cov.shape))
#
#
# print("Means: ", mu_s)
# print("Covariance: ", cov)


tsize=int((Xstar.shape[0])**(1/3)) + 1

xeva = Xstar.T[0, :].reshape((tsize,tsize,tsize))
yeva = Xstar.T[1, :].reshape((tsize,tsize,tsize))
zeva = Xstar.T[2, :].reshape((tsize,tsize,tsize))

means_reshaped = mu_s.reshape(xeva.shape)

print(means_reshaped)
mlab.clf()
mlab.contour3d(means_reshaped, contours=[-0.001, 0.0, 0.001])
mlab.show()

#print(mgrid[-1:1:35j])
