import numpy as np
import open3d as o3d

from numpy.linalg import inv   # this is an efficient way to do inversion
from numpy.linalg import cholesky
from scipy.optimize import minimize


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
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + noise**2 * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + 0.5 * len(X_train) * np.log(2*np.pi)

    return fn



###############################################################################################################################################################


def doIt(pcd, train_test_split_ratio):
    pcd_arr_original = np.asarray(pcd.points)

    # crop the pointcloud if there are too many points
    if len(pcd_arr_original) > 1440:
        pcd_arr = pcd_arr_original[:1440]
    else:
        pcd_arr = pcd_arr_original


    # From now on we work with the cropped pointcloud
    test_train_threshold = int(train_test_split_ratio * len(pcd_arr)) # Determine the index for the last training point/start of first test point
    splitarr = pcd_arr[0:test_train_threshold, :]

    splitpcd= o3d.geometry.PointCloud()
    splitpcd.points = o3d.utility.Vector3dVector(splitarr)

    o3d.io.write_point_cloud("Pointclouds/myProcessedTrainingPCL.ply", splitpcd) # store the preprocessed pointcloud

    trainpcd = o3d.io.read_point_cloud("Pointclouds/myProcessedTrainingPCL.ply")
    trainpcd_arr = np.asarray(trainpcd.points)

    noise_3D = 0.1

    X_3D_test = pcd_arr[test_train_threshold:len(pcd_arr), :]

    #X_3D_test = 100*np.random.rand(len(pcd_arr) - test_train_threshold, 3) + 100  # compare with random data

    X_3D_train = trainpcd_arr
    Y_3D_train = np.zeros(len(trainpcd_arr))

    # Without minimizing the log likelihood function:
    # mu_s, cov = prediction(X_3D_test, X_3D_train, Y_3D_train, sigma_y=noise_3D)

    # Find optimal parameters:
    res = minimize(loglikelihood_fn(X_3D_train, Y_3D_train, noise_3D), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')  # Use the L-BFGS-B algorithm for bound constrained minimization.

    mu_s, cov = prediction(X_3D_test, X_3D_train, Y_3D_train, *res.x, sigma_y=noise_3D)

    print("Means after optimizing: ", mu_s)
    print("Covariance Matrix shape: ", cov.shape)

    ### Visualize the test result

    mu = mu_s.ravel()
    uncertainty_normalization_factor = np.max(np.diag(cov))

    print(np.diag(cov).shape)

    print(cov)
    # print(np.diag(cov))

    testpcd = o3d.geometry.PointCloud()
    testpcd.points = o3d.utility.Vector3dVector(X_3D_test)
    testpcd.colors = o3d.utility.Vector3dVector(0.1 * np.ones(shape=(X_3D_test.shape[0], 3)))
    o3d.visualization.draw_geometries([testpcd, trainpcd])



pcd = o3d.io.read_point_cloud("Pointclouds/knot.ply")

#o3d.visualization.draw_geometries([pcd])

doIt(pcd, 0.8)



