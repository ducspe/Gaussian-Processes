# Gaussian-Processes
Gaussian Process Implicit Surfaces Project to play with the theory
# gpis.py:
Still need to add y(x) = -1 and y(x) = +1 for a proper training, 
as well as calculate the surface normals and use them for the training process

# gpis4.py:
Fixed the above. TODO: Check isosurface.m from matlab and try to translate the code in Python

# isosurface.py:
Added the isosurface with a contour of value zero. The reconstructed bunny is missing a ear though...
Removed open3d and used mayavi instead, which is based on C++'s VTK library and is much closer/similar to Matlab functionality.
TODO: inject prior knowledge in the prediction process.

# prior.py:
Injected the prior distribution when calculating the Gaussian mean. This way we can make use of apriori defined surfaces where the reconstruction has low confidence. TODO: fix the matrix inversion because it is too inneficient.

# fix_inverse.py:
Inverse of the matrix is computed more efficiently. This is very important because in the worst case scenario this operation can have complexity of O(N^3). Hence it wouldn't scale for large surfaces with high point density (/non-sparse surfaces).
