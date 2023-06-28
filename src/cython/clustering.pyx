import numpy as np
cimport numpy as np
from hnswlib import Index
from libcpp.map cimport map
from libcpp cimport bool
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

from numpy.linalg import norm

cimport cython

np.import_array()

ctypedef np.uint64_t DTYPE_t


############# META POINTS #############

cpdef np.float32_t distance(np.ndarray[np.float32_t, ndim=1] a, np.ndarray[np.float32_t, ndim=1] b, str metric = 'cosine'):
    if metric == 'euclidean':
        return norm(a - b)
    elif metric == 'cosine':
        return 1-np.dot(a, b)/(norm(a)*norm(b))
    else:
        raise ValueError('Metric not supported')

# Class to represent a meta point
cdef class MetaPoint():
    cdef np.float32_t[:] center
    cdef np.float32_t radius

    def __init__(self, np.ndarray[np.float32_t, ndim=1] center,  np.float32_t radius, DTYPE_t point_id):
        self.center = center
        self.radius = radius

    def get_center(self):
        return np.asarray(self.center)

    def get_radius(self):
        return self.radius

    def set_center(self, np.ndarray[dtype=np.float32_t, ndim=1] center):
        self.center = center


# Given a point and a list of meta points, return the meta id point that the point is in
cpdef DTYPE_t get_meta_point_id(
    np.ndarray[dtype=np.float32_t, ndim=1] point,
    np.ndarray[object, ndim=1] meta_points,
    np.ndarray[np.float32_t, ndim=2] centers,
    str metric='cosine'
):
    if len(centers) > 0:
        closest = np.argmin(cdist(point.reshape(1, -1), centers, metric=metric))
        if distance(point, centers[closest], metric) <= meta_points[closest].get_radius():
            return closest

    return len(meta_points)


# Insert a point into a meta point if it is in the radius of the meta point
# Otherwise create a new meta point
def insert_point(
    np.ndarray[dtype=np.float32_t, ndim=1] point,
    np.ndarray[object, ndim=1] meta_points,
    DTYPE_t point_id,
    np.float32_t radius,
    np.ndarray[np.float32_t, ndim=2] centers,
    str metric='cosine'
):
    cdef DTYPE_t meta_point_id = get_meta_point_id(point, meta_points, centers, metric)
    if meta_point_id == len(meta_points):
        # Create a new meta point
        meta_points = np.append(meta_points, MetaPoint(point, radius, point_id))
        centers = np.append(centers, point.reshape(1, -1), axis=0)
    else:
        # Add point to meta point
        # meta_points[meta_point_id].add_point(point_id)
        pass

    return meta_points, centers, meta_point_id


# NOTE: This method is not used and could be integrated in `create_meta_points`.
def update_meta_points(
    np.ndarray[object, ndim=1] meta_points,
    np.ndarray[dtype=np.float32_t, ndim=2] points,
    np.float32_t radius,
    str metric='cosine'
):
    cdef DTYPE_t i
    cdef np.ndarray[np.float32_t, ndim=2] centers = np.array([mp.get_center() for mp in meta_points])
    cdef np.ndarray[DTYPE_t, ndim=1] point_representatives = np.ndarray(len(points), dtype=np.uint64)
    for i in range(len(points)):
        meta_points, centers, meta_point_id = insert_point(points[i], meta_points, np.uint64(i), radius, centers, metric)
        point_representatives[i] = meta_point_id
    return meta_points, point_representatives


# Method to create a numpy array of meta points from a numpy array of points
def create_meta_points(np.ndarray[dtype=np.float32_t, ndim=2] points, np.float32_t radius, str metric='cosine'):
    # Normalize the points
    if metric == 'cosine':
        points = normalize(points, axis=1)
    
    cdef np.ndarray[object, ndim=1] meta_points = np.ndarray(0, dtype=object)
    cdef np.ndarray[np.float32_t, ndim=2] centers = np.ndarray((0, len(points[0])), dtype=np.float32)
    cdef np.ndarray[DTYPE_t, ndim=1] point_representatives = np.ndarray(len(points), dtype=np.uint64)

    for i in range(len(points)):
        meta_points, centers, meta_point_id = insert_point(points[i], meta_points, np.uint64(i), radius, centers, metric)
        point_representatives[i] = meta_point_id
    return meta_points, point_representatives


# Function to collapse meta points
def collapse_meta_points(np.ndarray[object, ndim=1] meta_points, int k, str metric='cosine'):
    cdef np.ndarray[np.float32_t, ndim=2] centers = np.array([mp.get_center() for mp in meta_points])

    space = 'cosine' if metric == 'cosine' else 'l2'
    index = Index(space=space, dim=centers.shape[1])
    index.init_index(max_elements=centers.shape[0], ef_construction=200, M=16)
    index.add_items(centers)

    # Collapse meta points
    prev_centers = np.copy(centers)
    for z in range(100):
        neighbours_idx, distances = index.knn_query(centers, k=k)
        centers = np.mean(centers[neighbours_idx[:, (z==0):]], axis=1, dtype=np.float32)
        # Normalize centers
        if metric == 'cosine':
            centers /= norm(centers, axis=1, keepdims=True)

        # Stop if centers don't change
        if np.array_equal(centers, prev_centers):
            break
        prev_centers = np.copy(centers)


    # Update meta points centers
    for i in range(len(meta_points)):
        meta_points[i].set_center(centers[i])

    return meta_points


# Function to label points that are inside the radius
cpdef np.ndarray[DTYPE_t, ndim=1] get_labels_meta_points(
    np.ndarray[object, ndim=1] meta_points,
    np.float32_t th,
    str metric='cosine'
):
    cdef np.ndarray[np.float32_t, ndim=2] centers = np.array([mp.get_center() for mp in meta_points])
    cdef np.ndarray[DTYPE_t, ndim=1] labels = np.zeros(meta_points.shape[0], dtype=np.uint64)

    visited = set()
    label_count = 0
    for i in range(len(meta_points)):
        if i not in visited:
            visited.add(i)
            labels[i] = label_count
            for j in range(i+1, len(meta_points)):
                if j not in visited and distance(centers[i], centers[j], metric) <= th:
                    visited.add(j)
                    labels[j] = label_count

            label_count += 1

    return labels