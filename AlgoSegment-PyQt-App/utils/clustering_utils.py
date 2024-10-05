import random

import numpy as np
from skimage.transform import resize


def rgb_to_xyz(rgb):
    """Convert RGB color values to XYZ color values."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X = 0.412453 * R + 0.35758 * G + 0.180423 * B
    Y = 0.212671 * R + 0.71516 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    return np.stack((X, Y, Z), axis=-1)


def xyz_to_luv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    constant = 903.3
    un = 0.19793943
    vn = 0.46832096

    epsilon = 1e-12  # to prevent division by zero
    u_prime = 4 * X / (X + 15 * Y + 3 * Z + epsilon)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z + epsilon)

    L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, constant * Y)
    U = 13 * L * (u_prime - un)
    V = 13 * L * (v_prime - vn)

    return np.stack((L, U, V), axis=-1)


def scale_luv_8_bits(luv_image):
    L, U, V = luv_image[..., 0], luv_image[..., 1], luv_image[..., 2]

    scaled_L = L * (255 / 100)
    scaled_U = (U + 134) * (255 / 354)
    scaled_V = (V + 140) * (255 / 262)

    return np.stack((L, U, V), axis=-1)


def anti_aliasing_resize(img):
    """This function can be used for resizing images of huge size to optimize the segmentation algorithm"""
    ratio = min(1, np.sqrt((512 * 512) / np.prod(img.shape[:2])))
    newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
    img = resize(img, newshape, anti_aliasing=True)
    return img


def gaussian_weight(distance, sigma):
    """
    Introduce guassian weighting based on the distance from the mean
    """
    return np.exp(-(distance**2) / (2 * sigma**2))


def generate_random_color():
    """
    Description:
        -   Generate a random color for the seeds and their corresponding region in the region-growing segmentation.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def map_rgb_luv(image):
    image = anti_aliasing_resize(image)
    normalized_image = (image - image.min()) / (
        image.max() - image.min()
    )  # nomalize before
    xyz_image = rgb_to_xyz(normalized_image)
    luv_image = xyz_to_luv(xyz_image)
    luv_image_normalized = (luv_image - luv_image.min()) / (
        luv_image.max() - luv_image.min()
    )  # normalize after  (point of question !!)
    # scaled_image = scale_luv_8_bits(luv_image)
    return luv_image_normalized


def agglo_reshape_image(image):
    """
    Description:
        -   It creates an array with each row corresponds to a pixel
            and each column corresponds to a color channel (R, G, B)
    """
    pixels = image.reshape((-1, 3))
    return pixels


def get_cluster_number(point, cluster):
    """
    Find cluster number of point
    """
    # assuming point belongs to clusters that were computed by fit functions
    return cluster[tuple(point)]


def get_cluster_center(point, clusters, centers):
    """
    Find center of the cluster that point belongs to
    """
    point_cluster_num = get_cluster_number(point, clusters)
    center = centers[point_cluster_num]
    return center


def downsample_image(agglo_input_image, agglo_scale_factor):
    """
    Description:
        -   Downsample the input image using nearest neighbor interpolation.
    """
    # Get the dimensions of the original image
    height, width, channels = agglo_input_image.shape

    # Calculate new dimensions after downsampling
    new_width = int(width / agglo_scale_factor)
    new_height = int(height / agglo_scale_factor)

    # Create an empty array for the downsampled image
    downsampled_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Iterate through the original image and select pixels based on the scale factor
    for y in range(0, new_height):
        for x in range(0, new_width):
            downsampled_image[y, x] = agglo_input_image[
                y * agglo_scale_factor, x * agglo_scale_factor
            ]

    return downsampled_image


def euclidean_distance(point1, point2):
    """
    Description:
        -   Computes euclidean distance of point1 and point2.
            Noting that "point1" and "point2" are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def max_clusters_distance_between_points(cluster1, cluster2):
    """
    Description:
        -   Computes distance between two clusters.
            cluster1 and cluster2 are lists of lists of points
    """
    return max(
        [
            euclidean_distance(point1, point2)
            for point1 in cluster1
            for point2 in cluster2
        ]
    )


def clusters_distance_between_centroids(cluster1, cluster2):
    """
    Description:
        -   Computes distance between two centroids of the two clusters
            cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


def partition_pixel_into_clusters(points, initial_k=25):
    """
    Description:
        -   It partitions pixels into self.initial_k groups based on color similarity
    """
    # Initialize a dictionary to hold the clusters each represented by:
    # The key: the centroid color.
    # The value: the list of pixels that belong to that cluster.
    initial_clusters = {}
    # Defining the partitioning step
    # 256 is the maximum value for a color channel
    d = int(256 / (initial_k))
    # Iterate over the range of initial clusters and assign the centroid colors for each cluster.
    # The centroid colors are determined by the multiples of the step size (d) ranging from 0 to 255.
    # Each centroid color is represented as an RGB tuple (j, j, j) where j is a multiple of d,
    # ensuring even distribution across the color space.
    for i in range(initial_k):
        j = i * d
        initial_clusters[(j, j, j)] = []
    # It calculates the Euclidean distance between the current pixel p and each centroid color (c)
    # It then assigns the pixel p to the cluster with the closest centroid color.
    # grops.keys() returns the list of centroid colors.
    # The min function with a custom key function (lambda c: euclidean_distance(p, c)) finds the centroid color with the minimum distance to the pixel p,
    # and the pixel p is appended to the corresponding cluster in the groups dictionary.
    for i, p in enumerate(points):
        if i % 100000 == 0:
            print("processing pixel:", i)
        nearest_group_key = min(
            initial_clusters.keys(), key=lambda c: euclidean_distance(p, c)
        )
        initial_clusters[nearest_group_key].append(p)
    # The function then returns a list of pixel groups (clusters) where each group contains
    # the pixels belonging to that cluster.
    # It filters out any empty clusters by checking the length of each cluster list.
    return [g for g in initial_clusters.values() if len(g) > 0]
