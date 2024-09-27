
from Clustering_utils import * 



def apply_region_growing(rg_input_grayscale, rg_input, rg_window_size, rg_seeds, rg_threshold):
    # Initialize visited mask and segmented image
    # 'visited' is initialized to keep track of which pixels have been visited (Mask)
    visited = np.zeros_like(rg_input_grayscale, dtype=bool)
    # 'segmented' will store the segmented image where each pixel belonging
    # to a region will be marked with the corresponding color
    segmented = np.zeros_like(rg_input)

    # Define 3x3 window for mean calculation
    half_window = rg_window_size // 2

    # Loop through seed points
    for seed in rg_seeds:
        seed_x, seed_y = seed

        # Check if seed coordinates are within image bounds
        if (
            0 <= seed_x < rg_input_grayscale.shape[0]
            and 0 <= seed_y < rg_input_grayscale.shape[1]
        ):
            # Process the seed point
            region_mean = rg_input_grayscale[seed_x, seed_y]

        # Initialize region queue with seed point
        # It holds the candidate pixels
        queue = [(seed_x, seed_y)]

        # Region growing loop
        # - Breadth-First Search (BFS) is used here to ensure
        # that all similar pixels are added to the region
        while queue:
            # Pop pixel from queue
            x, y = queue.pop(0)

            # Check if pixel is within image bounds and not visited
            if (
                (0 <= x < rg_input_grayscale.shape[0])
                and (0 <= y < rg_input_grayscale.shape[1])
                and not visited[x, y]
            ):
                # Mark pixel as visited
                visited[x, y] = True

                # Check similarity with region mean
                if (
                    abs(rg_input_grayscale[x, y] - region_mean)
                    <= rg_threshold
                ):
                    # Add pixel to region
                    segmented[x, y] = rg_input[x, y]

                    # Update region mean
                    # Incremental update formula for mean:
                    # new_mean = (old_mean * n + new_value) / (n + 1)
                    number_of_region_pixels = np.sum(
                        segmented != 0
                    )  # Number of pixels in the region
                    region_mean = (
                        region_mean * number_of_region_pixels
                        + rg_input_grayscale[x, y]
                    ) / (number_of_region_pixels + 1)

                    # Add neighbors to queue
                    for i in range(-half_window, half_window + 1):
                        for j in range(-half_window, half_window + 1):
                            if (
                                0 <= x + i < rg_input_grayscale.shape[0]
                                and 0 <= y + j < rg_input_grayscale.shape[1]
                            ):
                                queue.append((x + i, y + j))

    return segmented


def kmeans_segmentation(
        image,
        n_clusters,
        max_iterations,
        spatial_segmentation,
        spatial_segmentation_weight,
        centroid_optimization = False,
        centroids_color=None,
        centroids_spatial=None):

        img = np.array(image, copy=True, dtype=float)

        if spatial_segmentation:
            h, w, _ = img.shape
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            xy_coords = np.column_stack(
                (x_coords.flatten(), y_coords.flatten())
            )  # spatial coordinates in the features space

        img_as_features = img.reshape(-1, img.shape[2])  # without spatial info included

        labels = np.zeros(
            (img_as_features.shape[0], 1)
        )  # (image size x 1) this array contains the labels of each pixel (belongs to which centroid)

        distance = np.zeros(
            (img_as_features.shape[0], n_clusters), dtype=float
        )  # (distance for each colored pixel over the entire clusters)

        # if the centriods have been not provided
        if centroids_color is None:
            centroids_indices = np.random.choice(
                img_as_features.shape[0], n_clusters, replace=False
            )  # initialize the centroids
            centroids_color = img_as_features[centroids_indices]  # in terms of color
            if spatial_segmentation:
                centroids_spatial = xy_coords[
                    centroids_indices
                ]  # this to introduce restriction in the spatial space of the image

            # Form initial clustering
            if centroid_optimization:
                rows = np.arange(img.shape[0])
                columns = np.arange(img.shape[1])

                sample_size = (
                    len(rows) // 16 if len(rows) > len(columns) else len(columns) // 16
                )
                ii = np.random.choice(rows, size=sample_size, replace=False)
                jj = np.random.choice(columns, size=sample_size, replace=False)
                subimage = img[
                    ii[:, np.newaxis], jj[np.newaxis, :], :
                ]  # subimage for redistribute the centriods

                if spatial_segmentation:
                    centroids_color, centroids_spatial, _ = kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                        centroids_spatial=centroids_spatial,
                    )
                else:
                    centroids_color, _ = kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                    )

        for _ in range(max_iterations):
            for centroid_idx in range(centroids_color.shape[0]):
                distance[:, centroid_idx] = np.linalg.norm(
                    img_as_features - centroids_color[centroid_idx], axis=1
                )

                if spatial_segmentation:
                    distance[:, centroid_idx] += (
                        np.linalg.norm(
                            xy_coords - centroids_spatial[centroid_idx], axis=1
                        ) * spatial_segmentation_weight
                    )

            labels = np.argmin(
                distance, axis=1
            )  # assign each point in the feature space a label according to its distance from each centriod based on (spatial and color distance)

            for centroid_idx in range(centroids_color.shape[0]):
                cluster_colors = img_as_features[labels == centroid_idx]
                if len(cluster_colors) > 0:  # Check if cluster is not empty
                    new_centroid_color = np.mean(cluster_colors, axis=0)
                    centroids_color[centroid_idx] = new_centroid_color

                    if spatial_segmentation:
                        cluster_spatial = xy_coords[labels == centroid_idx]
                        new_centroid_spatial = np.mean(cluster_spatial, axis=0)
                        centroids_spatial[centroid_idx] = new_centroid_spatial

        if spatial_segmentation:
            return centroids_color, centroids_spatial, labels
        else:
            return centroids_color, labels




def mean_shift_clusters(
    image, window_size, threshold, sigma, max_iterations=100):
    """
    Perform Mean Shift clustering on an image.

    Args:
        image (numpy.ndarray): The input image.
        window_size (float): The size of the window for the mean shift.
        threshold (float): The convergence threshold.
        sigma (float): The standard deviation for the Gaussian weighting.

    Returns:
        list: A list of dictionaries representing the clusters. Each dictionary contains:
            - 'points': A boolean array indicating the points belonging to the cluster.
            - 'center': The centroid of the cluster.
    """
    image = (
        (image - image.min()) * (1 / (image.max() - image.min())) * 255
    ).astype(np.uint8)
    img = np.array(image, copy=True, dtype=float)

    img_as_features = img.reshape(
        -1, img.shape[2]
    )  # feature space (each channel elongated)

    num_points = len(img_as_features)
    visited = np.full(num_points, False, dtype=bool)
    clusters = []
    iteration_number = 0
    while (
        np.sum(visited) < num_points and iteration_number < max_iterations
    ):  # check if all points have been visited, thus, assigned a cluster.
        initial_mean_idx = np.random.choice(
            np.arange(num_points)[np.logical_not(visited)]
        )
        initial_mean = img_as_features[initial_mean_idx]

        while True:
            distances = np.linalg.norm(
                initial_mean - img_as_features, axis=1
            )  # distances

            weights = gaussian_weight(
                distances, sigma
            )  # weights for computing new mean

            within_window = np.where(distances <= window_size / 2)[0]
            within_window_bool = np.full(num_points, False, dtype=bool)
            within_window_bool[within_window] = True

            within_window_points = img_as_features[within_window]

            new_mean = np.average(
                within_window_points, axis=0, weights=weights[within_window]
            )

            # Check convergence
            if np.linalg.norm(new_mean - initial_mean) < threshold:
                merged = False  # Check merge condition
                for cluster in clusters:
                    if (
                        np.linalg.norm(cluster["center"] - new_mean)
                        < 0.5 * window_size
                    ):
                        # Merge with existing cluster
                        cluster["points"] = (
                            cluster["points"] + within_window_bool
                        )  # bool array that represent the points of each cluster
                        cluster["center"] = 0.5 * (cluster["center"] + new_mean)
                        merged = True
                        break

                if not merged:
                    # No merge, create new cluster
                    clusters.append(
                        {"points": within_window_bool, "center": new_mean}
                    )

                visited[within_window] = True
                break

            initial_mean = new_mean
        iteration_number += 1
        
    return clusters



   
def fit_agglomerative_clusters(points, agglo_initial_num_of_clusters, agglo_number_of_clusters, distance_calculation_method):
    # initially, assign each point to a distinct cluster
    print("Computing initial clusters ...")
    clusters_list = partition_pixel_into_clusters(
        points, initial_k= agglo_initial_num_of_clusters
    )
    print("number of initial clusters:", len(clusters_list))
    print("merging clusters ...")

    while len(clusters_list) > agglo_number_of_clusters:
        # Find the closest (most similar) pair of clusters
        if distance_calculation_method == "distance between centroids":
            cluster1, cluster2 = min(
                [
                    (c1, c2)
                    for i, c1 in enumerate(clusters_list)
                    for c2 in clusters_list[:i]
                ],
                key=lambda c: clusters_distance_between_centroids(c[0], c[1]),
            )
        else:
            cluster1, cluster2 = min(
                [
                    (c1, c2)
                    for i, c1 in enumerate(clusters_list)
                    for c2 in clusters_list[:i]
                ],
                key=lambda c: max_clusters_distance_between_points(c[0], c[1]),
            )

        # Remove the two clusters from the clusters list
        clusters_list = [
            c for c in clusters_list if c != cluster1 and c != cluster2
        ]

        # Merge the two clusters
        merged_cluster = cluster1 + cluster2

        # Add the merged cluster to the clusters list
        clusters_list.append(merged_cluster)

        print("number of clusters:", len(clusters_list))

    print("assigning cluster num to each point ...")
    cluster = {}
    for cluster_number, cluster in enumerate(clusters_list):
        for point in cluster:
            cluster[tuple(point)] = cluster_number

    print("Computing cluster centers ...")
    centers = {}
    for cluster_number, cluster in enumerate(clusters_list):
        centers[cluster_number] = np.average(cluster, axis=0)

    return cluster

