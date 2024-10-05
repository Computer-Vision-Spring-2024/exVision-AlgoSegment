import numpy as np


def pad_image(kernel_size, grayscale_image):
    """
    Description:
        - Pads the grayscale image with zeros.

    Returns:
        - [numpy.ndarray]: A padded grayscale image.
    """
    pad_width = kernel_size // 2
    return np.pad(
        grayscale_image,
        ((pad_width, pad_width), (pad_width, pad_width)),
        mode="edge",
    )


def Normalized_histogram_computation(Image):
    """
    Compute the normalized histogram of a grayscale image.

    Parameters:
    - Image: numpy.ndarray.

    Returns:
    - Histogram: numpy array
        A 1D array representing the normalized histogram of the input image.
        It has 256 element, each element corresponds to the probability of certain pixel intensity (0 to 255).
    """
    # Get the dimensions of the image
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    # Initialize the histogram array with zeros. The array has 256 element, each corresponding to a pixel intensity value (0 to 255)
    Histogram = np.zeros([256])

    # Compute the histogram for each pixel in each channel
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            # Increment the count of pixels in the histogram for the same pixel intensity at position (x, y) in the image.
            # This operation updates the histogram to track the number of pixels with a specific intensity value.
            Histogram[Image[x, y]] += 1
    # Normalize the histogram by dividing each bin count by the total number of pixels in the image
    Histogram /= Image_Height * Image_Width

    return Histogram


def generate_combinations(k, step, start=1, end=255):
    """
    Generate proper combinations of thresholds for histogram bins based on the number of thresholds

    Parameters:
    - k: the number of thresholds
    - step: the increment distance for the threshold, not 1 by default for optimization purposes
    - start: the first number in histogram bins which is 1 by default.
    - end: last number in histogram bins which is 255 by default.

    Returns:
    a list of the proper combinations of thresholds
    """
    combinations = []  # List to store the combinations

    def helper(start, end, k, prefix):
        if k == 0:
            combinations.append(prefix)  # Add the combination to the list
            return
        for i in range(start, end - k + 2, step):
            helper(i + 1, end, k - 1, prefix + [i])

    helper(start, end, k, [])

    return combinations  # Return the list of combinations
