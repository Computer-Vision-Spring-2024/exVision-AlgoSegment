# exVision: AlgoSegment

> [!NOTE]  
> This repository contains the segmentation and thresholding techniques that were originally part of the "exVision-FeatureCraft" repository. For earlier contributions and detailed commit history related to this work, please refer to the <a href="https://github.com/Computer-Vision-Spring-2024/exVision-FeatureCraft">exVision-FeatureCraft</a> repository. The activities and contributions of individual users prior to this separation can be viewed there.

## Overview 
 AlgoSegment is a desktop application designed to offer a variety of classical image segmentation techniques. It provides users with the ability to apply advanced algorithms to divide images into meaningful regions based on pixel intensity, color, and spatial features. The app supports both clustering-based segmentation for colored images and thresholding-based segmentation for grayscale images. The application is implemented using PyQt5 for the desktop interface, providing an intuitive environment for real-time usage. You can refer to [the project documentation]("README-Assets/AlgoSegment_documentation.pdf") to get a rough sense of the parameters associated with each algorithm.

## Segmentation Techniques

### Optimal Thresholding (Binary Thresholding)
- This method involves selecting a threshold value to separate the pixel values in an image into two distinct classes: foreground and background.
- The optimal threshold is typically selected based on maximizing the separation between the two classes.
- Useful for images where the object of interest is distinctly different in intensity from the background.



<p align="center">
  <img src="README-Assets\images\optimal_image.png" alt="input/output" width="450"/>
  <br>
  <img src="README-Assets\images\optimal_hist.png" alt="intensity histogram" width="450"/>
</p>

### Otsu's Thresholding (Binary and Multi Modal)
- Otsu’s method determines an optimal threshold that minimizes the weighted sum of the w**ithin-class variances** (or maximizes the **between-class variance**) for a grayscale image. 
- For binary images, Otsu finds a single threshold that best separates the two classes.
-  In cases where the histogram has multiple peaks (multi-modal), Otsu can extend to find multiple thresholds to segment the image into more than two classes.

<p align="center">
  <img src="README-Assets\images\Otsu_image.png" alt="input/output" width="450"/>
  <br>
  <img src="README-Assets\images\Otsu_hist.png" alt="intensity histogram for multi-modal threshodling" width="450"/>
</p>

### K-Means Clustering
- A popular clustering algorithm that partitions the data into `k` clusters, each represented by its centroid. 
- By iteratively assigning each pixel to the nearest cluster centroid and updates the centroid based on the new assignments until convergence.
- Suitable for segmenting images based on color or intensity, where pixels are grouped into clusters based on similarity. 

- In addition, you have the option to enforce **spatial segmentation** to make sure that there is spatial consistency in the segemented image.

<p align="center">
  <img src="README-Assets\images\k-means_color_space.png" alt="input/output with only color space segmentation" width="450"/>
  <br>
  <img src="README-Assets\images\k_means_spatial_space.png" alt="input/output with spatial constraints" width="450"/>
</p>


### Mean-Shift Clustering
- A non-parametric clustering technique that identifies clusters by finding dense regions in the feature space. 
- By iteratively shifting each data point towards the mean of the points in its neighborhood, effectively locating the peaks of the density function.
- Effective for images with arbitrary-shaped clusters and varying densities. 

<p align="center">
  <img src="README-Assets\images\mean_shift.png" alt="input/output mean shift clustering" width="450"/>
</p> 


### Region Growing
- A segmentation technique that starts with seed points and grows regions by adding neighboring pixels that have similar properties (e.g., color, intensity).
- Begins with one or more seed points and checks neighboring pixels for similarity, merging them into the growing region.
- Useful for segmenting objects based on predefined criteria and allows for more control over the segmentation process.

<p align="center">
  <img src="README-Assets\images\Region_growing.png" alt="input/output mean shift clustering" width="500"/>
</p> 


### Agglomerative Clustering 
- A hierarchical clustering method that starts with each data point as a separate cluster and merges them iteratively based on proximity until a single cluster is formed. 

- Effective for creating a hierarchy of clusters and visualizing relationships. 

<p align="center">
  <img src="README-Assets\images\Agglomerative.png" alt="input/output mean shift clustering" width="500"/>
</p> 

> **Disclaimer**  
> We did not implement the agglomerative clustering algorithm ourselves. Instead, we used the implementation provided in the following repository: [Image-Clustering by Alianoroozi](https://github.com/alianoroozi/Image-Clustering).  
> Our contribution was adding an optional and initial downsampling step before the algorithm begins, to enhance performance in certain use cases.

**For a more in-depth understanding of each algorithm, please refer to the attached notebooks as well as [the project documentation]("README-Assets\AlgoSegment-Documentation.pdf").**

## Getting Started

To be able to use our app, you can simply follow these steps:
1. Install Python3 on your device. You can download it from <a href="https://www.python.org/downloads/">Here</a>.
2. Install the required packages by the following command.
```
pip install -r requirements.txt
```
3. Run the file with the name "AlgoSegment_Backend.py" in the [app folder](app).

## Acknowledgments

Refer to [this organization's README](https://github.com/Computer-Vision-Spring-2024#acknowledgements) for more details about contributors and supervisors. 

## References 

- Gonzalez, R. C., & Woods, R. E. (Year). Chapter 10: [Image Segmentation]. In *Digital Image Processing* (4th ed., pp. 10-46 to 10-77). 
