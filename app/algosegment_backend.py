import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from AlgoSegment_UI import Ui_MainWindow
from implementation.clustering_algo import *
from implementation.thresholding_algo import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from utils.clustering_utils import *
from utils.helper_functions import *
from utils.thresholding_utils import *


class BackendClass(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ### ==== Region-Growing ==== ###
        self.rg_input = None
        self.rg_input_grayscale = None
        self.rg_output = None
        self.rg_seeds = None
        self.rg_threshold = 20
        self.rg_window_size = 3
        self.ui.window_size_spinbox.valueChanged.connect(self.update_rg_window_size)
        self.ui.region_growing_input_figure.canvas.mpl_connect(
            "button_press_event", self.rg_canvas_clicked
        )
        self.ui.region_growing_threshold_slider.valueChanged.connect(
            self.update_region_growing_threshold
        )

        # Region Growing Buttons
        self.ui.apply_region_growing.clicked.connect(self.apply_region_growing)
        self.ui.apply_region_growing.setEnabled(False)
        self.ui.reset_region_growing.clicked.connect(self.reset_region_growing)
        self.ui.reset_region_growing.setEnabled(False)

        ### ==== Agglomerative Clustering ==== ###
        self.agglo_input_image = None
        self.agglo_output_image = None
        self.agglo_number_of_clusters = 2
        self.downsampling = False
        self.agglo_scale_factor = 4
        self.distance_calculation_method = "distance between centroids"
        self.ui.distance_calculation_method_combobox.currentIndexChanged.connect(
            self.get_agglomerative_parameters
        )
        self.agglo_initial_num_of_clusters = 25
        self.ui.apply_agglomerative.setEnabled(False)
        self.ui.apply_agglomerative.clicked.connect(self.apply_agglomerative_clustering)
        self.ui.downsampling.stateChanged.connect(self.get_agglomerative_parameters)
        self.ui.agglo_scale_factor.valueChanged.connect(
            self.get_agglomerative_parameters
        )
        self.ui.initial_num_of_clusters_spinBox.valueChanged.connect(
            self.get_agglomerative_parameters
        )

        ### ==== K_Means ==== ###
        self.k_means_input = None
        self.k_means_luv_input = None
        self.k_means_output = None
        self.n_clusters = 4
        self.max_iterations = 4
        self.spatial_segmentation = False
        self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)
        self.spatial_segmentation_weight = 1
        self.centroid_optimization = True
        self.k_means_LUV = False

        # K_Means Buttons
        self.ui.apply_k_means.setEnabled(False)
        self.ui.apply_k_means.clicked.connect(self.apply_k_means)
        self.ui.spatial_segmentation.stateChanged.connect(
            self.enable_spatial_segmentation
        )

        ### ==== Mean-Shift ==== ###
        self.mean_shift_input = None
        self.mean_shift_luv_input = None
        self.mean_shift_output = None
        self.mean_shift_window_size = 200
        self.mean_shift_sigma = 20
        self.mean_shift_threshold = 10
        self.mean_shift_luv = False

        # Mean-Shift Buttons
        self.ui.apply_mean_shift.setEnabled(False)
        self.ui.apply_mean_shift.clicked.connect(self.apply_mean_shift)

        ### ==== Thresholding ==== ###
        self.thresholding_grey_input = None
        self.thresholding_output = None
        self.number_of_thresholds = 2
        self.thresholding_type = "Optimal - Binary"
        self.local_or_global = "Global"
        self.otsu_step = 1
        self.separability_measure = 0
        self.global_thresholds = None
        self.ui.thresholding_comboBox.currentIndexChanged.connect(
            self.get_thresholding_parameters
        )

        # Thresholding Buttons and checkbox
        self.ui.apply_thresholding.setEnabled(False)
        self.ui.apply_thresholding.clicked.connect(self.apply_thresholding)
        self.ui.number_of_thresholds_slider.setEnabled(False)
        self.ui.number_of_thresholds_slider.valueChanged.connect(
            self.get_thresholding_parameters
        )
        self.ui.local_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.global_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.otsu_step_spinbox.setEnabled(False)

        ### ==== General ==== ###
        # Connect menu action to load_image
        self.ui.actionImport_Image.triggered.connect(self.load_image)

        # change the app icon and title
        self.change_the_icon()

    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon("assets/app_icon.png"))
        self.setWindowTitle("exVision - AlgoSegment")

    def load_image(self):
        # Open file dialog if file_path is not provided
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "Images",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)",
        )

        if file_path and isinstance(file_path, str):
            # Read the matrix, convert to rgb
            img = cv2.imread(file_path, 1)
            img = convert_BGR_to_RGB(img)

            current_tab = self.ui.tabWidget.currentIndex()

            if current_tab == 0:
                self.rg_input = img
                self.rg_input_grayscale = convert_to_gray(self.rg_input)
                self.rg_output = img
                self.display_image(
                    self.rg_input,
                    self.ui.region_growing_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.display_image(
                    self.rg_output,
                    self.ui.region_growing_output_figure_canvas,
                    "Output Image",
                    False,
                )
                self.ui.apply_region_growing.setEnabled(True)
                self.ui.reset_region_growing.setEnabled(True)
                height = self.rg_input.shape[0]
                width = self.rg_input.shape[1]
                self.ui.initial_num_of_clusters_spinBox.setMaximum(height * width)
            elif current_tab == 1:
                self.agglo_input_image = img
                self.display_image(
                    self.agglo_input_image,
                    self.ui.agglomerative_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_agglomerative.setEnabled(True)
            elif current_tab == 2:
                self.k_means_luv_input = map_rgb_luv(img)
                self.k_means_input = img

                if self.ui.k_means_LUV_conversion.isChecked():
                    self.display_image(
                        self.k_means_luv_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.k_means_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_k_means.setEnabled(True)
            elif current_tab == 3:
                self.mean_shift_luv_input = map_rgb_luv(img)
                self.mean_shift_input = img

                if self.ui.mean_shift_LUV_conversion.isChecked():
                    self.display_image(
                        self.mean_shift_luv_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.mean_shift_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_mean_shift.setEnabled(True)
            elif current_tab >= 4:
                self.thresholding_grey_input = convert_to_gray(img)
                self.ui.number_of_thresholds_slider.setEnabled(True)
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.thresholding_input_figure_canvas,
                    "Input Image",
                    True,
                )
                self.ui.apply_thresholding.setEnabled(True)

    def display_image(
        self, image, canvas, title, grey, hist_or_not=False, axis_disabled="off"
    ):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        if not hist_or_not:
            if not grey:
                ax.imshow(image)
            elif grey:
                ax.imshow(image, cmap="gray")
        else:
            self.ui.histogram_global_thresholds_label.setText(" ")
            if grey:
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )
            else:
                image = convert_to_gray(image)
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )

        ax.axis(axis_disabled)
        ax.set_title(title)
        canvas.figure.subplots_adjust(left=0.1, right=0.90, bottom=0.08, top=0.95)
        canvas.draw()

    ## ============== Region-Growing Methods ============== ##
    def rg_canvas_clicked(self, event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            print(
                f"Clicked pixel at ({x}, {y}) with value {self.rg_input_grayscale[y, x]}"
            )

            # Plot a dot at the clicked location
            ax = self.ui.region_growing_input_figure_canvas.figure.gca()
            ax.scatter(
                x, y, color="red", s=10
            )  # Customize the color and size as needed
            self.ui.region_growing_input_figure_canvas.draw()

            # Store the clicked coordinates as seeds
            if self.rg_seeds is None:
                self.rg_seeds = [(x, y)]
            else:
                self.rg_seeds.append((x, y))

    def update_region_growing_threshold(self):
        self.rg_threshold = self.ui.region_growing_threshold_slider.value()
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")

    def update_rg_window_size(self):
        self.rg_window_size = self.ui.window_size_spinbox.value()

    def apply_region_growing(self):
        segmented = apply_region_growing(
            self.rg_input_grayscale,
            self.rg_input,
            self.rg_window_size,
            self.rg_seeds,
            self.rg_threshold,
        )
        self.display_image(
            segmented,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
            False,
            "off",
        )
        self.plot_rg_output(segmented)

        ## =========== Display the segmented image =========== ##

    def plot_rg_output(self, segmented_image):
        # Find contours of segmented region
        contours, _ = cv2.findContours(
            cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on input image
        output_image = self.rg_input.copy()
        cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

        # Display the output image
        self.display_image(
            output_image,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    def reset_region_growing(self):
        self.rg_seeds = None
        self.rg_threshold = 20
        self.ui.region_growing_threshold_slider.setValue(self.rg_threshold)
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")
        self.rg_output = self.rg_input
        self.display_image(
            self.rg_input,
            self.ui.region_growing_input_figure_canvas,
            "Input Image",
            False,
        )
        self.display_image(
            self.rg_output,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    ## ============== K-Means Methods ============== ##
    def get_new_k_means_parameters(self):
        self.n_clusters = self.ui.n_clusters_spinBox.value()
        self.max_iterations = self.ui.k_means_max_iteratation_spinBox.value()
        self.centroid_optimization = self.ui.centroid_optimization.isChecked()

        self.spatial_segmentation_weight = (
            self.ui.spatial_segmentation_weight_spinbox.value()
        )

        self.spatial_segmentation = self.ui.spatial_segmentation.isChecked()
        self.k_means_LUV = self.ui.k_means_LUV_conversion.isChecked()

    def enable_spatial_segmentation(self):
        if self.ui.spatial_segmentation.isChecked():
            self.spatial_segmentation = True
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(True)
        else:
            self.spatial_segmentation = False
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)

    def apply_k_means(self):
        self.get_new_k_means_parameters()
        if self.spatial_segmentation:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = kmeans_segmentation(
                    self.k_means_luv_input,
                    self.n_clusters,
                    self.max_iterations,
                    self.spatial_segmentation,
                    self.spatial_segmentation_weight,
                    self.centroid_optimization,
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = kmeans_segmentation(
                    self.k_means_input,
                    self.n_clusters,
                    self.max_iterations,
                    self.spatial_segmentation,
                    self.spatial_segmentation_weight,
                    self.centroid_optimization,
                )
        else:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = kmeans_segmentation(
                    self.k_means_luv_input,
                    self.n_clusters,
                    self.max_iterations,
                    self.spatial_segmentation,
                    self.spatial_segmentation_weight,
                    self.centroid_optimization,
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = kmeans_segmentation(
                    self.k_means_input,
                    self.n_clusters,
                    self.max_iterations,
                    self.spatial_segmentation,
                    self.spatial_segmentation_weight,
                    self.centroid_optimization,
                )

        self.k_means_output = centroids_color[labels]

        if self.k_means_LUV:
            self.k_means_output = self.k_means_output.reshape(
                self.k_means_luv_input.shape
            )
        else:
            self.k_means_output = self.k_means_output.reshape(self.k_means_input.shape)

        self.k_means_output = (self.k_means_output - self.k_means_output.min()) / (
            self.k_means_output.max() - self.k_means_output.min()
        )
        self.display_image(
            self.k_means_output,
            self.ui.k_means_output_figure_canvas,
            "K-Means Output",
            False,
        )

    ## ============== Mean-Shift Methods ============== ##
    def get_new_mean_shift_parameters(self):
        self.mean_shift_window_size = self.ui.mean_shift_window_size_spinbox.value()
        self.mean_shift_sigma = self.ui.mean_shift_sigma_spinbox.value()
        self.mean_shift_threshold = self.ui.mean_shift_threshold_spinbox.value()

        self.mean_shift_luv = self.ui.mean_shift_LUV_conversion.isChecked()

    def calculate_mean_shift_clusters(self, image):
        clusters = mean_shift_clusters(
            image,
            self.mean_shift_window_size,
            self.mean_shift_threshold,
            self.mean_shift_sigma,
        )
        output = np.zeros(image.shape)

        for cluster in clusters:
            bool_image = cluster["points"].reshape(image.shape[0], image.shape[1])
            output[bool_image, :] = cluster["center"]

        return output

    def apply_mean_shift(self):
        self.get_new_mean_shift_parameters()

        if self.mean_shift_luv:
            self.display_image(
                self.mean_shift_luv_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_luv_input
            )
        else:
            self.display_image(
                self.mean_shift_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_input
            )

        self.mean_shift_output = (
            self.mean_shift_output - self.mean_shift_output.min()
        ) / (self.mean_shift_output.max() - self.mean_shift_output.min())
        self.display_image(
            self.mean_shift_output,
            self.ui.mean_shift_output_figure_canvas,
            "Mean Shift Output",
            False,
        )

    ## ============== Agglomerative Clustering ============== ##
    def get_agglomerative_parameters(self):
        self.downsampling = self.ui.downsampling.isChecked()
        self.agglo_number_of_clusters = self.ui.agglo_num_of_clusters_spinBox.value()
        self.agglo_scale_factor = self.ui.agglo_scale_factor.value()
        self.agglo_initial_num_of_clusters = (
            self.ui.initial_num_of_clusters_spinBox.value()
        )
        self.ui.initial_num_of_clusters_label.setText(
            "Initial Number of Clusters: " + str(self.agglo_initial_num_of_clusters)
        )
        self.distance_calculation_method = (
            self.ui.distance_calculation_method_combobox.currentText()
        )

    def apply_agglomerative_clustering(self):
        if self.downsampling:
            agglo_downsampled_image = downsample_image(
                self.agglo_input_image, self.agglo_scale_factor
            )
        else:
            agglo_downsampled_image = self.agglo_input_image
        self.get_agglomerative_parameters()
        pixels = agglo_reshape_image(agglo_downsampled_image)
        self.cluster = fit_agglomerative_clusters(
            pixels,
            self.agglo_initial_num_of_clusters,
            self.agglo_number_of_clusters,
            self.distance_calculation_method,
        )

        self.agglo_output_image = [
            [get_cluster_center(pixel, self.cluster) for pixel in row]
            for row in agglo_downsampled_image
        ]
        self.agglo_output_image = np.array(self.agglo_output_image, np.uint8)

        self.display_image(
            self.agglo_output_image,
            self.ui.agglomerative_output_figure_canvas,
            f"Segmented image with k={self.agglo_number_of_clusters}",
            False,
        )

    ## ============== Thresholding Methods ============== ##
    def get_thresholding_parameters(self):
        self.number_of_thresholds = self.ui.number_of_thresholds_slider.value()
        self.thresholding_type = self.ui.thresholding_comboBox.currentText()
        self.otsu_step = self.ui.otsu_step_spinbox.value()
        self.ui.number_of_thresholds.setText(
            "Number of thresholds: " + str(self.number_of_thresholds)
        )
        if self.thresholding_type == "OTSU":
            self.ui.number_of_thresholds_slider.setEnabled(True)
            self.ui.otsu_step_spinbox.setEnabled(True)
        else:
            self.ui.number_of_thresholds_slider.setEnabled(False)
            self.ui.otsu_step_spinbox.setEnabled(False)

    def local_global_thresholding(self, state):
        sender = self.sender()
        if state == 2:  # Checked state
            if sender == self.ui.local_checkbox:
                self.ui.global_checkbox.setChecked(False)
                self.local_or_global = "Local"
            else:
                self.ui.local_checkbox.setChecked(False)
                self.local_or_global = "Global"

    def apply_thresholding(self):
        self.get_thresholding_parameters()
        if self.thresholding_type == "Optimal - Binary":
            if self.local_or_global == "Local":
                self.thresholding_output = local_thresholding(
                    self.thresholding_grey_input, optimal_thresholding
                )
            elif self.local_or_global == "Global":
                self.thresholding_output, self.global_thresholds, _ = (
                    optimal_thresholding(self.thresholding_grey_input)
                )
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.histogram_global_thresholds_figure_canvas,
                    "Histogram",
                    True,
                    True,
                    "on",
                )

        elif self.thresholding_type == "OTSU":
            if self.local_or_global == "Local":
                self.thresholding_output = local_thresholding(
                    grayscale_image=self.thresholding_grey_input,
                    threshold_algorithm=lambda img: multi_otsu(
                        img, self.number_of_thresholds, self.otsu_step
                    ),
                    kernel_size=5,
                )
            elif self.local_or_global == "Global":
                (
                    self.thresholding_output,
                    self.global_thresholds,
                    self.separability_measure,
                ) = multi_otsu(
                    self.thresholding_grey_input,
                    self.number_of_thresholds,
                    self.otsu_step,
                )
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.histogram_global_thresholds_figure_canvas,
                    "Histogram",
                    True,
                    True,
                    "on",
                )
                self.ui.separability_measure.setText(
                    "Separability Measure = {:.3f}".format(self.separability_measure)
                )

        self.display_image(
            self.thresholding_output,
            self.ui.thresholding_output_figure_canvas,
            "Thresholding Output",
            True,
        )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
