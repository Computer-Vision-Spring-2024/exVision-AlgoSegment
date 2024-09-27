import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, AgloSegment):
        AgloSegment.setObjectName("MainWindow")
        AgloSegment.resize(1101, 732)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(12)
        AgloSegment.setFont(font)
        self.centralwidget = QtWidgets.QWidget(AgloSegment)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.region_growing_input = QtWidgets.QFrame(self.tab_4)
        self.region_growing_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.region_growing_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.region_growing_input.setObjectName("region_growing_input")
        self.horizontalLayout_14.addWidget(self.region_growing_input)
        self.region_growing_output = QtWidgets.QFrame(self.tab_4)
        self.region_growing_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.region_growing_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.region_growing_output.setObjectName("region_growing_output")
        self.horizontalLayout_14.addWidget(self.region_growing_output)
        self.verticalLayout_5.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.region_growing_threshold = QtWidgets.QLabel(self.tab_4)
        self.region_growing_threshold.setObjectName("region_growing_threshold")
        self.horizontalLayout_16.addWidget(self.region_growing_threshold)
        self.region_growing_threshold_slider = QtWidgets.QDoubleSpinBox(self.tab_4)
        # self.region_growing_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.region_growing_threshold_slider.setObjectName(
            "region_growing_threshold_slider"
        )
        self.region_growing_threshold_slider.setValue(20)
        self.region_growing_threshold_slider.setSingleStep(1)
        self.region_growing_threshold_slider.setMinimum(1)
        self.region_growing_threshold_slider.setMaximum(100)
        self.horizontalLayout_16.addWidget(self.region_growing_threshold_slider)

        self.window_size_HLayout = QtWidgets.QHBoxLayout()
        self.window_size_HLayout.setObjectName("window-size-hlayout")
        self.window_size_label = QtWidgets.QLabel(self.tab_4)
        self.window_size_label.setObjectName("window_size_label")
        self.window_size_spinbox = OddSpinBox(self.tab_4)
        self.window_size_spinbox.setObjectName("window_size_spinbox")
        self.window_size_spinbox.setValue(3)
        self.window_size_spinbox.setSingleStep(2)
        self.window_size_spinbox.setMinimum(3)
        self.window_size_spinbox.setMaximum(21)
        self.window_size_HLayout.addWidget(self.window_size_label)
        self.window_size_HLayout.addWidget(self.window_size_spinbox)

        self.horizontalLayout_15.addLayout(self.window_size_HLayout)
        self.horizontalLayout_15.addLayout(self.horizontalLayout_16)
        self.reset_region_growing = QtWidgets.QPushButton(self.tab_4)
        self.reset_region_growing.setObjectName("Reset")
        self.apply_region_growing = QtWidgets.QPushButton(self.tab_4)
        self.apply_region_growing.setObjectName("apply_region_growing")
        self.horizontalLayout_15.addWidget(self.apply_region_growing)
        self.horizontalLayout_15.addWidget(self.reset_region_growing)
        self.verticalLayout_5.addLayout(self.horizontalLayout_15)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_5)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.agglomerative_input = QtWidgets.QFrame(self.tab_5)
        self.agglomerative_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.agglomerative_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.agglomerative_input.setObjectName("frame")
        self.horizontalLayout_18.addWidget(self.agglomerative_input)
        self.agglomerative_output = QtWidgets.QFrame(self.tab_5)
        self.agglomerative_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.agglomerative_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.agglomerative_output.setObjectName("frame_2")
        self.horizontalLayout_18.addWidget(self.agglomerative_output)
        self.verticalLayout_6.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")

        self.horizontalLayout_35 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.agglo_num_of_clusters_label = QtWidgets.QLabel(self.tab_5)
        self.agglo_num_of_clusters_label.setObjectName("agglo_num_of_clusters_label")
        self.horizontalLayout_35.addWidget(self.agglo_num_of_clusters_label)
        self.agglo_num_of_clusters_spinBox = QtWidgets.QSpinBox(self.tab_5)
        self.agglo_num_of_clusters_spinBox.setMinimum(2)
        self.agglo_num_of_clusters_spinBox.setMaximum(10)
        self.agglo_num_of_clusters_spinBox.setObjectName(
            "agglo_num_of_clusters_spinBox"
        )
        self.horizontalLayout_35.addWidget(self.agglo_num_of_clusters_spinBox)

        self.initial_num_of_clusters_HLayout = QtWidgets.QHBoxLayout()
        self.initial_num_of_clusters_HLayout.setObjectName(
            "initial_num_of_clusters_HLayout"
        )
        self.initial_num_of_clusters_label = QtWidgets.QLabel(self.tab_5)
        self.initial_num_of_clusters_label.setObjectName(
            "initial_num_of_clusters_label"
        )
        self.initial_num_of_clusters_HLayout.addWidget(
            self.initial_num_of_clusters_label
        )
        self.initial_num_of_clusters_spinBox = QtWidgets.QSpinBox(self.tab_5)
        self.initial_num_of_clusters_spinBox.setMinimum(3)
        self.initial_num_of_clusters_spinBox.setObjectName(
            "initial_num_of_clusters_spinBox"
        )
        self.initial_num_of_clusters_HLayout.addWidget(
            self.initial_num_of_clusters_spinBox
        )

        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.agglo_elapsed_time = QtWidgets.QLabel(self.tab_5)
        self.agglo_elapsed_time.setObjectName("agglo_elapsed_time")
        self.verticalLayout_12.addLayout(self.horizontalLayout_35)
        self.verticalLayout_12.addLayout(self.initial_num_of_clusters_HLayout)
        self.verticalLayout_12.addWidget(self.agglo_elapsed_time)

        self.horizontalLayout_36.addLayout(self.verticalLayout_12)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("horizontalLayout_19")

        self.sampling_HLayout = QtWidgets.QHBoxLayout()
        self.sampling_HLayout.setObjectName("horizontalLayout_37")
        self.downsampling = QtWidgets.QCheckBox(self.tab_5)
        self.downsampling.setObjectName("downsampling")

        self.agglo_scale_factor = QtWidgets.QSpinBox(self.tab_5)
        self.agglo_scale_factor.setObjectName("sampling_factor")
        self.agglo_scale_factor.setValue(4)
        self.agglo_scale_factor.setSingleStep(1)
        self.agglo_scale_factor.setMinimum(2)
        self.agglo_scale_factor.setMaximum(10)

        self.sampling_HLayout.addWidget(self.downsampling)
        self.sampling_HLayout.addWidget(self.agglo_scale_factor)

        self.distance_calculation_method = QtWidgets.QHBoxLayout()
        self.distance_calculation_method.setObjectName("distance_calculation_method")
        self.distance_calculation_method_label = QtWidgets.QLabel(self.tab_5)
        self.distance_calculation_method_label.setObjectName(
            "distance_calculation_method_label"
        )

        self.distance_calculation_method_combobox = QtWidgets.QComboBox(self.tab_5)
        self.distance_calculation_method_combobox.setObjectName(
            "distance_calculation_method_combobox"
        )
        self.distance_calculation_method_combobox.addItem("distance between centroids")
        self.distance_calculation_method_combobox.addItem("max distance between pixels")

        self.distance_calculation_method.addWidget(
            self.distance_calculation_method_label
        )
        self.distance_calculation_method.addWidget(
            self.distance_calculation_method_combobox
        )

        self.apply_agglomerative = QtWidgets.QPushButton(self.tab_5)
        self.apply_agglomerative.setObjectName("apply_agglomerative")
        self.verticalLayout_13.addLayout(self.sampling_HLayout)
        self.verticalLayout_13.addLayout(self.distance_calculation_method)
        self.verticalLayout_13.addWidget(self.apply_agglomerative)
        self.horizontalLayout_36.addLayout(self.verticalLayout_13)
        self.verticalLayout_6.addLayout(self.horizontalLayout_36)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tab_6)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.k_means_input = QtWidgets.QFrame(self.tab_6)
        self.k_means_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.k_means_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.k_means_input.setObjectName("k_means_input")
        self.horizontalLayout_17.addWidget(self.k_means_input)
        self.k_means_output = QtWidgets.QFrame(self.tab_6)
        self.k_means_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.k_means_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.k_means_output.setObjectName("k_means_output")
        self.horizontalLayout_17.addWidget(self.k_means_output)
        self.verticalLayout_8.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.n_clusters_label = QtWidgets.QLabel(self.tab_6)
        self.n_clusters_label.setObjectName("n_clusters_label")
        self.horizontalLayout_20.addWidget(self.n_clusters_label)
        self.n_clusters_spinBox = QtWidgets.QSpinBox(self.tab_6)
        self.n_clusters_spinBox.setObjectName("n_clusters_spinBox")
        self.n_clusters_spinBox.setValue(4)
        self.n_clusters_spinBox.setSingleStep(1)
        self.n_clusters_spinBox.setMinimum(2)
        self.n_clusters_spinBox.setMaximum(30)
        self.horizontalLayout_20.addWidget(self.n_clusters_spinBox)
        self.horizontalLayout_23.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.k_means_max_iterations_label = QtWidgets.QLabel(self.tab_6)
        self.k_means_max_iterations_label.setObjectName("k_means_max_iterations_label")
        self.horizontalLayout_21.addWidget(self.k_means_max_iterations_label)
        self.k_means_max_iteratation_spinBox = QtWidgets.QSpinBox(self.tab_6)
        self.k_means_max_iteratation_spinBox.setObjectName(
            "k_means_max_iteratation_spinBox"
        )
        self.k_means_max_iteratation_spinBox.setValue(4)
        self.k_means_max_iteratation_spinBox.setSingleStep(1)
        self.k_means_max_iteratation_spinBox.setMinimum(2)
        self.k_means_max_iteratation_spinBox.setMaximum(30)
        self.horizontalLayout_21.addWidget(self.k_means_max_iteratation_spinBox)
        self.horizontalLayout_23.addLayout(self.horizontalLayout_21)
        self.verticalLayout_8.addLayout(self.horizontalLayout_23)
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.spatial_segmentation = QtWidgets.QCheckBox(self.tab_6)
        self.spatial_segmentation.setObjectName("spatial_segmentation")
        self.verticalLayout_7.addWidget(self.spatial_segmentation)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.spatial_segmentation_weight_label = QtWidgets.QLabel(self.tab_6)
        self.spatial_segmentation_weight_label.setObjectName(
            "spatial_segmentation_weight_label"
        )
        self.horizontalLayout_22.addWidget(self.spatial_segmentation_weight_label)
        self.spatial_segmentation_weight_spinbox = QtWidgets.QDoubleSpinBox(self.tab_6)
        self.spatial_segmentation_weight_spinbox.setObjectName(
            "spatial_segmentation_weight_spinbox"
        )
        self.spatial_segmentation_weight_spinbox.setValue(1)
        self.spatial_segmentation_weight_spinbox.setSingleStep(0.1)
        self.spatial_segmentation_weight_spinbox.setMinimum(0)
        self.spatial_segmentation_weight_spinbox.setMaximum(2)
        self.horizontalLayout_22.addWidget(self.spatial_segmentation_weight_spinbox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_22)
        self.horizontalLayout_24.addLayout(self.verticalLayout_7)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.centroid_optimization = QtWidgets.QCheckBox(self.tab_6)
        self.centroid_optimization.setObjectName("Centroid_Optimization")
        self.centroid_optimization.setChecked(True)
        self.k_means_LUV_conversion = QtWidgets.QCheckBox(self.tab_6)
        self.k_means_LUV_conversion.setObjectName("LUV_conversion")
        self.verticalLayout_10.addWidget(self.centroid_optimization)
        self.verticalLayout_10.addWidget(self.k_means_LUV_conversion)
        self.horizontalLayout_24.addLayout(self.verticalLayout_10)
        self.apply_k_means = QtWidgets.QPushButton(self.tab_6)
        self.apply_k_means.setObjectName("apply_k_means")
        self.horizontalLayout_24.addWidget(self.apply_k_means)
        self.verticalLayout_8.addLayout(self.horizontalLayout_24)
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_7)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.mean_shift_input = QtWidgets.QFrame(self.tab_7)
        self.mean_shift_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mean_shift_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mean_shift_input.setObjectName("mean_shift_input")
        self.horizontalLayout_26.addWidget(self.mean_shift_input)
        self.mean_shift_output = QtWidgets.QFrame(self.tab_7)
        self.mean_shift_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mean_shift_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mean_shift_output.setObjectName("mean_shift_output")
        self.horizontalLayout_26.addWidget(self.mean_shift_output)
        self.verticalLayout_9.addLayout(self.horizontalLayout_26)
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.mean_shift_window_size_label = QtWidgets.QLabel(self.tab_7)
        self.mean_shift_window_size_label.setObjectName("mean_shift_window_size_label")
        self.horizontalLayout_28.addWidget(self.mean_shift_window_size_label)
        self.mean_shift_window_size_spinbox = QtWidgets.QSpinBox(self.tab_7)
        self.mean_shift_window_size_spinbox.setObjectName(
            "mean_shift_window_size_spinbox"
        )
        self.mean_shift_window_size_spinbox.setValue(200)
        self.mean_shift_window_size_spinbox.setSingleStep(10)
        self.mean_shift_window_size_spinbox.setMinimum(20)
        self.mean_shift_window_size_spinbox.setMaximum(1000)
        self.horizontalLayout_28.addWidget(self.mean_shift_window_size_spinbox)
        self.horizontalLayout_30.addLayout(self.horizontalLayout_28)
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.mean_shift_sigma_label = QtWidgets.QLabel(self.tab_7)
        self.mean_shift_sigma_label.setObjectName("mean_shift_sigma_label")
        self.horizontalLayout_29.addWidget(self.mean_shift_sigma_label)
        self.mean_shift_sigma_spinbox = QtWidgets.QSpinBox(self.tab_7)
        self.mean_shift_sigma_spinbox.setObjectName("mean_shift_sigma_spinbox")
        self.mean_shift_sigma_spinbox.setValue(20)
        self.mean_shift_sigma_spinbox.setSingleStep(5)
        self.mean_shift_sigma_spinbox.setMinimum(5)
        self.mean_shift_sigma_spinbox.setMaximum(100)
        self.horizontalLayout_29.addWidget(self.mean_shift_sigma_spinbox)
        self.horizontalLayout_30.addLayout(self.horizontalLayout_29)
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.mean_shift_threshold_label = QtWidgets.QLabel(self.tab_7)
        self.mean_shift_threshold_label.setObjectName("mean_shift_threshold_label")
        self.horizontalLayout_27.addWidget(self.mean_shift_threshold_label)
        self.mean_shift_threshold_spinbox = QtWidgets.QSpinBox(self.tab_7)
        self.mean_shift_threshold_spinbox.setObjectName("mean_shift_threshold_spinbox")
        self.mean_shift_threshold_spinbox.setValue(10)
        self.mean_shift_threshold_spinbox.setSingleStep(2)
        self.mean_shift_threshold_spinbox.setMinimum(1)
        self.mean_shift_threshold_spinbox.setMaximum(50)
        self.horizontalLayout_27.addWidget(self.mean_shift_threshold_spinbox)
        self.horizontalLayout_30.addLayout(self.horizontalLayout_27)
        self.verticalLayout_9.addLayout(self.horizontalLayout_30)
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_31.addItem(spacerItem3)
        self.apply_mean_shift = QtWidgets.QPushButton(self.tab_7)
        self.apply_mean_shift.setObjectName("apply_mean_shift")
        self.mean_shift_LUV_conversion = QtWidgets.QCheckBox(self.tab_6)
        self.mean_shift_LUV_conversion.setObjectName("LUV_conversion")
        self.horizontalLayout_31.addWidget(self.mean_shift_LUV_conversion)
        self.horizontalLayout_31.addWidget(self.apply_mean_shift)
        self.verticalLayout_9.addLayout(self.horizontalLayout_31)
        self.tabWidget.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.tab_8)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.thresholding_input = QtWidgets.QFrame(self.tab_8)
        self.thresholding_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.thresholding_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.thresholding_input.setObjectName("thresholding_input")
        self.horizontalLayout_25.addWidget(self.thresholding_input)
        self.thresholding_output = QtWidgets.QFrame(self.tab_8)
        self.thresholding_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.thresholding_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.thresholding_output.setObjectName("thresholding_output")
        self.horizontalLayout_25.addWidget(self.thresholding_output)
        self.verticalLayout_11.addLayout(self.horizontalLayout_25)

        ## Thresholding
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.thresholding_type = QtWidgets.QLabel(self.tab_8)
        self.thresholding_type.setObjectName("thresholding_type")
        self.horizontalLayout_32.addWidget(self.thresholding_type)
        self.thresholding_comboBox = QtWidgets.QComboBox(self.tab_8)
        self.thresholding_comboBox.addItem("Optimal - Binary")
        self.thresholding_comboBox.addItem("OTSU")
        self.thresholding_comboBox.setObjectName("thresholding_comboBox")
        self.horizontalLayout_32.addWidget(self.thresholding_comboBox)
        self.local_checkbox = QtWidgets.QCheckBox(self.tab_8)
        self.local_checkbox.setObjectName("local_checkbox")
        self.local_checkbox.setChecked(False)
        self.horizontalLayout_32.addWidget(self.local_checkbox)
        self.global_checkbox = QtWidgets.QCheckBox(self.tab_8)
        self.global_checkbox.setObjectName("global_checkbox")
        self.global_checkbox.setChecked(True)
        self.horizontalLayout_32.addWidget(self.global_checkbox)
        self.verticalLayout_10.addLayout(self.horizontalLayout_32)
        ### End of type

        self.horizontalLayout_39 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.separability_measure = QtWidgets.QLabel(self.tab_8)
        self.separability_measure.setObjectName("separability_measure")
        self.horizontalLayout_39.addWidget(self.separability_measure)
        self.verticalLayout_10.addLayout(self.horizontalLayout_39)
        ### Second Row

        self.horizontalLayout_38 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.otsu_step_label = QtWidgets.QLabel(self.tab_8)
        self.otsu_step_label.setObjectName("otsu_step_label")
        self.horizontalLayout_38.addWidget(self.otsu_step_label)
        self.otsu_step_spinbox = QtWidgets.QSpinBox(self.tab_8)
        self.otsu_step_spinbox.setObjectName("otsu_step_spinbox")
        self.otsu_step_spinbox.setValue(1)
        self.otsu_step_spinbox.setSingleStep(1)
        self.otsu_step_spinbox.setMinimum(1)
        self.otsu_step_spinbox.setMaximum(200)
        self.horizontalLayout_38.addWidget(self.otsu_step_spinbox)

        self.horizontalLayout_33 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.horizontalLayout_33.addLayout(self.horizontalLayout_38)
        self.number_of_thresholds = QtWidgets.QLabel(self.tab_8)
        self.number_of_thresholds.setObjectName("number_of_modes")
        self.horizontalLayout_33.addWidget(self.number_of_thresholds)
        self.number_of_thresholds_slider = QtWidgets.QSlider(self.tab_8)
        self.number_of_thresholds_slider.setOrientation(QtCore.Qt.Horizontal)
        self.number_of_thresholds_slider.setObjectName("horizontalSlider")
        self.number_of_thresholds_slider.setValue(1)
        self.number_of_thresholds_slider.setSingleStep(1)
        self.number_of_thresholds_slider.setMinimum(1)
        self.number_of_thresholds_slider.setMaximum(5)
        self.horizontalLayout_33.addWidget(self.number_of_thresholds_slider)
        self.verticalLayout_10.addLayout(self.horizontalLayout_33)
        self.verticalLayout_11.addLayout(self.verticalLayout_10)
        self.horizontalLayout_34 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_34.addItem(spacerItem4)
        self.apply_thresholding = QtWidgets.QPushButton(self.tab_8)
        self.apply_thresholding.setObjectName("pushButton_3")
        self.horizontalLayout_34.addWidget(self.apply_thresholding)
        self.verticalLayout_11.addLayout(self.horizontalLayout_34)
        self.tabWidget.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.verticalLayout_global_thresholds = QtWidgets.QVBoxLayout(self.tab_9)
        self.verticalLayout_global_thresholds.setObjectName("horizontalLayout_37")
        self.histogram_global_thresholds_frame = QtWidgets.QFrame(self.tab_9)
        self.histogram_global_thresholds_frame.setFrameShape(
            QtWidgets.QFrame.StyledPanel
        )
        self.histogram_global_thresholds_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.histogram_global_thresholds_frame.setObjectName("frame_3")
        self.histogram_global_thresholds_label = QtWidgets.QLabel(self.tab_9)
        self.histogram_global_thresholds_label.setObjectName("global_thresholds_label")
        self.verticalLayout_global_thresholds.addWidget(
            self.histogram_global_thresholds_frame
        )
        self.verticalLayout_global_thresholds.addWidget(
            self.histogram_global_thresholds_label
        )
        self.tabWidget.addTab(self.tab_9, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        AgloSegment.setCentralWidget(self.centralwidget)

        ## Menu Bar
        self.menubar = QtWidgets.QMenuBar(AgloSegment)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 28))
        self.menubar.setObjectName("menubar")
        AgloSegment.setMenuBar(self.menubar)

        ## Status Bar
        self.statusbar = QtWidgets.QStatusBar(AgloSegment)
        self.statusbar.setObjectName("statusbar")
        AgloSegment.setStatusBar(self.statusbar)

        ### File Menu
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menubar.addAction(self.menuFile.menuAction())
        #### Load Image
        self.actionImport_Image = QtWidgets.QAction(AgloSegment)
        self.actionImport_Image.setShortcut("Ctrl+I")
        self.actionImport_Image.setObjectName("actionLoad_Image")
        self.menuFile.addAction(self.actionImport_Image)
        #### Exit app
        self.actionExit = QtWidgets.QAction(AgloSegment)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionExit.triggered.connect(self.exit_application)
        self.menuFile.addAction(self.actionExit)

        ## Region Growing Input
        self.region_growing_input_vlayout = QtWidgets.QHBoxLayout(
            self.region_growing_input
        )
        self.region_growing_input_vlayout.setObjectName("region_growing_input_hlayout")
        self.region_growing_input_figure = plt.figure()
        self.region_growing_input_figure_canvas = FigureCanvas(
            self.region_growing_input_figure
        )
        self.region_growing_input_vlayout.addWidget(
            self.region_growing_input_figure_canvas
        )
        ## Region Growing Output
        self.region_growing_output_vlayout = QtWidgets.QHBoxLayout(
            self.region_growing_output
        )
        self.region_growing_output_vlayout.setObjectName(
            "region_growing_output_hlayout"
        )
        self.region_growing_output_figure = plt.figure()
        self.region_growing_output_figure_canvas = FigureCanvas(
            self.region_growing_output_figure
        )
        self.region_growing_output_vlayout.addWidget(
            self.region_growing_output_figure_canvas
        )
        ## End of Region Growing canvas

        ## Agglomerative Inpput
        self.agglomerative_input_vlayout = QtWidgets.QHBoxLayout(
            self.agglomerative_input
        )
        self.agglomerative_input_vlayout.setObjectName("agglomerative_input_vlayout")
        self.agglomerative_input_figure = plt.figure()
        self.agglomerative_input_figure_canvas = FigureCanvas(
            self.agglomerative_input_figure
        )
        self.agglomerative_input_vlayout.addWidget(
            self.agglomerative_input_figure_canvas
        )
        ## Agglomerative Output
        self.agglomerative_output_vlayout = QtWidgets.QHBoxLayout(
            self.agglomerative_output
        )
        self.agglomerative_output_vlayout.setObjectName("agglomerative_output_vlayout")
        self.agglomerative_output_figure = plt.figure()
        self.agglomerative_output_figure_canvas = FigureCanvas(
            self.agglomerative_output_figure
        )
        self.agglomerative_output_vlayout.addWidget(
            self.agglomerative_output_figure_canvas
        )
        ## End of Agglomerative Canvas

        ## K-Means input
        self.k_means_input_vlayout = QtWidgets.QHBoxLayout(self.k_means_input)
        self.k_means_input_vlayout.setObjectName("k_means_input_vlayout")
        self.k_means_input_figure = plt.figure()
        self.k_means_input_figure_canvas = FigureCanvas(self.k_means_input_figure)
        self.k_means_input_vlayout.addWidget(self.k_means_input_figure_canvas)
        ## K-Means output
        self.k_means_output_vlayout = QtWidgets.QHBoxLayout(self.k_means_output)
        self.k_means_output_vlayout.setObjectName("k_means_output_vlayout")
        self.k_means_output_figure = plt.figure()
        self.k_means_output_figure_canvas = FigureCanvas(self.k_means_output_figure)
        self.k_means_output_vlayout.addWidget(self.k_means_output_figure_canvas)
        ## End of K-Means

        ## Mean-Shift input
        self.mean_shift_input_vlayout = QtWidgets.QHBoxLayout(self.mean_shift_input)
        self.mean_shift_input_vlayout.setObjectName("mean_shift_input_vlayout")
        self.mean_shift_input_figure = plt.figure()
        self.mean_shift_input_figure_canvas = FigureCanvas(self.mean_shift_input_figure)
        self.mean_shift_input_vlayout.addWidget(self.mean_shift_input_figure_canvas)
        ## Mean-Shift output
        self.mean_shift_output_vlayout = QtWidgets.QHBoxLayout(self.mean_shift_output)
        self.mean_shift_output_vlayout.setObjectName("mean_shift_output_vlayout")
        self.mean_shift_output_figure = plt.figure()
        self.mean_shift_output_figure_canvas = FigureCanvas(
            self.mean_shift_output_figure
        )
        self.mean_shift_output_vlayout.addWidget(self.mean_shift_output_figure_canvas)
        ## End of Mean-Shift

        ## Thresholding input
        self.thresholding_input_vlayout = QtWidgets.QHBoxLayout(self.thresholding_input)
        self.thresholding_input_vlayout.setObjectName("thresholding_input_vlayout")
        self.thresholding_input_figure = plt.figure()
        self.thresholding_input_figure_canvas = FigureCanvas(
            self.thresholding_input_figure
        )
        self.thresholding_input_vlayout.addWidget(self.thresholding_input_figure_canvas)
        ## Thresholding output
        self.thresholding_output_vlayout = QtWidgets.QHBoxLayout(
            self.thresholding_output
        )
        self.thresholding_output_vlayout.setObjectName("thresholding_output_vlayout")
        self.thresholding_output_figure = plt.figure()
        self.thresholding_output_figure_canvas = FigureCanvas(
            self.thresholding_input_figure
        )
        self.thresholding_output_vlayout.addWidget(
            self.thresholding_output_figure_canvas
        )
        ## Histogram Global Thresholds
        self.histogram_global_thresholds_hlayout = QtWidgets.QHBoxLayout(
            self.histogram_global_thresholds_frame
        )
        self.histogram_global_thresholds_hlayout.setObjectName(
            "histogram_global_thresholds_hlayout"
        )
        self.histogram_global_thresholds_figure = plt.figure()
        self.histogram_global_thresholds_figure_canvas = FigureCanvas(
            self.histogram_global_thresholds_figure
        )
        self.histogram_global_thresholds_hlayout.addWidget(
            self.histogram_global_thresholds_figure_canvas
        )
        ## End of Thresholding

        self.retranslateUi(AgloSegment)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(AgloSegment)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.region_growing_threshold.setText(_translate("MainWindow", "Threshold: 20"))
        self.apply_region_growing.setText(
            _translate("MainWindow", "Apply Region Growing")
        )
        self.reset_region_growing.setText(_translate("MainWindow", "Reset"))
        self.window_size_label.setText(_translate("MainWindow", "Window Size"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_4),
            _translate("MainWindow", "Region Growing"),
        )
        self.agglo_num_of_clusters_label.setText(
            _translate("MainWindow", "Number of Clusters (k)")
        )
        self.downsampling.setText(_translate("MainWindow", "Downsample the image"))
        self.apply_agglomerative.setText(
            _translate("MainWindow", "Apply Agglomerative Clustering")
        )
        self.distance_calculation_method_label.setText(
            _translate("MainWindow", "Distance Calculation Method")
        )
        self.agglo_elapsed_time.setText(_translate("MainWindow", "Elapsed Time is"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_5),
            _translate("MainWindow", "Agglomerative Clustering"),
        )
        self.n_clusters_label.setText(_translate("MainWindow", "n clusters"))
        self.initial_num_of_clusters_label.setText(
            _translate("MainWindow", "Initial number of clusters")
        )
        self.k_means_max_iterations_label.setText(
            _translate("MainWindow", "Max Iterations")
        )
        self.spatial_segmentation.setText(
            _translate("MainWindow", "Spatial Segmentation")
        )
        self.spatial_segmentation_weight_label.setText(
            _translate("MainWindow", "Weight")
        )
        self.centroid_optimization.setText(
            _translate("MainWindow", "Centroid Optimization")
        )
        self.k_means_LUV_conversion.setText(_translate("MainWindow", "LUV_conversion"))
        self.apply_k_means.setText(_translate("MainWindow", "Apply K-Means"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "K-Means")
        )
        self.mean_shift_window_size_label.setText(
            _translate("MainWindow", "Window Size")
        )
        self.mean_shift_sigma_label.setText(_translate("MainWindow", "Sigma"))
        self.mean_shift_threshold_label.setText(
            _translate("MainWindow", "Convergence Threshold")
        )
        self.apply_mean_shift.setText(_translate("MainWindow", "Apply Mean Shift"))
        self.mean_shift_LUV_conversion.setText(
            _translate("MainWindow", "LUV_conversion")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Mean Shift")
        )
        self.thresholding_type.setText(_translate("MainWindow", "Thresholding Type"))
        self.local_checkbox.setText(_translate("MainWindow", "Local Thresholding"))
        self.global_checkbox.setText(_translate("MainWindow", "Global Thresholding"))
        self.separability_measure.setText(
            _translate("MainWindow", "Separability Measure = ")
        )
        self.otsu_step_label.setText(_translate("MainWindow", "Step"))
        self.number_of_thresholds.setText(
            _translate("MainWindow", "Number of thresholds: 1")
        )
        self.apply_thresholding.setText(_translate("MainWindow", "Apply Thresholding"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_8), _translate("MainWindow", "Thresholding")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_9),
            _translate("MainWindow", "Histogram Global Thresholds"),
        )
        font_global_thresholds_label = QtGui.QFont()
        font_global_thresholds_label.setPointSize(14)
        self.histogram_global_thresholds_label.setFont(font_global_thresholds_label)
        self.histogram_global_thresholds_label.setText(
            _translate("MainWindow", "Thresholds values are ")
        )
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionImport_Image.setText(_translate("MainWindow", "Import Image"))
        self.actionExit.setText(_translate("MainWindow", "Exit app"))

    def exit_application(self):
        sys.exit()


class OddSpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.editingFinished.connect(self.adjustValue)

    def adjustValue(self):
        if self.value() % 2 == 0:
            self.setValue(self.value() + 1)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
