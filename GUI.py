import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QGraphicsView,
                             QGraphicsScene, QGraphicsEllipseItem,
                             QVBoxLayout, QWidget,
                             QGraphicsTextItem,
                             QLabel, QGraphicsProxyWidget)
from PyQt5.QtCore import Qt, QPointF
import UCTMA

# Given parameters
N = 4
Fc = 1e9  # Carrier frequency in Hz
Fs = 1e10  # Sampling frequency in Hz
total_sampling_time = 2e-5  # Total sampling time in seconds
Fm = 1e6  # Modulation frequency in Hz
D_lambda = 0.5  # D/lambda
total_sampling_points = 200000  # Total number of sampling points
Tp = 1e-6  # Period of the modulation function
SNR = 10


class AngleSimulation(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Set up the XY plane
        self.setup_plane()

        # Default (theta, phi) pair
        self.theta = 0  # azimuth angle in degrees
        self.phi = -450  # elevation angle in degrees

        # Draw the point for the default (theta, phi) pair
        self.draw_point(self.theta, self.phi)

    def setup_plane(self):
        # Draw the XY plane
        self.scene.addLine(0, -540, 0, 540)  # Y-axis
        self.scene.addLine(-540, 0, 540, 0)  # X-axis

        # Add labels for X-axis and Y-axis
        y_axis_label = QGraphicsTextItem("Azimuth (θ)")
        y_axis_label.setPos(550, -30)
        self.scene.addItem(y_axis_label)

        x_axis_label = QGraphicsTextItem("Elevation (φ)")
        x_axis_label.setPos(-120, 350)
        self.scene.addItem(x_axis_label)

        # Add labels for angles on the X-axis
        for angle in range(-90, 91, 10):
            angle_label = QGraphicsTextItem(str(angle))
            angle_label.setPos(6 * (angle - 2), 8)
            self.scene.addItem(angle_label)

        # Add labels for angles on the Y-axis
        for angle in range(-180, 181, 20):
            angle_label = QGraphicsTextItem(str(angle))
            angle_label.setPos(10, -3 * (angle + 4))
            self.scene.addItem(angle_label)

        static_point = QGraphicsEllipseItem(540 - 5, -540 + 5, 10, 10)  # Adjust position of the point
        static_point.setBrush(Qt.red)
        self.scene.addItem(static_point)

        # Add label for actual incident angle
        incident_angle_label = QGraphicsTextItem("Actual incident angle(θ, φ)")
        incident_angle_label.setPos(545, -540)  # Adjust position of the label
        self.scene.addItem(incident_angle_label)

        static_point = QGraphicsEllipseItem(540 - 5, -542 + 30, 10, 10)  # Adjust position of the point
        static_point.setBrush(Qt.green)
        incident_angle_label = QGraphicsTextItem("Estimated incident angle(θ, φ)")
        incident_angle_label.setPos(545, -542 + 25)  # Adjust position of the label
        self.scene.addItem(incident_angle_label)
        self.scene.addItem(static_point)

        constraints = (QGraphicsTextItem
                       ("Constraints:\nNumber of Antenna "
                        "Elements :{0}\nCarrier Frequency (Fc): "
                        "{1}GHz\nSampling Frequency(Fs) = "
                        "{2}GHz\nTotal Sampling Time: {3}μs\nTotal Sampling Points: "
                        "{4}\nModulation Frequency(Fp): {5}MHz\nD/λ: "
                        "{6}\nSignal-to-Noise Ratio(SNR): {7}dB"
                        .format(N, int(Fc / 1e9), int(Fs / 1e9),
                                int(total_sampling_time / 1e-6),
                                total_sampling_points,
                                int(int(1 / Tp) / 1e6), D_lambda, SNR)))
        constraints.setPos(-540, -540)
        self.scene.addItem(constraints)

    def draw_point(self, theta, phi):
        # Clear previous point
        self.scene.clear()

        # Redraw the XY plane
        self.setup_plane()

        # Convert (theta, phi) to Cartesian coordinates
        y = theta
        x = -phi

        # Draw the point
        point = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        point.setBrush(Qt.red)
        self.scene.addItem(point)

        # Update angle display
        if self.parentWidget() is not None:
            cur_theta = self.theta
            cur_phi = self.phi
            actual_theta, actual_phi = self.angles2actual(cur_theta, cur_phi)
            actual_new_theta1, actual_new_phi1 = (UCTMA.
                                                DoAEstimator(actual_theta,
                                                             actual_phi,
                                                             Fc, Fs,
                                                             total_sampling_time,
                                                             total_sampling_points,
                                                             N, D_lambda, Tp, SNR, 1))
            eps_error1 = np.sqrt((actual_theta - actual_new_theta1) ** 2 +
                                (actual_phi - actual_new_phi1) ** 2)
            new_theta, new_phi = self.actual2non(actual_new_theta1, actual_new_phi1)
            self.draw_new_point1(new_theta, new_phi)
            (self.parentWidget()
             .angle_label.setText
             (f"Chosen Angle: (θ={actual_theta:.2f}, "f"φ={actual_phi:.2f}), "
              f"Estimated Angle:(θ={actual_new_theta1:.2f}, "
              f"φ={actual_new_phi1:.2f}), " 
              f"Square Error(°):  ε = {eps_error1:.2f}°"))

            actual_new_theta2, actual_new_phi2 = (UCTMA.
                                                  DoAEstimator(actual_theta,
                                                               actual_phi,
                                                               Fc, Fs,
                                                               total_sampling_time,
                                                               total_sampling_points,
                                                               N, D_lambda, Tp, SNR, -1))
            eps_error2 = np.sqrt((actual_theta - actual_new_theta2) ** 2 +
                                (actual_phi - actual_new_phi2) ** 2)
            new_theta, new_phi = self.actual2non(actual_new_theta2, actual_new_phi2)
            self.draw_new_point2(new_theta, new_phi)
            (self.parentWidget()
             .angle_label.setText
             (f"Chosen Angle: (θ={actual_theta:.2f}, "f"φ={actual_phi:.2f}), "
              f"Estimated Angle:(θ={actual_new_theta1:.2f}, "
              f"φ={actual_new_phi1:.2f}), "
              f"Square Error(°):  ε = {eps_error2:.2f}°"))

    def angles2actual(self, theta, phi):
        actual_theta = -(phi / 540) * 90
        actual_phi = -(theta / 540) * 180
        return actual_theta, actual_phi

    def actual2non(self, actual_theta, actual_phi):
        phi = -6 * actual_theta
        theta = -3 * actual_phi
        return theta, phi

    def draw_new_point1(self, theta, phi):
        self.setup_plane()

        # Convert (theta, phi) to Cartesian coordinates
        y = theta
        x = -phi

        # Draw the point
        point = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        point.setBrush(Qt.green)
        self.scene.addItem(point)

    def draw_new_point2(self, theta, phi):
        self.setup_plane()

        # Convert (theta, phi) to Cartesian coordinates
        y = theta
        x = -phi

        # Draw the point
        point = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        point.setBrush(Qt.yellow)
        self.scene.addItem(point)

    def mousePressEvent(self, event):
        # Get mouse position
        mouse_position = self.mapToScene(event.pos())

        # Convert mouse position to (theta, phi) pair
        self.theta = mouse_position.y()
        self.phi = -mouse_position.x()

        # Draw the new point
        self.draw_point(self.theta, self.phi)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.angle_simulation = AngleSimulation()
        layout.addWidget(self.angle_simulation)

        # Add label for displaying chosen angle
        self.angle_label = QLabel("Chosen Angle: (θ=, φ=0)")
        layout.addWidget(self.angle_label)

        self.setLayout(layout)
        self.setWindowTitle("Angle Visualization")
        self.setGeometry(100, 100, 800, 600)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
