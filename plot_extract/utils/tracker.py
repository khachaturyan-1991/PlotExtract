import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


class Tracker:

    """
    kinetics: i:[(None, None), (None, None), (None, None)], ...
    trace: i:[(None, None), ...], ...
    expectations: i:(None, None), ...
    """
    def __init__(self):
        self.num_of_plots = 0
        self.kinetics = {}
        self.trace = {}
        self.expectations = {}


class CCD:

    def __init__(self, tracker_dict, iniertia: float = 0.8, velocity: float = 10, accelaration: float = 5):

        self.plots = tracker_dict
        self.inertia = iniertia
        self.n_missdetections = 0
        self.v = velocity
        self.a = accelaration
        self.cluster_counter = 0

    def get_cetners(self, img, n_of_cluster):
        """
        use KMean to find clusters centers
        """
        y, x = np.where(img)
        kmeans = KMeans(n_clusters=n_of_cluster)
        kmeans.fit(np.column_stack((x, y)))
        return kmeans.cluster_centers_

    def estimate_cluster_num(self, img, eps=2, min_samples=4):
        """
        use DBSCAN to estimate number of clsuters on an image
        """
        y, x = np.where(img)
        coordinates = np.column_stack((x, y))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = db.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)
        return len(unique_labels)

    def show_pice(self, img, n_clusters):
        """
        shows a given img (piece of img)
        indicated number of detected cluster
        """
        plt.imshow(img)
        plt.title(n_clusters)
        plt.show()

    def n_clusters_controle(self, n_clusters):
        """
        controles number of cluster reduction
        to prevent missdetectins on intersections
        Less amount of clusters should be detected
            several times in a raw to be confirmed
        """
        if n_clusters < self.plots.num_of_plots:
            self.cluster_counter += 1
            if self.cluster_counter == 2:
                self.cluster_counter = 0
                self.plots.num_of_plots -= 1
                return n_clusters
            else:
                return self.plots.num_of_plots
        else:
            self.cluster_counter = 0
            return n_clusters

    def assign(self, detections):
        """
        updates tracking objects with new positions;
        i. assign detectins to existin plots
            the rest is return for new plots update_positions()
        ii. new detections are added as new plots add_new_plots()
        iii. if new_positions are found
            then plots kinetics is updated update_kinetics()
        iv. expectations are updated with update_expectations()
        v. after all new detectins are added to trace
        """
        expectations = self.plots.expectations
        new_positions, new_detections = self.update_positions(expectations, detections)
        # a. add new plots
        self.add_new_plots(new_detections)
        # b. update kinetics
        if new_positions:
            self.update_kinetics(new_positions)
        # c. update expectations
        self.update_expectations()
        # d. update trace dict
        self.update_trace()

    def add_new_plots(self, new_detections):
        """
        adds new plot for new detections
        """
        for detection in new_detections:
            self.plots.kinetics[self.plots.num_of_plots] = [detection, (0, 0), (0, 0)]
            self.plots.trace[self.plots.num_of_plots] = []
            self.plots.num_of_plots += 1

    def update_kinetics(self, new_positions):
        """
        updating kinetics based on new detections and previouse positions;
        accounts inertia
        """
        for key in new_positions.keys():
            # new velocity
            dx_new = new_positions[key] - self.plots.kinetics[key][0]
            dx_last = np.array(self.plots.kinetics[key][1])
            # new acceleration
            dx2_new = dx_new - dx_last
            dx2_last = np.array(self.plots.kinetics[key][2])
            # updated kinetics
            self.plots.kinetics[key][2] = self.inertia * dx2_last + (1 - self.inertia) * dx2_new
            self.plots.kinetics[key][1] = self.inertia * dx_last + (1 - self.inertia) * dx_new
            self.plots.kinetics[key][0] = new_positions[key]

    def update_expectations(self):
        """
        updates expectations based on kinetics
        accounting velocity and acceleration coefficients
        """
        for key in self.plots.kinetics:
            x = np.array(self.plots.kinetics[key][0])
            dx = np.array(self.plots.kinetics[key][1])
            dx2 = np.array(self.plots.kinetics[key][2])
            self.plots.expectations[key] = x + self.v * dx + self.a * dx2

    def update_trace(self):
        for key in self.plots.kinetics.keys():
            self.plots.trace[key].append(self.plots.kinetics[key][0])

    def update_positions(self, positions, detections, max_distance_threshold=None):
        """
        Related new detections to existing plots
        and returns remaining detectins to asign to new plots
        """
        object_positions = np.array(list(positions.values()))
        detection_positions = np.array(detections)
        num_objects = len(object_positions)
        num_detections = len(detection_positions)
        # Initialize and fill the cost matrix
        cost_matrix = np.zeros((num_objects, num_detections))
        for i, obj_pos in enumerate(object_positions):
            for j, det_pos in enumerate(detection_positions):
                cost_matrix[i, j] = np.linalg.norm(np.array(obj_pos) - np.array(det_pos))
        # Perform the assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment = {}
        assigned_detections = set()
        for i, j in zip(row_ind, col_ind):
            if max_distance_threshold is None or cost_matrix[i, j] <= max_distance_threshold:
                assignment[list(positions.keys())[i]] = detections[j]
                assigned_detections.add(j)
        # Determine remaining detections
        remaining_detections = [detections[j] for j in range(num_detections) if j not in assigned_detections]
        return assignment, remaining_detections

    def run(self, my_img, p_size: int = 1):
        """
        splits image into pieces and implements cluster tracking slice by slice
        """
        thresh = 0.7
        _, my_img = cv2.threshold(my_img, thresh, 1, cv2.THRESH_BINARY)
        my_img = my_img[:265, 40:]  # trying to cut out only plots
        n_steps = my_img.shape[-1] // p_size
        for e, step in enumerate(range(0, n_steps)):
            # get piece of img
            piece = my_img[:, step * p_size:(step + 1) * p_size]
            # n of cluster changed?
            if np.all(piece == 0):
                n_of_clusters = 0
            else:
                n_of_clusters = self.estimate_cluster_num(piece, eps=2, min_samples=1)
            # make detections
            if n_of_clusters > 0:
                n_of_clusters = self.n_clusters_controle(n_clusters=n_of_clusters)
                detections = self.get_cetners(piece, n_of_cluster=n_of_clusters)
                detections += np.array([step * p_size, 0]) + np.array([40, 0])  # adding what was cut out
                # now solving an asignment problem
                self.assign(detections)


class RelateCoordinates:

    def __init__(self, segmented_labels):
        _, self.mask = cv2.threshold(segmented_labels, 0.9, 1, cv2.THRESH_BINARY)

    def estimate_cluster_num(self, eps=2, min_samples=1):
        """uses DBSCAN to estimate number of cluster, i.e. number"""
        y, x = np.where(self.mask)
        coordinates = np.column_stack((x, y))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = db.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)
        return len(unique_labels)

    def get_cetners(self):
        """knowing number of clusters find its locations
            using KMean"""
        n_of_cluster = self.estimate_cluster_num()
        y, x = np.where(self.mask)
        kmeans = KMeans(n_clusters=n_of_cluster)
        kmeans.fit(np.column_stack((x, y)))
        return kmeans.cluster_centers_

    def get_uniform_positions(self, tolerance: int = 15):
        """
        Since detectins are not aling along axes,
        they are being align using mean posiiton
        Returns
        -------
        Coordinates of x- and y-labesl in a dictionary:
        """
        centers = self.get_cetners()
        coords = []
        for slice in range(2):
            # Determine the minimum or maximum value along the current axis
            x_min_y_max = np.min(centers, axis=0) if slice == 0 else np.max(centers, axis=0)
            threshold = np.abs(centers[..., slice] - x_min_y_max[slice]) < tolerance
            # Select the positions close to x_min or y_max
            selected_positions = centers[threshold]
            # Align all positions along the mean of the current axis
            mean_value = np.mean(selected_positions, axis=0)
            selected_positions[..., slice] = int(mean_value[slice])
            # Sort the other axis values and create uniform spacing
            sorted_values = np.sort(selected_positions[..., 1 - slice])
            spacing = sorted_values - np.roll(sorted_values, 1, axis=0)
            spacing = int(np.mean(spacing[1:]))
            values_1 = [sorted_values[0] + spacing * i for i in range(len(sorted_values))]
            values_2 = [sorted_values[-1] - spacing * i for i in range(len(sorted_values))]
            uniform_values = (np.array(values_1) + np.array(values_2)[::-1]) // 2
            # Combine aligned positions with uniform distribution
            if slice == 0:
                coords.append(np.array([(int(mean_value[slice]), int(v)) for v in uniform_values]))
            else:
                coords.append(np.array([(int(v), int(mean_value[slice])) for v in uniform_values]))
        return {"x": coords[1], "y": coords[0]}


if __name__ == "__main__":

    from models_zoo.unet import UNet
    from data.data import create_dataloader
    from utils.utils import load_model
    # import cv2

    LIST_OF_COLOURS = {0: "blue", 1: "lime", 2: "red", 3: "magenta"}
    BATCH_SIZE = 32
    model = UNet()
    model = load_model(model, "./weights/08-09-07-16.pth")
    test_dataloader = create_dataloader(mode="test", num_samples=128, batch_size=BATCH_SIZE, shuffle=False, img_size=128)
    img, mask = next(iter(test_dataloader))

    pred = model(img).detach().numpy()

    for n in range(BATCH_SIZE):
        origin = img[n]
        my_img = pred[n][1]
        my_img[my_img < 0.8] = 0
        my_img[my_img > 0.5] = 1
        # my_img = cv2.erode(my_img, np.ones((1, 2), np.uint8), cv2.BORDER_REFLECT)

        tracker = Tracker()
        obj = CCD(tracker, iniertia=0.7, velocity=1, accelaration=3)
        obj.run(my_img, p_size=1)

        trace = tracker.trace
        plot = {}
        for key in trace.keys():
            plot[key] = np.array(trace[key])

        _, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].imshow(np.transpose(origin, (1, 2, 0)))
        ax[0].set_title("original")

        ax[1].imshow(my_img)
        ax[1].axis("off")
        ax[1].set_title("segmented")

        for key in trace.keys():
            ax[2].scatter(plot[key][:, 0], -plot[key][:, 1], c=LIST_OF_COLOURS[key])
        ax[2].set_ylim(-120, 0)
        ax[2].set_xlim(0, 120)
        ax[2].axis("off")
        ax[2].set_title("Clustered")

        plt.savefig(f"./torm/{n}.png")
