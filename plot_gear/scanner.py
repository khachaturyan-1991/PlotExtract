import numpy as np
import cv2
import matplotlib.pylab as plt
import torch
import os
from nn_engine.models_zoo.unet import UNet
from nn_engine.models_zoo.cnn_lstm import CNN_LSTM
from nn_engine.utils.tracker import Tracker, CCD, RelateCoordinates
from nn_engine.utils.utilities import load_model, embedded_to_number, Rescaler
from plot_gear.plot_processor import PlotProcessor
import matplotlib
matplotlib.use('Agg')


class PlotScanner(PlotProcessor):

    def __init__(self) -> None:
        super().__init__()
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        if not os.path.exists("./checkpoints/results"):
            os.mkdir("./checkpoints/results/")
        self.RES_FOLDER = "./checkpoints/results/"
        self.img = None
        self.segmented = None
        self.rescaler = None
        model_names = "unet label_x label_y".split()
        self.models = {"unet": None, "label_x": None, "label_y": None}
        models = self._load_models()
        for i, model in enumerate(models):
            self.models[model_names[i]] = model

    def _prepare_img(self, img):
        img = np.transpose(img, (2, 0, 1)) / 255.0
        _, self.img = cv2.threshold(img, 0.9, 1, cv2.THRESH_BINARY)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig(self.RES_FOLDER + "original.png")

    def _load_models(self):
        DEPTH = 3
        unet_model = UNet(depth=DEPTH)
        unet_model = load_model(unet_model, "./pretrained/segmentation2.pth")
        crnn_models = {"x": CNN_LSTM(), "y": CNN_LSTM()}
        for i in "x y".split():
            crnn_models[i] = load_model(crnn_models[i], f"./pretrained/{i}text.pth")
        return unet_model, crnn_models["x"], crnn_models["y"]

    def _segment(self):
        segmented = self.img.copy()
        segmented = np.expand_dims(segmented, axis=0)
        segmented = torch.tensor(segmented, dtype=torch.float32)
        segmented = self.models["unet"](segmented)
        segmented = segmented.detach().numpy()
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(segmented[0][0])
        ax[0].set_title("labels")
        ax[1].imshow(segmented[0][1])
        ax[1].set_title("Plots")
        plt.savefig(self.RES_FOLDER + "segmented.png")
        self.segmented = segmented

    def _get_labels_positions(self):
        square = self.segmented[0][0].copy()
        label_extractor = RelateCoordinates(square)
        extracted_axes = label_extractor.get_uniform_positions()
        plt.imshow(np.transpose(self.img, (1, 2, 0)))
        for i in "x y".split():
            plt.scatter(extracted_axes[i][..., 0], extracted_axes[i][..., 1])
        plt.savefig(self.RES_FOLDER + "labels_positions_marked.png")
        return extracted_axes

    def _get_labels(self):
        crops_lim = {"x": [270, 290], "y": [0, 30]}
        labels_nums = {}
        for i in "x y".split():
            a, b = crops_lim[i]
            if i == "x":
                label_croped = self.img[1][a:b, :].copy()
            else:
                label_croped = self.img[1][:, a:b].copy()
            label_croped = np.expand_dims(label_croped, axis=0)
            label_croped = np.expand_dims(label_croped, axis=0)
            text_res = self.models[f"label_{i}"](torch.tensor(label_croped))
            labels_nums[i] = embedded_to_number(text_res)
        return labels_nums

    def _set_rescaler(self, labels_pos, labels_nums):
        rescaler = Rescaler(labels_pos, labels_nums)
        self.rescaler = rescaler

    def _create_tracker(self):
        plots_extracted = self.segmented[0][1].copy()
        tracker = Tracker()
        extractor = CCD(tracker, iniertia=0.9, velocity=1, accelaration=3)
        extractor.run(plots_extracted, p_size=1)
        trace = tracker.trace
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(np.transpose(self.img, (1, 2, 0)))
        for key in trace.keys():
            extracted_plots = np.array(trace[key])
            res = self.rescaler.rescale(extracted_plots)
            ax[1].plot(res[:, 0], -res[:, 1])
        ax[1].set_xlim(-2.2, 2.2)
        ax[1].set_ylim(-10.2, 10.2)
        plt.savefig(self.RES_FOLDER + "final.png")

    def handle(self, img):
        img = self._prepare_img(img)
        self._segment()
        labels_pos = self._get_labels_positions()
        labels_nums = self._get_labels()
        self._set_rescaler(labels_pos, labels_nums)
        self._create_tracker()
