import numpy as np
import cv2
import matplotlib.pylab as plt
import torch
from nn_engine.models_zoo.unet import UNet
from nn_engine.models_zoo.cnn_lstm import CNN_LSTM
from nn_engine.utils.tracker import Tracker, CCD, RelateCoordinates
from nn_engine.utils.utilities import load_model, embedded_to_number, Rescaler
import os
import matplotlib
matplotlib.use('Agg')


def plots_extract():
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    if not os.path.exists("./checkpoints/results"):
        os.mkdir("./checkpoints/results/")
    RES_FOLDER = "./checkpoints/results/"
    # prepare models
    DEPTH = 3
    unet_model = UNet(depth=DEPTH)
    unet_model = load_model(unet_model, "./pretrained/segmentation2.pth")
    crnn_models = {"x": CNN_LSTM(), "y": CNN_LSTM()}
    for i in "x y".split():
        crnn_models[i] = load_model(crnn_models[i], f"./pretrained/{i}text.pth")
    # get image
    # img = np.load("./data/plots/test/image/0.npy")
    img = cv2.imread("./data/plots/test/image/0.png").astype(np.float32)
    img = np.transpose(img, (2, 0, 1)) / 255.0
    _, img = cv2.threshold(img, 0.9, 1, cv2.THRESH_BINARY)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(RES_FOLDER + "original.png")
    # segment
    seg_res = img.copy()
    seg_res = np.expand_dims(seg_res, axis=0)
    seg_res = unet_model(torch.tensor(seg_res, dtype=torch.float32)).detach().numpy()
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(seg_res[0][0])
    ax[0].set_title("Lables")
    ax[1].imshow(seg_res[0][1])
    ax[1].set_title("Plots")
    plt.savefig(RES_FOLDER + "segmented.png")
    # extract labels positions
    square = seg_res[0][0].copy()
    label_extractor = RelateCoordinates(square)
    extracted_axes = label_extractor.get_uniform_positions()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    for i in "x y".split():
        plt.scatter(extracted_axes[i][..., 0], extracted_axes[i][..., 1])
    plt.savefig(RES_FOLDER + "labels_positions_marked.png")
    # extracting numbers on x axis
    crops_lim = {"x": [270, 290], "y": [0, 30]}
    nums = {}
    for i in "x y".split():
        a, b = crops_lim[i]
        if i == "x":
            label_croped = img[1][a:b, :].copy()
        else:
            label_croped = img[1][:, a:b].copy()
        label_croped = np.expand_dims(label_croped, axis=0)
        label_croped = np.expand_dims(label_croped, axis=0)
        text_res = crnn_models[i](torch.tensor(label_croped))
        nums[i] = embedded_to_number(text_res)
    # rescale
    rescaler = Rescaler(extracted_axes, nums)
    # alltogether
    plots_extracted = seg_res[0][1].copy()
    tracker = Tracker()
    extractor = CCD(tracker, iniertia=0.9, velocity=1, accelaration=3)
    extractor.run(plots_extracted, p_size=1)
    trace = tracker.trace

    _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(np.transpose(img, (1, 2, 0)))
    for key in trace.keys():
        extracted_plots = np.array(trace[key])
        res = rescaler.rescale(extracted_plots)
        ax[1].plot(res[:, 0], -res[:, 1])
    ax[1].set_xlim(-2.2, 2.2)
    ax[1].set_ylim(-10.2, 10.2)
    plt.savefig(RES_FOLDER + "final.png")
