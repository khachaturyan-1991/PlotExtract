import cv2
import numpy as np
from plot_gear.scanner import PlotScanner


if __name__ == "__main__":
    img = cv2.imread("./data/plots/test/image/0.png").astype(np.float32)
    img = resized_image = cv2.resize(img, (296, 296), interpolation=cv2.INTER_CUBIC)
    obj = PlotScanner()
    obj.handle(img)
