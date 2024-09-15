import numpy as np
from scipy.optimize import curve_fit


class FitExtract():

    def __init__(self) -> None:
        pass

    def rescale(self, dictx):
        rescaled = {}
        for key in dictx:
            new_key = (key - 66) / 24
            rescaled[new_key] = ((-1) * dictx[key] + 61) / 4
        return rescaled

    def my_func(self, x, a11, a12, a13):
        y1 = np.poly1d([a11, a12, a13])
        return y1(x)

    def fit(self, t_rescaled):
        x_values = np.array(list(t_rescaled.keys()))
        y_values = np.array(list(t_rescaled.values()))
        bnds = [(-8, -8, -8), (8, 8, 8)]
        coefficients, _ = curve_fit(self.my_func, x_values, y_values, bounds=bnds)
        return coefficients

    def fit_on(self, plot):
        t_rescaled = self.rescale(plot)
        coefs = self.fit(t_rescaled)
        return coefs


if __name__ == "__main__":

    import matplotlib.pylab as plt
    from models_zoo.unet import UNet
    from data.data import create_dataloader
    from nn_engine.utils.utilities import load_model
    from utils.tracker import CCD, Tracker

    fit_machine = FitExtract()

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
        my_img[my_img > 0.8] = 1

        tracker = Tracker()
        obj = CCD(tracker, iniertia=0.7, velocity=1, accelaration=3)
        obj.run(my_img, p_size=1)

        trace = tracker.trace
        plot = {}
        for key in trace.keys():
            plot[key] = np.array(trace[key])

        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(np.transpose(origin, (1, 2, 0)))
        ax[0].axis("off")
        ax[0].set_title("Input image")
        for key in trace.keys():
            t_dict = {p[0]: p[1] for p in plot[key]}
            coefs = fit_machine.fit_on(t_dict)
            my_f = np.poly1d(coefs)
            y = [my_f(x) for x in np.linspace(-2, 2, 100)]
            ax[1].plot(np.linspace(-2, 2, 100), y, c=LIST_OF_COLOURS[key], marker="o")

        ax[1].set_title("Extracted Image")
        ax[1].set_ylim(-10, 10)
        ax[1].set_xlim(-2, 2)

        plt.savefig(f"./torm/fit_{n}.png")
