from nn_engine.utils.utilities import parse_arguments

if __name__ == "__main__":

    args = parse_arguments()
    ACTION_TYPE = args.action
    IMG_TYPE = args.img_type
    # data
    IMG_SIZE = args.img_size
    FIG_SIZE = args.fig_size
    DPI = args.dpi
    NUM_OF_SAMPLES = args.num_of_samples
    AXIS = args.axis
    # train
    BATCH_SIZE = args.batch_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    DEPTH = args.unet_depth
    RUN_DESCRIPTION = args.run_description
    EXPERIMENT_NAME = args.experiment_name
    LR = args.lr
    WEIGHTS = args.weights
    DICE_COEF = args.dice_coef
    # scan
    MY_IMG = args.my_img

    if ACTION_TYPE == "data":
        if IMG_TYPE == "plots":
            from nn_engine.actors.generate_plots import generate_data
        elif IMG_TYPE == "labels":
            from nn_engine.actors.generate_labels import generate_data
        print(f"Starting {IMG_TYPE} generation")
        generate_data(mode="train", num_samples=NUM_OF_SAMPLES[0], img_size=IMG_SIZE, fig_size=FIG_SIZE, dpi=DPI, )
        generate_data(mode="validation", num_samples=NUM_OF_SAMPLES[1], img_size=IMG_SIZE, fig_size=FIG_SIZE, dpi=DPI)
        generate_data(mode="test", num_samples=NUM_OF_SAMPLES[2], img_size=IMG_SIZE, fig_size=FIG_SIZE, dpi=DPI)

    elif ACTION_TYPE == "train":
        if IMG_TYPE == "plots":
            from nn_engine.actors.train_unet import train
        elif IMG_TYPE == "labels":
            from nn_engine.actors.train_cnn_lstm import train
        train(RUN_DESCRIPTION, DICE_COEF, DEVICE, EXPERIMENT_NAME, IMG_SIZE, FIG_SIZE,
              DEPTH, BATCH_SIZE, LR, WEIGHTS, OUTPUT_FREQUENCY, EPOCHS, NUM_OF_SAMPLES, AXIS)

    else:
        import cv2
        import numpy as np
        from nn_engine.actors.extract import PlotScanner
        import matplotlib.pylab as plt
        from nn_engine.utils.fitting_zoo import FitExtract
        img = cv2.imread(MY_IMG).astype(np.float32)
        img = resized_image = cv2.resize(img, (296, 296), interpolation=cv2.INTER_CUBIC)
        scanner = PlotScanner()
        plots = scanner.handle(img)

        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img.astype(np.int32))
        ax[0].axis("off")
        ax[0].set_title("Input image")
        fit_machine = FitExtract()
        for key in plots.keys():
            plot_i = np.array(plots[key])
            coefs = fit_machine.fit_on(plot_i)
            my_f = np.poly1d(coefs)
            y = [my_f(x) for x in np.linspace(-2, 2, 100)]
            ax[1].plot(np.linspace(-2, 2, 100), y)
        ax[1].set_title("Extracted Image")
        ax[1].set_ylim(-10, 10)
        ax[1].set_xlim(-2, 2)
        plt.savefig("tested.png")
