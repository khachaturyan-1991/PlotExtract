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
        from nn_engine.actors.extract import plots_extract
        plots_extract()
