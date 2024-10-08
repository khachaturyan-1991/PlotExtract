import torch
from nn_engine.utils.utilities import parse_arguments, count_torch_parameters, read_run_description, load_model
from nn_engine.models_zoo.unet import UNet
from nn_engine.data.loader import create_plots_loader
from nn_engine.train.trainer_unet import Trainer
from nn_engine.metrics_zoo.losses import DiceLoss, SMSE, CombinedLoss
from torchinfo import summary
import datetime
import matplotlib
matplotlib.use('Agg')


def train(RUN_DESCRIPTION,
          DICE_COEF,
          DEVICE,
          EXPERIMENT_NAME,
          IMG_SIZE,
          FIG_SIZE,
          DEPTH,
          BATCH_SIZE,
          LR,
          WEIGHTS,
          OUTPUT_FREQUENCY,
          EPOCHS,
          NUM_OF_SAMPLES,
          AXIS):
    time_stamp = datetime.datetime.now()
    run_name = time_stamp.strftime("%m-%d-%H-%M")
    if RUN_DESCRIPTION:
        run_description = read_run_description()
    else:
        run_description = f"DicePart: {DICE_COEF}"
    print("Running on: ", DEVICE)
    print("With Run Time: ", run_name)

    mlflow_input = {"experiment_name": EXPERIMENT_NAME,
                    "img_size": IMG_SIZE,
                    "fig_size": FIG_SIZE,
                    "depth": DEPTH,
                    "batch_size": BATCH_SIZE,
                    "run_name": run_name,
                    "run_description": run_description,
                    "learning_rate": LR,
                    "dice_coef": DICE_COEF}

    train_dataloader = create_plots_loader(mode="train", num_samples=NUM_OF_SAMPLES[0],
                                           batch_size=BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
    val_dataloader = create_plots_loader(mode="validation", num_samples=NUM_OF_SAMPLES[1],
                                         batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)
    test_dataloader = create_plots_loader(mode="test", num_samples=NUM_OF_SAMPLES[2],
                                          batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

    model = UNet(depth=DEPTH)
    if isinstance(WEIGHTS, str):
        model = load_model(model=model, file_name=WEIGHTS)
    summary(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE))
    count_torch_parameters(model)

    model.to(device=DEVICE)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=LR)

    model_train = Trainer(model=model,
                          loss_fn=CombinedLoss(dice_weight=DICE_COEF),
                          segment_loss=DiceLoss(),
                          numeric_loss=SMSE(),
                          optimizer=optim,
                          device=DEVICE,
                          **mlflow_input)

    FIRST_STEP = 0
    model_train.fit(train_dataloder=train_dataloader,
                    validation_dataloder=val_dataloader,
                    test_dataloder=test_dataloader,
                    output_freq=OUTPUT_FREQUENCY,
                    epochs=EPOCHS,
                    first_step=FIRST_STEP)


if __name__ == "__main__":

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    IMG_SIZE = args.img_size
    FIG_SIZE = args.fig_size
    DEPTH = args.unet_depth
    RUN_DESCRIPTION = args.run_description
    EXPERIMENT_NAME = args.experiment_name
    LR = args.lr
    WEIGHTS = args.weights
    DICE_COEF = args.dice_coef
    NUM_OF_SAMPLES = args.num_of_samples
    AXIS = args.axis

    train(RUN_DESCRIPTION,
          DICE_COEF,
          DEVICE,
          EXPERIMENT_NAME,
          IMG_SIZE,
          FIG_SIZE,
          DEPTH,
          BATCH_SIZE,
          LR,
          WEIGHTS,
          OUTPUT_FREQUENCY,
          EPOCHS,
          NUM_OF_SAMPLES,
          AXIS)
