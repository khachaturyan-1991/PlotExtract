import torch
from utils.utils import parse_arguments, count_torch_parameters, read_run_description, load_model
from models_zoo.cnn_lstm import CNN_LSTM
from data.loader import create_numbers_loader
from train.trainer_cnn_lstm import Trainer
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
        run_description = "torm"
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
                    "dice_coef": DICE_COEF,
                    "axis": AXIS}

    train_dataloader = create_numbers_loader(mode="train", axis=AXIS, num_samples=NUM_OF_SAMPLES[0],
                                             batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = create_numbers_loader(mode="validation", axis=AXIS, num_samples=NUM_OF_SAMPLES[1],
                                           batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = create_numbers_loader(mode="test", axis=AXIS, num_samples=NUM_OF_SAMPLES[2],
                                            batch_size=BATCH_SIZE, shuffle=True)

    model = CNN_LSTM(num_classes=10)
    if isinstance(WEIGHTS, str):
        model = load_model(model=model, file_name=WEIGHTS)
    summary(model, input_size=(1, 1, 26, 296))
    count_torch_parameters(model)

    model.to(device=DEVICE)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=LR)

    model_train = Trainer(model=model,
                          loss_fn=torch.nn.CrossEntropyLoss(),
                          optimizer=optim,
                          device=DEVICE,
                          num_classes=10,  # has to be input
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
    AXIS = args.axis
    NUM_OF_SAMPLES = args.num_of_samples

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
