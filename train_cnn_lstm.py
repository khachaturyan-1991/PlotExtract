import torch
from utils import parse_arguments, count_torch_parameters, read_run_description, load_model
from models_zoo.cnn_lstm import CNN_LSTM
from data.loader import create_numbers_loader
from train.cnn_lstm_trainer import Trainer
from torchinfo import summary
import datetime
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    RUN_DESCRIPTION = args.run_description
    EXPERIMENT_NAME = args.experiment_name
    LR = args.lr
    WEIGHTS = args.weights

    time_stamp = datetime.datetime.now()
    run_name = time_stamp.strftime("%m-%d-%H-%M")
    if RUN_DESCRIPTION:
        run_description = read_run_description()
    else:
        run_description = "torm"
    print("Running on: ", DEVICE)
    print("With Run Time: ", run_name)

    mlflow_input = {"experiment_name": EXPERIMENT_NAME,
                    "batch_size": BATCH_SIZE,
                    "run_name": run_name,
                    "run_description": run_description,
                    "learning_rate": LR}

    train_dataloader = create_numbers_loader(mode="train", num_samples=600,
                                             batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = create_numbers_loader(mode="test", num_samples=32,
                                            batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = create_numbers_loader(mode="validation", num_samples=128,
                                           batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_LSTM(num_classes=10)
    if isinstance(WEIGHTS, str):
        model = load_model(model=model, file_name=WEIGHTS)
    summary(model, input_size=(1, 1, 30, 250))
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
    h_train, h_test = model_train.fit(train_dataloder=train_dataloader,
                                      validation_dataloder=val_dataloader,
                                      test_dataloder=test_dataloader,
                                      output_freq=OUTPUT_FREQUENCY,
                                      epochs=EPOCHS,
                                      first_step=FIRST_STEP)
