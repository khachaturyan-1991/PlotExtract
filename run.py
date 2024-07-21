import torch
from utils import plot_hystory, parse_arguments, count_torch_parameters
from models_zoo.unet import UNet
from data.data import create_dataloader
from train.train import Trainer
from metrics.losses import CombinedLoss
from torchinfo import summary
import pandas as pd
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    IMG_SIZE = args.img_size
    FIG_SIZE = args.fig_size
    DEPTH = args.depth

    print("Running on: ", DEVICE)

    train_dataloader = create_dataloader(mode="train", num_samples=1280, batch_size=BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
    test_dataloader = create_dataloader(mode="test", num_samples=128, batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)
    val_dataloader = create_dataloader(mode="validation", num_samples=128, batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

    model = UNet(depth=DEPTH)
    summary(model, input_size=(1, 3, 128, 128))
    count_torch_parameters(model)

    model.to(device=DEVICE)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=3e-4)

    model_train = Trainer(model=model,
                          loss_fn=CombinedLoss(),
                          optimizer=optim,
                          device=DEVICE)

    FIRST_STEP = 0
    experiment_description = f"U-Net depth: {DEPTH}, Image size: {IMG_SIZE}; Plot size: {FIG_SIZE}"
    h_train, h_test = model_train.fit(train_dataloder=train_dataloader,
                                      validation_dataloder=val_dataloader,
                                      test_dataloder=test_dataloader,
                                      output_freq=OUTPUT_FREQUENCY,
                                      epochs=EPOCHS,
                                      first_step=FIRST_STEP,
                                      run_description=f"./{DEPTH}_{IMG_SIZE}_{FIG_SIZE}")

    torch.save(model.state_dict(), f"./{DEPTH}_{IMG_SIZE}_{FIG_SIZE}.pth")

    plot_hystory(h_train,
                 h_test,
                 f"./history_plots/{DEPTH}_{IMG_SIZE}_{FIG_SIZE}")

    df = pd.DataFrame({"step": list(h_train.keys()),
                       "train_loss": list(h_train.values()),
                       "test_loss": list(h_test.values())})
    df.to_csv(f"./csv/{DEPTH}_{IMG_SIZE}_{FIG_SIZE}.csv", index=False)
