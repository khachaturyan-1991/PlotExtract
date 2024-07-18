import torch
from utils import plot_hystory, parse_arguments, count_torch_parameters
import pandas as pd
from models_zoo.unet import UNet
from data.data import create_dataloader
from train.train import Trainer, test_prediction
from metrics.losses import CombinedLoss
from torchinfo import summary
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    IMG_SIZE = args.img_size

    print("Running on: ", DEVICE)

    model = UNet()
    summary(model, input_size=(1, 3, 128, 128))
    count_torch_parameters(model)

    model.to(device=DEVICE)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=1e-4)

    model_train = Trainer(model=model,
                          loss_fn=CombinedLoss(),
                          optimizer=optim,
                          device=DEVICE)

    train_dataloader = create_dataloader(num_samples=128, batch_size=BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
    test_dataloader = create_dataloader(num_samples=32, batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)
    val_dataloader = create_dataloader(num_samples=128, batch_size=BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

    FIRST_STEP = 0
    h_train, h_test = model_train.fit(train_dataloder=train_dataloader,
                                      test_dataloder=test_dataloader,
                                      output_freq=OUTPUT_FREQUENCY,
                                      epochs=EPOCHS,
                                      first_step=FIRST_STEP)

    torch.save(model.state_dict(), 'model.pth')

    plot_hystory(h_train,
                 h_test,
                 "hystory")

    test_prediction(model, test_dataloader, CombinedLoss(), "run_test")

    df = pd.DataFrame({'step': list(h_train.keys()), 'train_loss': list(h_train.values()), 'test_loss': list(h_test.values())})
    df.to_csv('./history.csv', index=False)
