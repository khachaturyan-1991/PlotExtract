from torch.utils.data import DataLoader
from nn_engine.data.sets import PlotImageLoader, NumberDataset


def create_plots_loader(num_samples: int,
                        batch_size: int,
                        shuffle: bool = False,
                        img_size: int = 128,
                        mode: str = "train"
                        ):
    dataset = PlotImageLoader(mode=mode, num_samples=num_samples, img_size=img_size)
    print(dataset.__len__())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_numbers_loader(num_samples: int,
                          batch_size: int,
                          shuffle: bool = False,
                          mode: str = "train",
                          axis: str = "x"
                          ):
    dataset = NumberDataset(mode=mode, axis=axis, num_samples=num_samples)
    print(dataset.__len__())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
