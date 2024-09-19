PlotExtract
'''Bash
PlotExtract is a Python project aimed at digitalise plots from images using deep learning. The digitaliser extract and separates plots and above all allows to fit them with a high order polynomial or a custom function (not yet implemented). Fitting the extracted plots returns fitting parameters and allows to restore the plots with any resolution.

The project is useful for anyone needing to automate the extraction of digital plot data from images, such as researchers and data analysts.

Digitalisation if happening in three steps:
    •    Segmentation: Segments plots and axis labels using a neural network with a UNet architecture.
    •    Label Extraction: Extracts axis labels from the segmented parts using a CNN-LSTM architecture.
    •    Coordinate Transformation: Extracts plot coordinates in pixel space and rescales them to actual image coordinates using axis information.

=========\n
How Does It Work

The extraction process is divided into three main steps:

    1.    Segmentation: The relevant parts of the plot (axes and labels) are segmented using a neural network with a UNet architecture.
    2.    Label Extraction: Axis labels are extracted from the segmented regions using a CNN-LSTM model, designed specifically for recognising numbers from plot axes.
    3.    Coordinate Transformation: The pixel coordinates of plot points are rescaled based on the extracted labels to provide the actual coordinates used in the plot.


Installation

To download the project use a command
    git clone https://github.com/khachaturyan-1991/PlotExtract.git
To install the necessary dependencies, run:
    pip install -r requirements.txt

=========\n
How to use:

A user can extract plots from an image using command:
    python -m plot_extract --my_img ./example_plot.png
assuming that models were trained and weights were saved in a folder pertained (in a same directory as the nn_engine). Weights names are segmentation.pth, labels_x.pth, and labesl_y.pth.

In addition to its core functionality, PlotExtract also provides tools for training and fine-tuning the underlying neural network models. This allows users to customise and improve the performance of both the segmentation model and the label extraction model based on their specific dataset.
User can train or fine-tune models as
    python -m nn_engine --action train --list_of_hyperparameters

User can create a dataset for training as
    python -m nn_engine --action data --list_of_hyperparameters


Project Structure

```
PlotExtract/
│
├──nn_engine
│   ├──actors                        # scripts to perform the main actions:
│   │    ├──extract.py                # extract plots from images
│   │    ├──generate_plots.py        # generates plots to train segmentation models
│   │    ├──generate_labels.py        # generates data to learn number on labels
│   │    ├──train_unet.py            # train segmentation model
│   │    ├──train_cnn_lstm.py        # train model to extract numbers from labels
│   │    └──__init__.py
│   │
│   ├──metrics_zoo
│   │    └──losses.py                # collection of loss funcitons
│   │
│   ├──models_zoo                    # deep neural models to perform
│   │    ├──unet.py                    # segmentations
│   │    ├──cnn_lstm.py                # extract numbers from labels
│   │    └──__init__.py
│   │
│   ├──train                        # train pipe-lines to train
│   │    ├──train_unet.py            # segmentation model
│   │    ├──train_cnn_lstm.py        # text extraction model
│   │    └──__init__.py
│   │
│   ├──utils                        # utilities
│   │    ├──tracking.py                # separate segmented lines from one another
│   │    ├──fitting_zoo.py            # allows fiting separated lines
│   │    ├──utilities.py                # heler functions
│   │    └──__init__.py
│   │
│   ├──__init__.py
│   └──__main__.py                    # main funciton that is executed when the packedge is called
|
├──mlruns                            # contains mlflow data
|
├──pretrained                        # contains weights for
│    ├──segmentation.pth                # plots and labels segmentations
│    ├──xtext.pth                    # extracting numbers from x-labels
│    └──ytext.pth                    # and y-labels
|

```

Contributing

If you want to contribute to PlotExtract, feel free to submit pull requests or open issues for feature suggestions or bug reports. Make sure to add unit tests for any new features or changes.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For any questions or issues, feel free to contact the author:

    •    Your Name: dr.khachaturyan@gmail.com
    •    GitHub: https://github.com/khachaturyan-1991
'''
