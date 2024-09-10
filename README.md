PlotExtract
'''Bash
PlotExtract is a Python project aimed at extracting numerical information from images with plots using deep learning. The project focuses on segmenting relevant parts of the image and extracting the corresponding axis labels as text. This project is useful for anyone needing to automate the extraction of digital plot data from images, such as researchers and data analysts.

Features

	•	Segmentation: Segments plots and axis labels using a neural network with a UNet architecture.
	•	Label Extraction: Extracts axis labels from the segmented parts using a CNN-LSTM architecture.
	•	Coordinate Transformation: Extracts plot coordinates in pixel space and rescales them to actual image coordinates using axis information.

How It Works

The extraction process is divided into three main steps:

	1.	Segmentation: The relevant parts of the plot (axes and labels) are segmented using a neural network with a UNet architecture.
	2.	Label Extraction: Axis labels are extracted from the segmented regions using a CNN-LSTM model, designed specifically for recognising numbers from plot axes.
	3.	Coordinate Transformation: The pixel coordinates of plot points are rescaled based on the extracted labels to provide the actual coordinates used in the plot.


Installation

To install the necessary dependencies, run:

	pip install -r requirements.txt

Usage

You can use PlotExtract via the plot_extract.py script. To run the script, use the following command:

	python plot_extract.py --image /path/to/your/image.png

Example usage:
	
	python plot_extract.py --path_to_image ./example_plot.png

In addition to its core functionality, PlotExtract also provides tools for training and fine-tuning the underlying neural network models. This allows users to customize and improve the performance of both the segmentation model and the label extraction model based on their specific dataset.
To generate the necessary training, validation, and test data, use the generate_data.py script. You can specify the mode of data generation by running the following command:
	python generate_data.py --mode <mode>
The mode parameter can be set to either segmentation for generating data related to the segmentation model (UNet) or labels for generating data for the label extraction model (CNN-LSTM).
Once the data has been generated, you can initiate the training process by running the train.py script:
	python train.py --mode <mode>
Similar to the data generation step, the mode parameter can be set to either segmentation or labels, depending on whether you’re training the segmentation model (UNet) or the label extraction model (CNN-LSTM), respectively.


Project Structure


PlotExtract/
│
├── plot_extract.py                # Main script to run the extraction pipeline
├── data_gen.py                    # Script to generate data for training and testing
├── train.py                       # Script to run the training pipeline
│
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── TrainBeast.ipynb
│   ├── TextBeast.ipynb
│   └── WorkingBeast.ipynb
│
├── model_zoo/                     # Contains the model implementations
│   ├── unet.py                    # UNet model for segmentation
│   └── cnn_lstm.py                # CNN-LSTM model for label extraction
│
├── data/                          # Folder for generated data
│   ├── generate_labels.py         # Script to generate label data
│   ├── generate_plots.py          # Script to generate plot data
│   ├── labels/                    # Directory for labels data
│   │   ├── train/                 # Training labels
│   │   ├── validation/            # Validation labels
│   │   └── test/                  # Test labels
│   └── plots/                     # Directory for plot data
│       ├── train/                 # Training plot data
│       ├── validation/            # Validation plot data
│       └── test/                  # Test plot data
│
├── train_zoo/                     # Scripts for training models
│   ├── train_unet.py              # Training script for the UNet model
│   └── train_cnn_lstm.py          # Training script for the CNN-LSTM model
│
├── metrics_zoo/                   # Custom metrics and loss functions
│   └── losses.py                  # Contains loss functions for training

Contributing

If you want to contribute to PlotExtract, feel free to submit pull requests or open issues for feature suggestions or bug reports. Make sure to add unit tests for any new features or changes.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For any questions or issues, feel free to contact the author:

	•	Your Name: dr.khachaturyan@gmail.com
	•	GitHub: https://github.com/khachaturyan-1991
'''