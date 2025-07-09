# DLSS using SRCNN

This is a very basic implementation of NVIDIA's DLSS technology that uses Super Resolution CNN (SRCNN) to upsample images to 1920x1080p quality.

## Repository structure
```
DLSS/
├── README.md
├── DIV2K_train_HR: Folder containing 800 high resolution training images
├── DIV2K_valid_HR: Folder containing 100 high resolution validation images
├── sample_tests: Folder for sample outputs
├── weights: Folder to store trained model weights (versions of best_weights.pth)
├── config.py: Configuration file for model parameters and settings
├── dataset.py: dataset class for handling data loading and preprocessing
├── downsampling-script.py: Script to generate low-resolution images from DIV2K high-resolution dataset
├── inference.py: Script for running inference with the trained model
├── info.py: Script for printing model architecture information
├── LICENSE: Project license file
├── loss_function.py: Defines the loss function used for training the model (currently MSELoss)
├── requirements.txt: List of Python dependencies for the project
├── model.py: The SRCNN model architecture class
├── streamlit_app.py: Streamlit webpage for user interface
├── test.py: Script for evaluating the trained model
└── train.py: Script for training the SRCNN model
```

## Will update this later