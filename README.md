# ML for Predicting Properties of Porous Media from 2D X-ray Images

This repository provides a deep learning framework to predict the physical properties of porous media using 2D X-ray images. By using 3D convolutional neural networks (CNNs), the model estimates porosity value from input images.

## Repository Structure

The repository contains the following files:

- `requirements.txt`: list of all dependencies required to run this repository
- `CNN_model_1.py`: Defines the architecture of the 3D CNN model used for porosity prediction.
- `train.py`: Script to train the CNN model using the provided dataset.
- `test.py`: Script to evaluate the trained model's performance on test data.
- `dataset.py`: Handles data loading and preprocessing of 2D X-ray images.
- `file_path_with_labels.csv`: CSV file containing the file paths of images along with their corresponding property labels.
- `best_model.pth`: Saved weights of the best-performing model during training.
- `README.md`: Provides an overview of the project and instructions for usage.
- `LICENSE`: Specifies the licensing information for the repository.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- wandb (not necessary but helpful in logging metrics while training)
- NumPy
- tqdm (not necessary but helpful in tracking progress of the training loop)

You can install the required packages using pip:

```bash
pip install -r "requirements.txt"
```

### Dataset Preparation

1. Organize your 2D X-ray images in a directory.
2. Update the `file_path_with_labels.csv` file to include the paths to your images and their corresponding property labels. The CSV should have the following structure:

```
image_path,porosity
/path/to/image1.png, 18.5
/path/to/image2.png, 21.6
...
```

### Training the Model

To train the model, run:

```bash
python train.py
```

This script will:

- Load and preprocess the dataset.
- Train the CNN model defined in `CNN_model_1.py`.
- Save the best-performing model weights to `best_model.pth`.

### Evaluating the Model

To evaluate the trained model on test data, run:

```bash
python test.py
```

This script will:

- Load the test dataset.
- Load the trained model weights from `best_model.pth`.
- Compute and display the prediction accuracy for each property.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This work is inspired by research on machine learning applications in predicting properties of porous media from 2D X-ray images.
