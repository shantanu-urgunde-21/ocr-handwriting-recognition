Handwritten Text Recognition with PyTorch

A complete pipeline for training a CRNN (Convolutional Recurrent Neural Network) model to recognize and transcribe handwritten text from images.
Features

    End-to-End Pipeline: From data preprocessing to training and inference.

    CRNN Architecture: Implements a standard and effective model for sequence recognition.

    Training from Scratch: The model learns entirely from the provided dataset.

    Data Augmentation & Resume Training: Includes features to improve model robustness and continue training from the last checkpoint.

    Inference Scripts: Provides a standalone script and a minimal notebook snippet to predict on new images.

Model Architecture

The model is a Convolutional Recurrent Neural Network (CRNN) composed of three parts:

    CNN: Extracts visual features from the image.

    RNN (Bi-LSTM): Analyzes the sequence of features to understand context.

    CTC Loss: Enables the model to be trained on unaligned text sequences.

Dataset

The model is trained on the IAM Handwriting Database, loaded directly from the Hugging Face datasets library.
Setup and Installation

    Clone the repository:

    git clone https://github.com/shantanu-urgunde-21/ocr-handwriting-recognition.git
    cd ocr-handwriting-recognition

    Install Dependencies:
    All required libraries can be installed by running the pip install command found in the first cell of the handwriting_recognition.ipynb notebook.

    GPU Support (NVIDIA only):
    For GPU acceleration, ensure you have the correct NVIDIA drivers installed and follow the instructions on the official PyTorch website to install PyTorch with CUDA support.

Usage
1. Training the Model

The main training logic is in handwriting_recognition.ipynb.

    To train: Open and run all cells in the notebook.

    Checkpoints: The best model is saved as handwriting_recognizer_best.pth. The script will automatically load this file to resume training if it exists.

2. Running Predictions
How to Prepare Images for Prediction

This is the most important step for getting accurate results. The model was trained on images of single, horizontal lines of text. Your images must be prepared in the same way.

    Rule 1: One Line Only. Crop the image so it contains only a single line of handwriting.

    Rule 2: Clean Background. Use images with high contrast (e.g., dark ink on light paper) and avoid shadows or complex backgrounds.

    Rule 3: Leave Padding. Do not crop tightly around the text. Leave some empty space above, below, and on the sides.

Good Input
A single, clean line of text.	

Bad Input
Multiple lines, which will be distorted.
Running the Script

Use the predict.py script to run inference from your terminal:

# Predict a single image
python predict.py path/to/your/image.png

# Predict multiple images
python predict.py image1.jpg image2.png

The script will load handwriting_recognizer_best.pth and print the predicted text.
