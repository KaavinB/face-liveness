# Face Liveness Detection

This project implements a face liveness detection system using deep learning to distinguish between real faces and spoofed images.

## Files

- `liveness.py`: Script to run liveness detection on a single image
- `model.ipynb`: Jupyter notebook containing model training code

## Model Architecture

The model uses a convolutional neural network (CNN) architecture:

- Input shape: (128, 128, 3)
- 3 convolutional layers with ReLU activation
- Max pooling and dropout layers
- Fully connected layers
- Output: Binary classification (real vs spoof)

## Training

The model is trained on a custom dataset of real and spoofed face images. Data augmentation is applied during training. The training process uses:

- Binary crossentropy loss
- Adam optimizer
- 50 epochs
- Batch size of 12

## Usage

To run liveness detection on a new image:

1. Ensure you have the required dependencies installed
2. Place the trained model file `liveness.h5` in the same directory as `liveness.py`
3. Run:
4. The script will load the image, run it through the model, and output the prediction (real or spoof) along with the confidence score.

## Performance

The model achieves over 90% accuracy on the training set and around 75-80% accuracy on the validation set after 50 epochs of training.

## Future Work

- Collect more diverse training data
- Experiment with different model architectures
- Implement real-time liveness detection on video streams

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
