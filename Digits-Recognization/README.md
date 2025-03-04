# Digits Recognition with TensorFlow and Keras

This project implements a digit recognition model using the MNIST dataset and TensorFlow/Keras.

## Dataset Description

The dataset consists of gray-scale images of hand-drawn digits (0-9), each 28x28 pixels in size.  
Each pixel has an intensity value between 0 and 255, where higher values indicate darker pixels.  

- **train.csv**: Contains 785 columns. The first column ("label") is the digit. The remaining 784 columns represent pixel values.  
- **test.csv**: Contains only pixel values for testing.  

Pixel positions are indexed as `pixelx`, where `x = i * 28 + j` (i and j range from 0 to 27).

## Steps in This Project

1. **Load the MNIST dataset**  
   - The dataset is loaded using `keras.datasets.mnist`.  
   - Training and testing images are separated.  

2. **Data Visualization**  
   - The first image is displayed with its true label.  
   - A grid of images is plotted to visualize training samples.  

3. **Data Preprocessing**  
   - Pixel values are scaled to the range `[0,1]` by dividing by 255.  

4. **Building the Model**  
   - The model consists of:
     - `Flatten()` layer to convert 28x28 images into 1D vectors.
     - `Dense(128, activation='relu')` for feature extraction.
     - `Dense(10, activation='softmax')` for classification.  

5. **Training the Model**  
   - The model is compiled with:
     - `Adam` optimizer
     - `sparse_categorical_crossentropy` loss function  
   - Trained for 5 epochs.  

6. **Making Predictions**  
   - Predictions are made on test images.  
   - Results are displayed alongside ground truth labels.  

## Running the Project

Ensure you have Python installed along with the required dependencies.  
Install dependencies using:

```bash
pip install tensorflow numpy pandas matplotlib
```

## Predict Result 

The trained model can classify hand-drawn digits with high accuracy.
Predicted vs. actual values are displayed in a 10x10 grid.