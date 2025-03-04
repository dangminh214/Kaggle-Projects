# Fashion-MNIST Classification with TensorFlow and Keras

This project implements a Convolutional Neural Network (CNN) model to classify images from the Fashion-MNIST dataset using TensorFlow and Keras.

## Dataset Description

Fashion-MNIST is a dataset of Zalando's article images, consisting of:
- **Training set**: 60,000 grayscale images (28x28 pixels).
- **Test set**: 10,000 grayscale images (28x28 pixels).
- **Classes**: 10 different clothing categories.

This dataset is intended as a direct drop-in replacement for the original MNIST dataset but contains fashion-related images instead of handwritten digits.

### Class Labels:
1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle Boot  

## Steps in This Project

1. **Loading the Dataset**  
   - The dataset is imported using `keras.datasets.fashion_mnist`.  
   - Training and testing images are separated.  

2. **Data Preprocessing**  
   - Normalize pixel values to the range `[0,1]` by dividing by 255.  

3. **Visualizing Sample Images**  
   - Displaying sample images along with their labels.  

4. **Building the Model**  
   - A simple neural network consisting of:
     - `Flatten()` layer to convert 28x28 images into 1D vectors.
     - `Dense(128, activation='relu')` for feature extraction.
     - `Dense(10, activation='softmax')` for classification.  

5. **Training the Model**  
   - Model is compiled with:
     - `Adam` optimizer.
     - `sparse_categorical_crossentropy` loss function.  
   - Trained for 5 epochs.  

6. **Evaluating the Model**  
   - Model accuracy and loss are computed on the test set.  

7. **Making Predictions & Visualization**  
   - Predictions are made on test images.  
   - Results are displayed in a 10x10 grid with green labels for correct predictions and red labels for incorrect predictions.  

## Running the Project

Ensure you have Python installed along with the required dependencies.  
Install dependencies using:

```bash
pip install tensorflow numpy matplotlib
```


## Predict Result 

The trained model can classify different clothes with high accuracy.
Predicted vs. actual values are displayed in a 10x10 grid.