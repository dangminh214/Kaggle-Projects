# Title: Digit Recognization using Tensorflow and MLP 
## Technical Stack: Pandas, Tensorflow, Keras, MLP 

### Create Model  
```ruby
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.leaky_relu), 
    keras.layers.Dense(10, activation=tf.nn.leaky_relu),
    keras.layers.Dense(10, activation=tf.nn.leaky_relu), 
    keras.layers.Dense(10, activation=tf.nn.leaky_relu), 
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

### Compile Model 
```ruby
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))
```

### Prediction 
```ruby
output = model.predict(test_images)
results = np.argmax(output,axis = 1)
results = pd.Series(results,name="Label")
```

### Result: submission.csv
### Result Visualization
![image](https://github.com/dangminh214/Digits-Recognization/assets/51837721/9513c55e-9240-422f-970a-9cfb791befc1)


