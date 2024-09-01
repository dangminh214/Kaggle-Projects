# Title: House Price Prediction on Kaggle 
### Technical Stack: Jupyter Notebook, Numpy, Pandas, Mean Square Error, Random Forrest Regression, Mathplotlib 

### Import Data to train and test 
```ruby
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
```

### Specifically handling string columns and missing values
#### Preprocess train data
```ruby
string_columns = train.select_dtypes(include='object')
unique_counts = string_columns.apply(lambda col: col.unique())

# Mapping: new data frames
mapping = {}
for col in string_columns:
    unique_values = string_columns[col].dropna().unique()
    mapping[col] = {value: i for i, value in enumerate(unique_values)}
    
for col, map_dict in mapping.items():
    train[col] = train[col].map(map_dict)

# Fill all NaN = mean
for col in train.columns: 
    train[col].fillna(round(train[col].mean()), inplace=True)
```
#### Preprocess test data 
```ruby
string_columns = test.select_dtypes(include='object')
unique_counts = string_columns.apply(lambda col: col.unique())

mapping = {}
for col in string_columns:
    unique_values = string_columns[col].dropna().unique()
    mapping[col] = {value: i for i, value in enumerate(unique_values)}
    
for col, map_dict in mapping.items():
    test[col] = test[col].map(map_dict)

# Fill all NaN = mean
for col in test.columns: 
    test[col].fillna(round(test[col].mean()), inplace=True)
```
### Create and train model
```ruby
model = RandomForestRegressor()
# Fit model
model.fit(X, y)

# Make predictions on the test set
y_pred = model.predict(test)
```

### Submit result 
```ruby
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': y_pred
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
```
