# Title: Titanic Spaceship Solution on Kaggle 
### Technical Stack: Jupyter Notebook, Logistic Regression - Binary Classification, Numpy, Matplotlib, Pandas, Train test split 
### Import dataset 
```ruby
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
```

### Preprocess Data
```ruby
test_string = test["PassengerId"]
```
```ruby
# Preprocess data: string => int

mappings = {}
for col in train.columns:
    unique_values = train[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    mappings[col] = mapping

# Step 3: Replace the original string values with these integers
for col, mapping in mappings.items():
    train[col] = train[col].map(mapping)
y = train["Transported"]
X = train.drop(["Transported"],axis=1)
```

### Split train test data to avoid Bias
```ruby
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Preprocess test data 
```ruby
# Preprocess Data test csv 

mappings = {}
for col in test.columns:
    unique_values = test[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    mappings[col] = mapping

# Step 3: Replace the original string values with these integers
for col, mapping in mappings.items():
    test[col] = test[col].map(mapping)

print(test)
print(test_string)
```

### Create Model and predict 
```ruby
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(test)
```

### Visualize Data 
![image](https://github.com/dangminh214/Titanic-Space-Ship-Kaggle/assets/51837721/912f28d4-4d89-4834-b252-afda7f8c2009)
![image](https://github.com/dangminh214/Titanic-Space-Ship-Kaggle/assets/51837721/6cf7935f-c59d-42eb-8c23-4e38eb819520)


