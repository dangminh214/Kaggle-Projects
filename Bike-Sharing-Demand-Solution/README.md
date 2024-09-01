# Title: Bike Sharing Demand - My Solution 

### Technical Stack: Jupyter Notebook, Pandas, Seaborn, Mathplotlib, Sklearn, Random Forrest Regression

### Load data
```ruby
# Loaad training data 
train = pd.read_csv("dataset/train.csv")
# train.describe(include="all")
train.head(5)
train.columns
# train.isna().sum()

# Test Data
test = pd.read_csv("dataset/test.csv")
# Check data
# test.describe(include="all")
test.isna().sum()
test.columns
```

### Visualize data 
```ruby
# Visualize data using pairplot
# Import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(train[["temp", "atemp", "holiday"]])
plt.show()
```
![image](https://github.com/dangminh214/Bike-Sharing-Demand-Solution/assets/51837721/717f7ece-1dce-4391-afe6-70074a999099)


### Preprocess data 
```ruby
columns_drop = ["count",'casual', 'registered', 'datetime']
X = train.drop(columns_drop, axis=1)
X.columns
y_test = train["count"]
```

### Create Random Forrest Regression Model 
```ruby
# Create Model
model = RandomForestRegressor()

#Train Model 
model.fit(X, y_test)
```

### Predict data: 
```ruby
test_without_datetime = test.drop(["datetime"], axis=1)
pred = model.predict(test_without_datetime)
```

### Submit prediction results
```ruby
submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': pred
})
# Save the submission file
submission.to_csv('submission.csv', index=False)
```
