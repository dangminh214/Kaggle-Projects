# Title: Rohliks Order Forecasting Challenge on Kaggle: 
### Technical Stack: Jupiter Notebook, Random Forrest Regression, Pandas, Numpy 
### Train Data Overview 
![image](https://github.com/dangminh214/Rohliks-Order-Forecasting-Challenge/assets/51837721/b11b5b71-eaeb-4bec-a330-e94d8695b9d1)

### Test Data Overview
![image](https://github.com/dangminh214/Rohliks-Order-Forecasting-Challenge/assets/51837721/ebebc1bf-94f9-41df-8d32-e4e1d6edfd09)

### Preprocess
### Merge Holidays name and holidays into one binary columns
```ruby
# Function to merge holiday and holiday name
def merge_columns(df, col1, col2): 
    if (pd.isna(df[col1]) and df[col2] == 0) or (pd.isna(df[col2]) and df[col1] == 0):
        return 0
    else:
        return 1
    
# Preprocess holiday and holiday name 
train['merged_holiday'] = train.apply(lambda row: merge_columns(row, 'holiday_name', 'holiday'), axis=1)
test['merged_holiday'] = test.apply(lambda row: merge_columns(row, 'holiday_name', 'holiday'), axis=1)
submissionId = test['id'] # Create Id for submission before preprocess Id column in test data 
```

#### Transform Id into ID, year, month, weekend 
```ruby
# Preprocess id to datetime
from sklearn.preprocessing import LabelEncoder
def transform_columns(df):
    for column in df.columns:
        if column == 'date':
            df[column] = pd.to_datetime(df[column])
            df['year'] = df[column].dt.year
            df['month'] = df[column].dt.month
            df['day'] = df[column].dt.day
            df['weekend'] = df[column].dt.weekday >= 5 
            df.drop(column, axis=1, inplace=True)
        elif df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df
```
#### Preprocess test data 
```ruby
test['merged_holiday'] = test.apply(lambda row: merge_columns(row, 'holiday_name', 'holiday'), axis=1)
test.drop(columns=['holiday_name'], inplace=True)
test.drop(columns=['holiday'], inplace=True)
```
### Create, train model 
```ruby
drop_columns = ['shutdown', 'mini_shutdown', 'blackout', 'mov_change', 'frankfurt_shutdown', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2']
X = train.drop(columns=drop_columns)
y = train["orders"]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train.drop(columns=['orders'], inplace=True)
rf_model.fit(X_train, y_train)
```

### Predict data 
```ruby
pred = rf_model.predict(test)
```

### Submit prediction data 
```ruby
# Submit 
submission = pd.DataFrame({
    'id': submissionId,
    'orders': pred
})
# Save the submission file
submission.to_csv('submission.csv', index=False)
```
