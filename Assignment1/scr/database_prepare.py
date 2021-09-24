# import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# ---------------------------------------------------

# Display settings on console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
print('\nStart data preparing\n')
# ---------------------------------------------------

# read database with parsing the datetime columns directly while reading
data = pd.read_csv('../database/flight_delay.csv',
                   parse_dates=['Scheduled depature time', 'Scheduled arrival time'])
print(data.head(5))
# ---------------------------------------------------

# Count total NaN at each column in a DataFrame
print("\nCount total data in a DataFrame :", data.shape)
print("Count total NaN in a DataFrame :", data.isnull().sum().sum())
# ---------------------------------------------------------------------

# extract datetime data and calculating flight duration in minute
data.insert(loc=2,
            column='Year',
            value=data['Scheduled depature time'].dt.year)
data.insert(loc=3,
            column='Departure month',
            value=data['Scheduled depature time'].dt.month)
data.insert(loc=4,
            column='Departure hour',
            value=data['Scheduled depature time'].dt.hour)
data.insert(loc=5,
            column='Departure minute',
            value=data['Scheduled depature time'].dt.minute)
data.insert(loc=6,
            column='Departure weekday',
            value=data['Scheduled depature time'].dt.weekday)
data.insert(loc=8,
            column='Arrival hour',
            value=data['Scheduled arrival time'].dt.hour)
data.insert(loc=9,
            column='Arrival minute',
            value=data['Scheduled arrival time'].dt.minute)
data.insert(loc=10,
            column='Arrival weekday',
            value=data['Scheduled arrival time'].dt.weekday)
data.insert(loc=11,
            column='Flight duration minute',
            value=(data['Scheduled arrival time']-data['Scheduled depature time'])/np.timedelta64(1, 'm'))

data.drop(columns='Scheduled depature time', inplace=True)
data.drop(columns='Scheduled arrival time', inplace=True)
print('\n', data.head(5))
# ---------------------------------------------------------------------

# count number of unique values in each categorical feature
print("\nCount number of unique values in Departure Airport column : ", data['Depature Airport'].nunique())
print("Count number of unique values in Destination Airport column : ", data['Destination Airport'].nunique())
# -------------------------------------------------------

# convert categorical features using label encoder
cat_feature = ['Depature Airport', 'Destination Airport']
le = LabelEncoder()
le.fit(data[cat_feature[0]])
data[cat_feature[0]] = le.transform(data[cat_feature[0]])
le.fit(data[cat_feature[1]])
data[cat_feature[1]] = le.transform(data[cat_feature[1]])
print('\n', data.head(5))
print('\n', data[['Flight duration minute', 'Delay']].describe())
# -------------------------------------------------------

# Split the data into train and test due to Assignment requirement and save the results
train = data.loc[data['Year'] != 2018]
test = data.loc[data['Year'] == 2018]
# -------------------------------------------------------
train.to_csv('./train_set/train.csv', index=False)
print('\ntrain data saved to /train_set/train.csv successfully')
test.to_csv('./test_set/test.csv', index=False)
print('train data saved to /test_set/test.csv successfully')
print('finish successfully')
# -------------------------------------------
