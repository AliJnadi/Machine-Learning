# import required libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
# %matplotlib inline
# -------------------------------

# Display settings on console and sub_plotting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
print("\nStart outliers filtering")
# ---------------------------------------------------

# read train databases
data = pd.read_csv('./train_set/train.csv')
# ---------------------------------------------------

# Check if there is an outliers in database by applying Isolated Forest on one month
random_state = np.random.RandomState(42)
outlier_test = data.loc[data['Departure month'] == 1]
clf = IsolationForest(n_estimators=100, contamination='auto',
                      max_features=12,
                      random_state=random_state)
test = clf.fit_predict(outlier_test)
print('\nNumber of Outliers in one month sample : ', len(np.where(test == -1)[0]))
# -------------------------------------------------------

# plot setting
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
fig.suptitle('Training data preparing plots')
plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.8,
                    top=0.9,
                    wspace=1,
                    hspace=0.4)
# plotting Flight duration in minute vs Delay plotting
data.plot(x='Flight duration minute', y='Delay', style='.', ax=ax[0, 0])
ax[0, 0].set_title('Flight duration in minute vs Delay in minute')
ax[0, 0].set(xlabel='Flight duration in minute', ylabel='Delay in minute')
# Visualize outlier in train data
data.boxplot(column=['Flight duration minute'], ax=ax[1, 0])
ax[1, 0].set_title('boxplot Flight duration in minute before removing outliers')
counts, bins = np.histogram(data['Flight duration minute'])
ax[2, 0].hist(bins[:-1], bins, weights=counts, ec='black')
ax[2, 0].set_title('Flight duration in minute Histogram')
# -------------------------------------------------------

# Outliers removing with Isolated Forest algorithm
test = clf.fit_predict(data)
print('\nNumber of data before filtering : ', data.shape[0])
data = data.iloc[np.where(test != -1)]
print('Number of data after filtering : ', data.shape[0])
# -------------------------------------------------------

# plotting
# plotting Flight duration in minute vs Delay plotting
data.plot(x='Flight duration minute', y='Delay', style='.', ax=ax[0, 1])
ax[0, 1].set_title('Flight duration in minute vs Delay in minute')
ax[0, 1].set(xlabel='Flight duration in minute', ylabel='Delay in minute')
# Visualize outlier in train data
data.boxplot(column=['Flight duration minute'], ax=ax[1, 1])
ax[1, 1].set_title('boxplot Flight duration in minute after removing outliers')
counts, bins = np.histogram(data['Flight duration minute'])
ax[2, 1].hist(bins[:-1], bins, weights=counts, ec='black')
ax[2, 1].set_title('Flight duration in minute Histogram')
plt.show()
# -------------------------------------------------------

data.to_csv('./train_set/train_o.csv', index=False)
print('\nfiltered train data saved to /train_set/train_o.csv successfully')
print('\nfinish successfully.')
# # ---------------------------------------------------

