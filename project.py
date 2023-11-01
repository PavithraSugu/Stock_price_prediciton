#import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#upload the file
from google.colab import files
uploadf = files.upload()

#load the data
df = pd.read_csv('MSFT.csv')
df

#data outliners
thresholds ={'Date': ("1986-03-19","2020-01-02")}
for col, (lower,upper) in thresholds.items():
  df = df[(df[col] >= lower) & (df[col] <= upper)]
print(df)

#checking missing values
missing_values = df.isnull().sum()
print(missing_values)

#create a new dataframe with only the 'Close column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train model on
training_data_len = math.ceil(len(dataset) * .8 )

training_data_len

#Scale the data
#normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming you have a 'Date' column and a 'Close' column in your data
# You can adjust these column names based on your data structure
X = df[['Date']]  # Features (in this case, only the date)
y = df['Close']   # Target variable (stock prices)

# Set the proportion of data to use for testing (e.g., 20%)
test_size = 0.2

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Now you have your data split into training and test sets:
# X_train, y_train are for training
# X_test, y_test are for testing

#Create the traning dataset
#Create the scaled training dataset
train_data = scaled_data[0:training_data_len , :]

#Splitrain_data.info() the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 60:
    print(x_train)
    print(y_train)
    print()

# Convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape x_train for use in an LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Print the shape to confirm
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

#create testing dataset
#create a new containing scaled values
test_data = scaled_data[training_data_len - 60: , :]

#create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#reshape the data
x_test = np.array(x_train)

# Reshape x_train for use in an LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#define an object (initializing RNN)
import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True, input_shape=(60, 1)))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units=120, activation='relu'))
# dropout layer
model.add(tf.keras.layers.Dropout(0.2))
#output layer
model.add(tf.keras.layers.Dense(units=1))

model.summary()

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_absolute_error'])

model.fit(x_train,y_train, batch_size=32, epochs=50)

# Make predictions on the test data
predictions = model.predict(x_test)

# Inverse transform the scaled predictions to get actual values
predictions = scaler.inverse_transform(predictions)
