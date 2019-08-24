#%% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from random import seed
from random import randint

#%% Generate Data
date = np.array('2018-01-01', dtype=np.datetime64)
date = date + np.arange(601)

numbers = []
seed(1)
for _ in range(601):
    numbers.append(randint(100, 400))

#data = pd.Series(numbers, index=date)
data = pd.DataFrame()
data["Month"] = date
data["Points"] = numbers
data.to_csv("time_series.csv", header=True, index=False)

#%% Plot Time Series
dataset = pd.read_csv("data.csv", usecols=[1], header=0)
plt.plot(dataset)
plt.show()

#%% Load Data
dataframe = pd.read_csv("data.csv", usecols=[1], header=0)
dataset = dataframe.values
dataset = dataset.astype("float64")

#%% Scale Values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset

#%% Train / Test split
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

#%% Create Dataset
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#%% Reshape
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
testX

#%% Create and fit LSTM Network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back), activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, shuffle=True)

#%% Predict
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#%% Plot Result
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(trainPredictPlot, label="Training set prediction")
plt.plot(testPredictPlot, label="Test set prediction")
plt.legend()
plt.show()

#%% Future Predict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(dataset[:-4]), label = "True value")
plt.plot(testPredictPlot, label="Forcast")
plt.legend()
plt.show()

#%% Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#%%