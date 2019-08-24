import io
import random
import pandas as pd
import numpy as np
import tensorflow as tf

from flask import Response, Flask
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from keras.models import load_model, model_from_json
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
look_back = 1
graph = tf.get_default_graph()

@app.route('/forcasting')
def plot_png():
    
    with graph.as_default():
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")

        dataset = load_data()
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        past, future = split(dataset)

        pastX, _ = create_dataset(past, look_back)
        futureX, _ = create_dataset(future, look_back)
        pastX = np.reshape(pastX, (pastX.shape[0], 1, pastX.shape[1]))
        futureX = np.reshape(futureX, (futureX.shape[0], 1, futureX.shape[1]))

        pastPredict = loaded_model.predict(pastX)
        futurePredict = loaded_model.predict(futureX)
        pastPredict = scaler.inverse_transform(pastPredict)
        futurePredict = scaler.inverse_transform(futurePredict)

        figure = create_figure(dataset, pastPredict, futurePredict, scaler)
        output = io.BytesIO()
        FigureCanvas(figure).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')


def create_figure(dataset, pastPredict, futurePredict, scaler):
    futurePredictPlot = np.empty_like(dataset)
    futurePredictPlot[:, :] = np.nan
    futurePredictPlot[len(pastPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = futurePredict
    
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Leaf Point Statistik 2019")
    axis.plot(scaler.inverse_transform(dataset[:-20]), label="Original")
    axis.plot(futurePredictPlot, label="Trend")
    axis.legend()
    return fig

def load_data():
    dataframe = pd.read_csv("data.csv", usecols=[1], header=0)
    dataset = dataframe.values
    dataset = dataset.astype("float64")
    return dataset

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def split(dataset):
    past_size = int(len(dataset) * 0.67)
    past, future = dataset[0:past_size,:], dataset[past_size:len(dataset),:]
    return past, future

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
