FROM python:latest

# Install python packages
RUN pip install Flask
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install tensorflow
RUN pip install keras
RUN pip install sklearn

# Copy App
RUN mkdir /app
COPY app.py /app
COPY data.csv /app
COPY model.h5 /app
COPY model.json /app

WORKDIR /app

EXPOSE 5000

# Run the web service
ENTRYPOINT ["python", "app.py"]