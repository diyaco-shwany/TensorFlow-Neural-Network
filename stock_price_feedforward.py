# -*- coding: utf-8 -*-
"""
The following code will use a simple feed forward Neural Network which takes historical
stock price data as inputs to predict values like highest stock price in the near future

I will predict a single stock price initially for the next day

Tensorflow, numpy
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


FILE_NAME = "GOOG.csv"



def read_normal_data(end_index):
    
    stock_data = pd.read_csv(
        FILE_NAME, 
        names = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
        header=0,
        skipinitialspace = True)
    
    stock_features = stock_data.copy()
    stock_features.tail()
    
    
    train_dataset_all = stock_features.iloc[1:int(len(stock_features)*0.8), :]
    test_dataset_all = stock_features.drop(train_dataset_all.index)
    #print(np.array(train_dataset))
    
    train_dataset = train_dataset_all.pop("Close").astype(float).values
    test_dataset = test_dataset_all.pop("Close").astype(float).values
    
    #train_dataset = np.expand_dims(train_dataset, -1)
    
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.expand_dims(train_dataset, -1))
    normalizer.mean.numpy()
    
    
    num_samples = len(train_dataset) - end_index - 1
    num_samples_1 = len(test_dataset) - end_index - 1
    train_data_fin = np.zeros((num_samples, end_index))
    train_label_fin = np.zeros(num_samples)
    test_data_fin = np.zeros((num_samples_1, end_index))
    test_label_fin = np.zeros(num_samples_1)
    

    for j in range(num_samples_1):
        test_data_fin[j] = train_dataset[j: end_index + j]
        test_label_fin[j] = train_dataset[end_index + j]
    for i in range(num_samples):
        train_data_fin[i] = train_dataset[i: end_index + i]
        train_label_fin[i] = train_dataset[end_index + i]
        
    print(f"Shape of training data: {train_data_fin.shape}")  # Debug print
    print(f"Shape of training labels: {train_label_fin.shape}")
    
    train_data_fin = normalizer(train_data_fin).numpy()
    #train_label_fin = normalizer(train_label_fin)
    test_data_fin = normalizer(test_data_fin).numpy()

    return train_data_fin, train_label_fin, normalizer, test_data_fin, test_label_fin
    
    
def model_create(training_data):
    
    model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape = (training_data.shape[1],)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(1)
    ])


    model.compile(loss='mean_absolute_error',
              optimizer=tf.keras.optimizers.Adam(0.001))
    return model
    
    
    
def main():
    
    training_data, training_labels, close_normalizer, test_data, test_labels = read_normal_data(59)
    
    stock_model = model_create(training_data)
    
    stock_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')
    
    stock_model.fit(
        training_data,
        training_labels,
        validation_split=0.2,
        verbose=1, epochs=200)
    
    print(stock_model.predict(test_data))
    print(test_labels)
    

main()




