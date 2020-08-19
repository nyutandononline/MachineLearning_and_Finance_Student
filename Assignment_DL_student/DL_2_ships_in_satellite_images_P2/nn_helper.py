import math

import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import pickle
import tensorflow as tf
from nose.tools import assert_equal
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


import json

class Helper():
  def __init__(self):
    return

  def scaleData(self, data, labels):
    X = data / 255.
    y = to_categorical(labels, num_classes=2)

    return X, y

  def getData(self, DATA_DIR, dataset):
    data,labels = self.json_to_numpy( os.path.join(DATA_DIR,dataset) )
    return data, labels

  def showData(self, data, labels, num_cols=5):
    # Plot the first num_rows * num_cols images in X
    (num_rows, num_cols) = ( math.ceil(data.shape[0]/num_cols), num_cols)

    fig = plt.figure(figsize=(10,10))
    # Plot each image
    for i in range(0, data.shape[0]):
        img, img_label = data[i], labels[i]
        ax  = fig.add_subplot(num_rows, num_cols, i+1)
        _ = ax.set_axis_off()
        ax.set_title(img_label)

        _ = plt.imshow(img)
    fig.tight_layout()

    return fig

  def modelPath(self, modelName):
      return os.path.join(".", "models", modelName)

  def saveModel(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      # Save JSON config to disk
      json_config = model.to_json()
      with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
          json_file.write(json_config)
      # Save weights to disk
      model.save_weights(os.path.join(model_path, 'weights.h5'))
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))

  def loadModel(self, modelName):
      model_path = self.modelPath(modelName)
      
      # Reload the model from the 2 files we saved
      with open(os.path.join(model_path, 'config.json')) as json_file:
          json_config = json_file.read()
    
      model = tf.keras.models.model_from_json(json_config)
      model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
      model.load_weights(os.path.join(model_path, 'weights.h5'))
      
      return model

  def saveModelNonPortable(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      model.save( model_path )
      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))
   
  def loadModelNonPortable(self, modelName):
      model_path = self.modelPath(modelName)
      model = self.load_model( model_path )
      
      # Reload the model 
      return model

  def saveHistory(self, history, model_name):
      history_path = self.modelPath(model_name)

      try:
          os.makedirs(history_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=history_path))

      # Save JSON config to disk
      with open(os.path.join(history_path, 'history'), 'wb') as f:
          pickle.dump(history.history, f)

  def loadHistory(self, model_name):
      history_path = self.modelPath(model_name)
      
      # Reload the model from the 2 files we saved
      with open(os.path.join(history_path, 'history'), 'rb') as f:
          history = pickle.load(f)
      
      return history

  def json_to_numpy(self, json_file):
    # Read the JSON file
    f = open(json_file)
    dataset = json.load(f)
    f.close()

    data = np.array(dataset['data']).astype('uint8')
    labels = np.array(dataset['labels']).astype('uint8')

    # Reshape the data
    data = data.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

    return data, labels


  def plotTrain(self, history, model_name="???"):
    fig, axs = plt.subplots( 1, 2, figsize=(12, 5) )

    # Plot loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title(model_name + " " + 'model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')
   
    # Plot accuracy
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].set_title(model_name + " " +'model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')

    return fig, axs
