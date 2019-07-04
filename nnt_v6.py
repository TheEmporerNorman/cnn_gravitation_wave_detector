import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

import pandas as pd
from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.metrics import binary_accuracy

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras_sequential_ascii import sequential_model_to_ascii_printout
from plot_losses import PlotLosses

# Keras layers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, Conv1D, MaxPool2D, Flatten
from keras.layers import Dropout, BatchNormalization

import time

import sys, string, os, getopt
import ctypes as ctypes

import gc

""" ~~~~~~~~~~ Setup ~~~~~~~~~~ """

class config_s():
	def __init__(self):

		self.num_dim = 3

		self.num_detects = 3
		self.num_streams = 1000
		self.stream_sample_rate = 16384
		self.stream_duration = 0.25

		self.num_epochs = 1

		self.stream_res = int(self.stream_sample_rate*self.stream_duration)

""" ~~~~~~~~~~ Setup ~~~~~~~~~~ """

def plotGraph(x_arr, y_arr, x_label, y_label, file_name):

	file_name = file_name.decode("ascii")
	y_label = y_label.decode("ascii")
	x_label = x_label.decode("ascii")

	plt.figure()

	plt.plot(x_arr, y_arr)
	
	plt.ylabel(y_label)
	plt.xlabel(x_label)

	plt.savefig(file_name + ".png")

	plt.close()


def plotSampleWaves(num_detects,  strain_axis_noise, time_axis, strain_axis, waves_present, num_samples, name, num, file_path):

  rows = num_samples
  fig, axs = plt.subplots(rows, 1, figsize=(8, 1.5 * rows))
  stream_idx = 0
  for i in range(rows):
    ax = axs[i]
    #stream_idx = np.random.randint(config["num_streams"])

    for detect_idx in range(num_detects):
      ax.plot(time_axis, strain_axis_noise[stream_idx][detect_idx])
              #cmap='Greys', interpolation='none')[detect_idx].time_data, batch[stream_idx].streams[detect_idx].amp_data)
    ax.set_title(str(waves_present[stream_idx]))
    
    for detect_idx in range(num_detects):
      ax.plot(time_axis, strain_axis[stream_idx][detect_idx])

    stream_idx += 1

  plt.savefig(file_path + "/example_waves_" + name + num + ".png")

  plt.close()

def plotAccuracyGraph(accuracy, false_positives, false_negatives, SNR, file_path):

	file_path = file_path.decode("ascii")

	plt.figure()

	SNR = np.asarray(SNR)
	SNR = np.divide(SNR,100)

	plt.plot(false_positives, label = "false_positives")
	plt.plot(false_negatives, label = "false_negatives")
	plt.plot(accuracy, label = "accuracy")
	plt.plot(SNR, label = "SNR/100")

	plt.legend(loc = "best")
	plt.title("Accuracy over time")
	plt.ylabel('Percent')
	plt.xlabel('generations')

	plt.savefig(file_path + "/accuracy_plot")

	plt.close()


def plotConfusionMatrix(cm, classes, file_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_path + "/" + title)

    plt.close()

def genPsuedoWaves(config, gen_waves_c):

	num_waves_present = np.zeros(config.num_streams, dtype=np.uint32)
	time_axis = np.zeros(config.stream_res, dtype=np.double)
	strain_axes = np.zeros((config.num_streams, config.num_detects, config.stream_res), dtype= np.float32)
	strain_axes_noise = np.zeros((config.num_streams, config.num_detects, config.stream_res), dtype= np.float32)

	gen_waves_c(ctypes.c_int(config.num_streams), ctypes.c_int(config.stream_sample_rate), ctypes.c_double(config.stream_durati-on), ctypes.c_void_p(num_waves_present.ctypes.data), ctypes.c_void_p(strain_axes.ctypes.data), ctypes.c_void_p(strain_axes_noise.ctypes.data),ctypes.c_void_p(time_axis.ctypes.data))
	plotSampleWaves(config.num_detects, strain_axes_noise, time_axis, strain_axes, num_waves_present, 10, "1")

	print(strain_axes_noise.shape)

	# channel for X
	strain_axes_noise = strain_axes_noise[..., np.newaxis].swapaxes(1,2)
	strain_axes = strain_axes[..., np.newaxis].swapaxes(1,2)

	print(strain_axes_noise.shape)
	num_waves_present = np_utils.to_categorical(num_waves_present, 2)

	# splitting data into training and test sets
	strain_axes_noise_train, strain_axes_noise_test, strain_axes_train, strain_axes_test = train_test_split(strain_axes_noise, strain_axes, test_size=0.20, random_state=12, shuffle = False)
	num_waves_present_train, num_waves_present_test, strain_axes_train, strain_axes_test = train_test_split(num_waves_present, strain_axes, test_size=0.20, random_state=12, shuffle = False)

	print(strain_axes_noise.shape,strain_axes.shape)
	print(strain_axes_noise_train.shape, strain_axes_noise_test.shape, strain_axes_train.shape, strain_axes_test.shape)

	del strain_axes
	del strain_axes_noise

	return strain_axes_noise_train, strain_axes_train, strain_axes_noise_test, strain_axes_test, num_waves_present_test


def main(num_streams_train, num_streams_test, num_detects, stream_res, gen_idx, num_epochs, num_waves_present_train, strain_axes_train, strain_axes_noise_train, num_waves_present_test, strain_axes_test, strain_axes_noise_test, time_axis, file_path): #num_waves_present
	
	file_path = file_path.decode("ascii")
	config = config_s() #<-- Creates config object

	strain_axes_train  = np.asarray(strain_axes_train, dtype= np.float32)
	strain_axes_train = np.reshape(strain_axes_train, [num_streams_train, num_detects, stream_res], order = "C" )
	strain_axes_noise_train = np.asarray(strain_axes_noise_train, dtype= np.float32)
	strain_axes_noise_train = np.reshape(strain_axes_noise_train, [num_streams_train, num_detects, stream_res], order = "C" )
	num_waves_present_train = np.asarray(num_waves_present_train, dtype= np.int32)

	strain_axes_test  = np.asarray(strain_axes_test, dtype= np.float32)
	strain_axes_test = np.reshape(strain_axes_test, [num_streams_test, num_detects, stream_res], order = "C" )
	strain_axes_noise_test = np.asarray(strain_axes_noise_test, dtype= np.float32)
	strain_axes_noise_test = np.reshape(strain_axes_noise_test, [num_streams_test, num_detects, stream_res], order = "C" )
	num_waves_present_test = np.asarray(num_waves_present_test, dtype= np.int32)

	plotSampleWaves(config.num_detects, strain_axes_noise_train, time_axis, strain_axes_train, num_waves_present_train, 10, "train", str(gen_idx), file_path)
	plotSampleWaves(config.num_detects, strain_axes_noise_test, time_axis, strain_axes_test, num_waves_present_test, 10, "validate", str(gen_idx), file_path)

	# channel for X
	strain_axes_noise_train= strain_axes_noise_train[..., np.newaxis].swapaxes(1,2)
	strain_axes_train = strain_axes_train[..., np.newaxis].swapaxes(1,2)

	strain_axes_noise_test = strain_axes_noise_test[..., np.newaxis].swapaxes(1,2)
	strain_axes_test = strain_axes_test[..., np.newaxis].swapaxes(1,2)

	print(strain_axes_noise_train.shape)
	num_waves_present_train = np_utils.to_categorical(num_waves_present_train, 2)
	num_waves_present_test = np_utils.to_categorical(num_waves_present_test, 2)

	print(strain_axes_noise_train.shape, strain_axes_noise_test.shape, strain_axes_train.shape, strain_axes_test.shape)

	model = load_model(file_path + "/model_1")
	print(int(np.floor(len(strain_axes_noise_train)/20)))

	model.fit(strain_axes_noise_train, num_waves_present_train,
           epochs= num_epochs,
           batch_size= 32, 
           validation_data =(strain_axes_noise_test, num_waves_present_test),
           callbacks=[],
           shuffle = True)

	model_pred = model.predict(strain_axes_noise_test)

	cnf_matrix = confusion_matrix(num_waves_present_test.argmax(axis=1), model_pred.argmax(axis=1))
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix

	class_names = ["Wave Absent","Waves Present"]
	plotConfusionMatrix(cnf_matrix, class_names, file_path, title = 'conf_mtrx_' + str(gen_idx));
	print("Complete!")

	cnf_matrix = cnf_matrix.astype(np.float32)
	cnf_matrix = cnf_matrix.flatten()

	return cnf_matrix.tolist(); 

def constructModel(stream_res, num_detects, num_classes, model_plan, file_path):

	file_path = file_path.decode("ascii")

	#Tells tensor flow I'm only using the GPU:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
     
	#Tells tensor flow not to allocate all posisble GPU memory to save some for the display:
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	print("Constructing model...") 

	num_conv_layers = int(model_plan[0]);

	if (num_conv_layers < 1):
		print("There must be at least one convoloutional layer, exiting.")
		exit(1);

	conv_kern_sizes_x = np.asarray(model_plan[1: num_conv_layers + 1], dtype = np.int32);
	conv_kern_sizes_y = np.asarray(model_plan[num_conv_layers + 1 : 2*num_conv_layers + 1], dtype = np.int32);
	conv_num_filters = np.asarray(model_plan[2*num_conv_layers + 1 : 3*num_conv_layers + 1], dtype = np.int32);
	conv_batch_norm_present = np.asarray(model_plan[3*num_conv_layers + 1 : 4*num_conv_layers + 1], dtype = np.int32);
	conv_dropout_present = np.asarray(model_plan[4*num_conv_layers + 1 : 5*num_conv_layers + 1], dtype = np.int32);
	kern_dropouts = np.asarray(model_plan[5*num_conv_layers + 1 : 6*num_conv_layers + 1], dtype = np.float);

	conv_layer_end = 6*num_conv_layers;

	num_dense_layers = int(model_plan[conv_layer_end + 1])
	dense_num_outputs = np.asarray(model_plan[conv_layer_end + 2: conv_layer_end + 2 + num_dense_layers], dtype = np.int32)
	dense_dropouts_present = np.asarray(model_plan[conv_layer_end + 2 + num_dense_layers: conv_layer_end + 2 + 2*num_dense_layers], dtype = np.int32)
	dense_dropouts = np.asarray(model_plan[conv_layer_end + 2 + 2*num_dense_layers: conv_layer_end + 2 + 3*num_dense_layers], dtype = float)

	learning_rate = model_plan[conv_layer_end+ 2 + 3*num_dense_layers]

	model = Sequential()

	for conv_idx in range(num_conv_layers):
		if (conv_idx == 0):
			model.add(Conv2D(filters=conv_num_filters[0], kernel_size=(conv_kern_sizes_x[0],conv_kern_sizes_y[0]), activation='relu', input_shape=(stream_res, num_detects, 1)))
		else:
			model.add(Conv2D(filters=conv_num_filters[conv_idx], kernel_size= (conv_kern_sizes_x[conv_idx],conv_kern_sizes_y[conv_idx]), activation='relu'))

		if (conv_batch_norm_present[conv_idx] == 1):
			model.add(BatchNormalization())

		if (conv_dropout_present[conv_idx] == 1): 
			model.add(Dropout(kern_dropouts[conv_idx]))

	model.add(Flatten())

	for dense_idx in range(num_dense_layers):
		model.add(Dense(dense_num_outputs[dense_idx], activation='softmax'))
		if (dense_dropouts_present[dense_idx] == 1): 
			model.add(Dropout(dense_dropouts[dense_idx]))

	model.add(Dense(num_classes, activation='softmax'))

	# compile the model
	opt = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
	model.compile(loss='categorical_crossentropy',optimizer=opt)

	model.save(file_path + "/model_1")

	sequential_model_to_ascii_printout(model)



if __name__ == "__main__": 
   main(sys.argv[1:])

   print("Finished.")

"""
To do: add power of 2 check.
"""
