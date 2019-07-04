import numpy as np
import matplotlib.pyplot as plt

def plotModels(model_names, layer_names, line_colors, title):
	plt.figure()

	num_models = len(model_names)

	for i in range(num_models):
		filepath = "./models/" + model_names[i] + "/output.csv"
		accuracy, false_positives, missed_positives, wave_amp, wave_amp_sigma, avg_snr, snr_cuttoff = np.loadtxt(filepath, delimiter = ",", unpack = True)

		plt.plot(avg_snr, accuracy, "x", color = line_colors[i])
		plt.plot(np.unique(avg_snr), np.poly1d(np.polyfit(avg_snr, accuracy, 1))(np.unique(avg_snr)), label = layer_names[i], color = line_colors[i])

	plt.grid()
	plt.xlabel("Average SNR")
	plt.ylabel("Accuracy")
	plt.legend(loc = "best")
	plt.savefig(title, format = 'pdf')

model_names = ["2018-8-9_2:14:26","2018-8-9_16:51:54","2018-8-8_23:26:51", "2018-8-9_20:40:30", "2018-8-9_23:26:5"]
layer_names = ["2 layers","3 layers","4 layers","5 layers","6 layers"]
line_colors = ["red","green","blue","black", "purple"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_layers.pdf")

model_names = ["2018-8-10_18:3:36","2018-8-11_0:58:49","2018-8-11_3:25:6","2018-8-11_21:8:57", "2018-8-12_2:6:39", "2018-8-11_16:24:53"]
layer_names = ["16,32,64","32,32,64","32,32,32","64,32,32","64,32,16", "64,64,64"]
line_colors = ["red","green" ,"blue","black", "purple","orange"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_layer_size_x_slope.pdf")

model_names = ["2018-8-12_2:6:39", "2018-8-11_23:36:12", "2018-8-12_4:43:34", "2018-8-12_22:37:23","2018-8-13_1:10:46"]
layer_names = ["64,32,16",  "128,64,32", "32,16,8", "64,32,8","128,32,16"]
line_colors = ["purple","green" ,"blue","black", "red"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_layer_size_x_size.pdf")

model_names = ["2018-8-12_2:6:39", "2018-8-13_3:40:53"]
layer_names = ["2,2,1","3,1,1"]
line_colors = ["red","green"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_layer_size_y.pdf")

model_names = ["2018-8-13_6:26:21", "2018-8-13_8:51:41", "2018-8-13_11:12:32","2018-8-13_18:57:1","2018-8-13_21:21:3","2018-8-13_23:42:3","2018-8-14_15:45:14","2018-8-16_18:16:47"]
layer_names = ["8,8,8","16,8,8","16,16,8","32,16,8","8,8,16","8,16,16","8,16,32","32,16,8"]
line_colors = ["red","green","blue","black","purple","orange","brown","grey"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_filters_8.pdf")

model_names = ["2018-8-14_19:36:22","2018-8-15_10:16:41","2018-8-15_16:43:20","2018-8-14_8:35:29","2018-8-15_19:27:1","2018-8-15_22:48:11"]
layer_names = ["16,16,16","32,16,16","32,32,16","64,32,16","16,16,32","16,32,32"]
line_colors = ["red","green","blue","yellow","black","purple"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_filters_16.pdf")

model_names = ["2018-8-16_18:16:47","2018-8-17_15:40:20","2018-8-17_23:47:33"]
layer_names = ["32,16,8","16,8,4","8,4,2"]
line_colors = ["red","green","blue","yellow"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_filters_steep.pdf")

model_names = ["2018-8-18_18:50:32","2018-8-18_22:29:5","2018-8-19_0:50:48","2018-8-19_15:24:34","2018-8-19_20:2:25","2018-8-19_21:35:23","2018-8-19_23:8:24","2018-8-17_15:40:20"]
layer_names = ["0,0,0","1,0,0","0,1,0","0,0,1","1,1,0","1,0,1","0,1,1","1,1,1"]
line_colors = ["red","green","blue","yellow","black","purple","brown","grey"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_norm_presnent.pdf")

model_names = ["2018-8-20_1:9:13","2018-8-20_2:42:31","2018-8-19_21:35:23","2018-8-20_4:16:39","2018-8-20_5:50:15","2018-8-20_19:16:28",]
layer_names = ["0","32","64","128","256","512"]
line_colors = ["red","green","blue","black","orange","purple"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_dense_layers.pdf")

model_names = ["2018-8-20_20:50:9","2018-8-20_23:7:28","2018-8-21_1:30:16","2018-8-21_3:57:4"]
layer_names = ["512","0","1024","2048"]
line_colors = ["red","green","blue","yellow"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_dense_layers_50.pdf")

model_names = ["2018-8-21_15:45:50","2018-8-21_18:2:39","2018-8-21_20:46:35","2018-8-21_23:25:47", "2018-8-22_1:44:23"]
layer_names = ["0.1","0.01","0.001","0.0001","0.00001"]
line_colors = ["red","green","blue","black","orange"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_learning_rate.pdf")

model_names = ["2018-8-21_20:46:35","2018-8-21_23:25:47", "2018-8-22_4:5:58","2018-8-22_16:41:53","2018-8-22_19:3:16", "2018-8-22_23:55:15"]
layer_names = ["0.001","0.0001","0.0005","0.005","0.0025","0.00075"]
line_colors = ["red","green","blue","black","orange","yellow","grey","orange"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_learning_rate-spec.pdf")

model_names = ["2018-8-23_8:44:59","2018-8-22_23:55:15","2018-8-23_5:38:41" ,"2018-8-23_2:15:42"]
layer_names = ["25","50","75","100"]
line_colors = ["red","green","blue","yellow"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_num_epochs.pdf")

model_names = ["2018-8-22_23:55:15","2018-8-24_14:3:40","2018-8-23_18:50:36"]
layer_names = ["1,1,1","0,0,0","1,0,0"]
line_colors = ["red","green","blue"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_dropout_present.pdf")