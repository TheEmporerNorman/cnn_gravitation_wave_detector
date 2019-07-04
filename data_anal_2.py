import numpy as np
import matplotlib.pyplot as plt

def plotModels(model_names, layer_names, line_colors, title):
	plt.figure()

	num_models = len(model_names)

	for i in range(num_models):
		filepath = "./models/" + model_names[i] + "/output.csv"
		accuracy, false_positives, missed_positives, wave_amp, wave_amp_sigma, avg_snr, avg_val_snr, snr_cuttoff = np.loadtxt(filepath, delimiter = ",", unpack = True)

		plt.plot(avg_val_snr, accuracy, "x", color = line_colors[i])
		plt.plot(np.unique(avg_val_snr), np.poly1d(np.polyfit(avg_val_snr, accuracy, 2))(np.unique(avg_val_snr)), label = layer_names[i], color = line_colors[i])

	plt.grid()
	plt.xlabel("Average SNR")
	plt.ylabel("Accuracy")
	plt.legend(loc = "best")
	plt.savefig(title, format = 'pdf')

model_names = ["2018-8-26_20:4:2","2018-8-26_22:34:37","2018-8-27_0:50:42", "2018-8-27_3:11:22", "2018-8-27_5:27:48","2018-8-27_15:39:53"]
layer_names = ["Chirplett","Static Burst","All Types","Gausian Burst","Ringdown","Single Sine"]
line_colors = ["red","green","blue","black", "purple", "orange"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_result_1.pdf")

model_names = ["2018-8-27_18:24:40","2018-8-27_21:3:34","2018-8-27_23:32:39","2018-8-28_1:57:12","2018-8-28_4:21:28","2018-8-28_13:57:10"]
layer_names = ["All Types","Chirplett","Gausian Burst","Ringdown","Single Sine","Static Burst"]
line_colors = ["red","green","blue","black","purple", "orange"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_result_2.pdf")

model_names = ["2018-8-27_18:24:40","2018-8-27_21:3:34","2018-8-27_23:32:39","2018-8-28_1:57:12","2018-8-28_4:21:28"]
layer_names = ["All Types","Chirplett","Gausian Burst","Ringdown","Single Sine"]
line_colors = ["red","green","blue","black","purple"]

plotModels(model_names, layer_names, line_colors, "accuracy_vs_snr_result_3.pdf")

