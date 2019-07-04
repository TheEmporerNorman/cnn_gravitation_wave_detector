void* caclConvFFT(double* f, double* g, double _Complex* conv, uint32_t len){
	double _Complex* f_freq = malloc(sizeof(double _Complex) * len);
	double _Complex* g_freq = malloc(sizeof(double _Complex) * len);

	calcFFT(f, f_freq, len);
	//calcFFT(g, g_freq, len); <-- window function fft worked out before 

	for (uint32_t freq_idx = 0; freq_idx < len; freq_idx++){
		conv = f_freq[time_idx]*g_freq[time_idx];
	}

	free(f_freq); free(g_freq);
}

void calcPSD(double _Complex* input, double _Complex* window_func_fft, uint32_t input_length, uint32_t num_segments, uint32_t seg_length, double** ret_psd, double** ret_freq_axis){
	

	//Seg length needs to be power of 2. Overlap inverval cannot be larger that seg_div

	if (checkPower2((size_t) seg_length) != 1) { seg_length = nearestPower2( (int) seg_length);
		printf("Seg length not power of 2, ajusting to nearest! Setting as %i. \n", seg_length);

	}

	uint32_t seg_div = (intput_length - 2*seg_length)/num_segments;
	double overlap_interval = (seg_length - seg_div)/2
	if ((overlap_interval > seg_div) || (num_segments > input_length)){ printf("Too many segments. Exiting. \n"); exit(2);}
	uint32_t overlap_length = overlap_interval * seg_div;
	double* freq_axis = malloc(sizeof(double)*seg_length);
	double* psd = calloc(seg_length, sizeof(double)*seg_length);

	for (uint32_t freq_idx = 0; time_idx < config.seg_length; time_idx++){
		freq_axis[time_idx] = time_idx*(config.stream_sample_rate/seg_length);
	}

	double _Complex* seg_conv = malloc(sizeof(double _Complex) * seg_len);

	for(uint32_t seg_idx = seg_len; seg_idx < num_segments; seg_idx++){
		calcConvFFT(input[seg_idx*seg_div - overlap_length], window_func_fft, seg_conv, seg_length);
		for (uint32_t freq_idx = 0; freq_idx < seg_length; freq_idx++){
				psd[time_idx] += (double) seg_conv[freq_idx]*seg_conv[freq_idx] 
		}

	}

	free(seg_conv);

	for (uint32_t freq_idx = 0; time_idx < seg_length; time_idx++){
		psd[time_idx] /= num_segments;
	}

	*ret_psd = psd;
	*ret_freq_axis = freq_axis; 

}

void interpArray(double* orig_arr_x, double* orig_arr_y, uint32_t orig_arr_len, double* interped_arr_x, double* interped_arr_y, uint32_t interped_arr_len){

		//Interpolation:
		for (size_t time_1_idx = 0; time_1_idx < interped_arr_len; time_1_idx++){
			for (size_t time_2_idx = 0; time_2_idx < orig_arr_len; time_2_idx++){
				if ((interped_arr_x[time_1_idx] >= orig_arr_x[time_2_idx]) && (interped_arr_x[time_1_idx] <= orig_arr_x[time_2_idx])){
					double grad = (orig_arr_x[time_2_idx] - orig_arr_x[time_2_idx + 1])/(orig_arr_y[time_2_idx] - orig_arr_y[time_2_idx + 1]);
					interped_arr_y[time_1_idx] = grad*(orig_arr_x - interped_arr_x) + orig_arr_y[time_2_idx +];
				}
			}

		}
}


void calcSNR(config_s config, double* strain_axis, double* strain_axis_sig, double* strain_axis_noise){
	
	double _Complex * strain_axis_c = malloc(sizeof(double _Complex) * config.stream_res);
	double _Complex * strain_axis_sig_c = = malloc(sizeof(double _Complex) * config.stream_res);

	double _Complex * strain_axis_cf = malloc(sizeof(double _Complex) * config.stream_res);

	double _Complex * freq_axis_c = malloc(sizeof(double) * config.stream_res);

	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		strain_axis_c[time_idx] = (double _Complex) strain_axis[time_idx];
		strain_axis_sig_c[time_idx] = (double _Complex) strain_axis_sig[time_idx];
		freq_axis_c[time_idx] = time_idx*(config.stream_sample_rate/config.stream_res);
	}

	calcFFT(strain_axis_c, strain_axis_cf, config.stream_res);
	calcFFT(strain_axis_sig_c, strain_axis_sig_cf, config.stream_res);

	free(strain_axis_c); free(strain_axis_sig_c);

	double* psd; double* freq_axis;

	uint32_t seg_length = 256;
	calcPSD(strain_axis_noise, config.window_func_fft, config.stream_res, floor(config.stream_res/4.0), seg_length, &ret_psd, &freq_axis); 

	double* interped_psd = malloc(sizeof(double) * config.stream_res);
	interpArray(freq_axis, psd, seg_length, freq_axis_c, interped_psd, config.stream_res);
	free(psd); free(freq_axis);

	double _Complex* optimal_cf = malloc(sizeof(double _Complex) * config.stream_res);
	
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		optimal_cf[time_idx] = strain_axis_cf[time_idx]*conj(strain_axis_sig_cf[time_idx]);
	}	

	free(strain_axis_cf);

	double _Complex* optimal_c = malloc(sizeof(double _Complex) * config.stream_res);
	calcIFFT(optimal_cf, optimal_c, config.stream_res);
	free(optimal_cf);
	double freq_step = fabs(freq_axis_c[1] - freq_axis_c[0]);

	free(freq_axis_c);

	double sigma = 0;

	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		optimal_c[time_idx] *= 2;
		sigma += ((2*strain_axis_sig_cf[time_idx]*conj(strain_axis_sig_cf[time_idx]))/interped_psd[time_idx])*freq_step;
	}	

	free(strain_axis_sig_cf); free(interped_psd);

	sigma = sqrt(fabs(sigma));
	double* snr = malloc(sizeof(double _Complex) * config.stream_res);

	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		snr[time_idx] = optimal_c[time_idx]/sigma;
	}	

}
