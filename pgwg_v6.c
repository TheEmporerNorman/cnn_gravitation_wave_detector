#include </usr/include/python3.5m/Python.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <limits.h>
#include <immintrin.h>
#include <complex.h> 
#include <stdlib.h>
#include <omp.h>

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

Pseudo Gravtiational Wave Generator and Multi-Detector Neural Network Trainer v5.

The following program was written in June-August 2018 as part of an MSc Project for Cardiff University School of Physics
and Astronomy. 

The program requires a number of configuration files which should be provided with this file, including:
detect_config.csv,
model_config.csv,
pseudo_grav_config.csv,
as well as detector noise profiles, stored within a folder named detect_noise_profiles, with the folowing nameing format:
DETECT_NAME_HERE_noise_profile.csv
the names of the detectors used to name the files should be entered into pseudo_grav_config.csv. These files can be altered
depenant on user need though it should be noted that the program has not been thougrally tested and is not equipped with much
error handeling so some configurations will probabably cause errors.

Contact Information:

Program Author: Michael Norman BSc michael@norvett.com
Project Supervisor: Professor Patrick Sutton patrick.sutton@astro.cf.ac.uk

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
*/  

#define RAND_64_MAX ~(0ULL) //<-- Used in random number generation process. (NOTE: May not function correctly on older Intel CPUs!)

typedef struct config_s {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Struture to hold general program configuration data.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	//Runtime information:
	uint16_t  bar_len, prg_chck_int; clock_t str_t, end_t; //<-- Time variables used to estimate program run time.
	uint16_t  vect_size_db, vect_size_16, vect_size_32; //<-- Quality of life variables that hold commonly used values.

	//Universal configuration:
	uint16_t num_dim; //<-- Number of dimensions in which vector calculations are performed (DO NOT CHANGE, PROGRAM WILL NOT RUN!).
	double equat_rad,  pole_rad; //<-- Earth's (Or wherever) equatorial and pole radius used to calculate detector coordinates.
	double* origin; //<-- Quality of life variable holding the origin vector.
	double* time_axis; //<-- Holds stream time axis which will remain constant throughout program operaton.
	double _Complex* window_func;
	double _Complex* window_func_fft; //<-- Holds windowing function required for SNR calculation.
	uint32_t seg_length;

	//Stream configuration:
	uint32_t num_streams, stream_sample_rate, stream_res; //<-- Information about stream length and number of streams.
	double stream_duration; 

	//Wave configuration:
	uint32_t num_waves;
	uint16_t num_wave_types;
	bool gen_waves; 
	double waves_present_mu, waves_present_sigma;
	uint16_t wave_present_min, wave_present_max;
	double wave_amp_mu, wave_amp_sigma, wave_amp_min, wave_amp_max;
	double wave_tau_mu, wave_tau_sigma, wave_tau_min, wave_tau_max;
	double wave_f0_mu, wave_f0_sigma, wave_f0_min, wave_f0_max;
	double wave_alpha_mu, wave_alpha_sigma, wave_alpha_min, wave_alpha_max;
	double wave_centre_time_min, wave_centre_time_max, wave_speed;

	//Detector configuration:
	uint16_t num_detects; 
	
	//Neural network configuration:
	uint16_t req_detects;
	double snr_cutoff;	
	double snr_min, snr_max, avg_snr;
	double noise_ajust;
	uint16_t num_gens;
	uint16_t num_epocs;
	uint16_t num_classes;
	double amp_min, amp_max, amp_step; 
	double snr_cutoff_max, snr_cutoff_min,snr_cutoff_step;
	double amp_sigma_max, amp_sigma_min, amp_sigma_step;

	//Glitch configuration:
	uint32_t num_glitches;
	uint16_t num_glitch_types;
	bool gen_glitches;
	double glitches_present_mu, glitches_present_sigma;
	uint16_t glitches_present_min, glitches_present_max;
	double glitch_amp_mu, glitch_amp_sigma, glitch_amp_min, glitch_amp_max;
	double glitch_tau_mu, glitch_tau_sigma, glitch_tau_min, glitch_tau_max;
	double glitch_f0_mu, glitch_f0_sigma, glitch_f0_min, glitch_f0_max;
	double glitch_alpha_mu, glitch_alpha_sigma, glitch_alpha_min, glitch_alpha_max;
	double glitch_time_min, glitch_time_max;

	//Gaussian noise configuration:
	bool gen_noise;
	double noise_amp_mu, noise_amp_sigma, noise_amp_min, noise_amp_max;

	//File structure:
	char* file_path; 

	
}config_s;

typedef struct wave_s {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Struture to hold variables used for pseudo-waveform generation.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	double amp, centre_time;
	double* direct_vect; double* polar_vect; 	
	double tau, f0, alpha, delta;
	uint16_t type; 

}wave_s;

typedef struct glitch_s{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Struture to hold variables used for glitch generation.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	double amp, mu;
	double* strain_axis_glitch;
	double tau, f0, alpha, delta;
	uint16_t type; 

}glitch_s;

typedef struct detect_s {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Struture to hold variables about simulated GWO detectors. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	char* name;

	double sensitivity;

	uint16_t latt_deg, latt_min, latt_sec;
	uint16_t long_deg, long_min, long_sec;
	double hasl;

	uint32_t x_arm_bear, y_arm_bear;

	double* pos; double* pos_sphere;

	double* noise_profile; double* noise_profile_interp; double noise_amp; 

	double* x_arm_direct_vect; double* y_arm_direct_vect; double* up_direct;

}detect_s;

typedef struct stream_s {
	
	uint16_t num_waves_present;
	wave_s* waves;

	uint16_t* num_glitches_present;
	glitch_s* glitches;

	double* centre_time_disps;
	double* plus_polar;
	double* ciota;
	double* wave_resps; 

	double* strain_axes;
	double* strain_axes_noise;

	double* snr;

}stream_s;

typedef struct model_s {

	double* accuracy;
	double* missed_positives;
	double* false_positives;
	double* wave_amp;  
	double* avg_snr;
	double* wave_amp_sigma;
	double* snr_cutoff;

	uint16_t num_conv_layers;
	
	uint32_t* conv_kern_sizes_x;
	uint32_t*  conv_kern_sizes_y;
	uint16_t* conv_num_filters;
	bool* conv_batch_norm_present;
	bool* conv_dropout_present;
	double* conv_dropouts;

	uint16_t num_dense_layers;
	uint32_t* dense_num_outputs;
	bool* dense_dropouts_present;
	double* dense_dropouts;

	double learning_rate;

}model_s;

typedef struct loading_s {
	clock_t srt_t, end_t;
	double loop_t, total_t, avg_t;
	size_t chck_idx;
	double time_to_cmplt; 

}loading_s;

void printProgress(size_t count, double time_to_cmplt, size_t max, config_s config)
{
	double percent = ((double)count/(double)max);
	count = floor(percent*config.bar_len);
	max = config.bar_len; 

	char prefix[] = "Progress: [";
	char suffix[] = "]";
	size_t prefix_length = sizeof(prefix) - 1;
	size_t suffix_length = sizeof(suffix) - 1;
	char *buffer = calloc(max + prefix_length + suffix_length + 1, 1); // +1 for \0
	size_t i = 0;

	strcpy(buffer, prefix);
	for (; i < max; ++i){
		buffer[prefix_length + i] = i < count ? '#' : ' ';
	}

	strcpy(&buffer[prefix_length + i], suffix);
	printf(" %.2f%%. ", percent*100);

	int n,day,hr,min,sec;
	n = (size_t)time_to_cmplt;

	if(n>86400){;min =n/60; sec = n%60; hr = min/60;  min = min%60; day = hr/24; hr = day%24;
		printf("ETC: %d:%d:%d:%d days",day,hr,min,sec);
	}
	else if(n>3600){min =n/60; sec = n%60; hr = min/60; min = min%60;
		printf("ETC: %d:%d:%d hours",hr,min,sec);
	}
	else if(n>60){
		min = n/60; sec = n%60;
		printf("ETC: %d:%d mins",min,sec);
	}
	else{
		printf("ETC: %d secs", n);
	}

	printf("\b%c\r%s", 5, buffer);

	fflush(stdout);
	free(buffer);

}

char* split(char* delim){ return strtok(NULL, delim);}	
void get_argf(double *val, char* delim){*val = (double) atof(split(delim));}
void get_args(char **val, char* delim){*val = split(delim);}
void get_arg16(uint16_t *val, char* delim){*val = atoi(split(delim));}
void get_arg32(uint32_t *val, char* delim){*val = atoi(split(delim));}
void get_argb(bool* val, char* delim){*val = (bool) atoi(split(delim));}

void lineSkip(FILE* f, char* buffer, size_t size){
		buffer = NULL;
		size = getline(&buffer, &size, f);
		strtok(buffer, ",");
		buffer = NULL;
}

void startSkip(FILE* f, char* buffer, size_t size){
		buffer = NULL;
		strtok(buffer, ",");
		buffer = NULL;
}

void readInputArgs(int argc, char** argv, config_s config, config_s* ret_config)
{
	int c;
	opterr = 0;

	while ((c = getopt(argc, argv, "n:s:e:g:")) != -1)
	{
		switch (c)
		{
			case 'n': config.noise_ajust = atof(optarg); break;
			case 's': config.num_streams = atoi(optarg); break;
			case 'e': config.num_epocs = atoi(optarg); break;
			case 'g': config.num_gens = atoi(optarg); break;
			
		}
	}	

	*ret_config = config;
	
}

void readConfig(char* filename, config_s* config)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads data from config file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	printf("Loading config file: %s. \n", filename); 
	FILE* f;

	f = fopen(filename, "r"); 

	if (f == NULL){ printf("Warning! Could not find config file: \"%s\"! Using defaults.\n", filename); return; }
	else {
		char* buffer = NULL;
		size_t size; 

		lineSkip(f, buffer, size); get_arg16(&config->bar_len, ",");
		size = getline(&buffer, &size, f);	
		lineSkip(f, buffer, size); get_arg16(&config->num_dim, ",");
		lineSkip(f, buffer, size); get_argf(&config->equat_rad, ",");
		lineSkip(f, buffer, size); get_argf(&config->pole_rad, ",");
		size = getline(&buffer, &size, f);
		lineSkip(f, buffer, size); get_arg32(&config->num_streams, ",");
		lineSkip(f, buffer, size); get_arg32(&config->stream_sample_rate, ",");
		lineSkip(f, buffer, size); get_argf(&config->stream_duration, ",");

		size = getline(&buffer, &size, f);
		lineSkip(f, buffer, size); get_argb(&config->gen_waves, ","); 
		startSkip(f, buffer, size); get_argb(&config->gen_glitches, ",");
		startSkip(f, buffer, size); get_argb(&config->gen_noise, ",");

		lineSkip(f, buffer, size); get_argf(&config->waves_present_mu, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitches_present_mu, ",");

		lineSkip(f, buffer, size); get_argf(&config->waves_present_sigma, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitches_present_sigma, ",");

		lineSkip(f, buffer, size); get_arg16(&config->wave_present_min, ",");
		startSkip(f, buffer, size);  get_arg16(&config->glitches_present_min, ",");

		lineSkip(f, buffer, size); get_arg16(&config->wave_present_max, ",");
		startSkip(f, buffer, size);  get_arg16(&config->glitches_present_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_amp_mu, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_amp_mu, ",");
		startSkip(f, buffer, size);  get_argf(&config->noise_amp_mu, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_amp_sigma, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_amp_sigma, ",");
		startSkip(f, buffer, size); get_argf(&config->noise_amp_sigma, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_amp_min, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_amp_min, ",");
		startSkip(f, buffer, size);  get_argf(&config->noise_amp_min, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_amp_max, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_amp_max, ",");
		startSkip(f, buffer, size);  get_argf(&config->noise_amp_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_tau_mu, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_tau_mu, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_tau_sigma, ",");
		startSkip(f, buffer, size); ; get_argf(&config->glitch_tau_sigma, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_tau_min, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_tau_min, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_tau_max, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_tau_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_f0_mu, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_f0_mu, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_f0_sigma, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_f0_sigma, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_f0_min, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_f0_min, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_f0_max, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_f0_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_alpha_mu, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_alpha_mu, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_alpha_sigma, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_alpha_sigma, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_alpha_min, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_alpha_min, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_alpha_max, ",");
		startSkip(f, buffer, size);  get_argf(&config->glitch_alpha_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_centre_time_min, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_time_min, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_centre_time_max, ",");
		startSkip(f, buffer, size); get_argf(&config->glitch_time_max, ",");

		lineSkip(f, buffer, size); get_argf(&config->wave_speed, ",");
		size = getline(&buffer, &size, f);
		lineSkip(f, buffer, size); get_arg16(&config->num_detects, ",");
		size = getline(&buffer, &size, f);
		lineSkip(f, buffer, size); get_argf(&config->snr_cutoff, ",");
		lineSkip(f, buffer, size); get_arg16(&config->req_detects, ",");

		fflush(stdout);
	}
	fclose(f);

}

void readModel(char* filename, config_s* config, model_s* model)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads data from config file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	printf("Loading model file: %s. \n", filename); 
	FILE* f;

	f = fopen(filename, "r"); 

	if (f == NULL){ printf("Warning! Could not find config file: \"%s\"! Using defaults.\n", filename); return; }
	else {
		char* buffer = NULL;
		size_t size; 

		lineSkip(f, buffer, size); get_arg16(&model->num_conv_layers, ",");
		size = getline(&buffer, &size, f);	
		model->conv_kern_sizes_x = malloc(sizeof(uint32_t)*model->num_conv_layers); model->conv_kern_sizes_y = malloc(sizeof(uint32_t)*model->num_conv_layers);
		model->conv_num_filters = malloc(sizeof(uint16_t)*model->num_conv_layers); 
		model->conv_batch_norm_present = malloc(sizeof(bool)*model->num_conv_layers); model->conv_dropout_present = malloc(sizeof(bool)*model->num_conv_layers);
		model->conv_dropouts = malloc(sizeof(double)*model->num_conv_layers);

		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_arg32(&model->conv_kern_sizes_x[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_arg32(&model->conv_kern_sizes_y[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_arg16(&model->conv_num_filters[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_argb(&model->conv_batch_norm_present[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_argb(&model->conv_dropout_present[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_conv_layers; conv_idx++){ get_argf(&model->conv_dropouts[conv_idx], ","); }
		size = getline(&buffer, &size, f);

		lineSkip(f, buffer, size); get_arg16(&model->num_dense_layers, ",");
		model->dense_num_outputs = malloc(sizeof(uint32_t)*model->num_dense_layers); model->dense_dropouts_present = malloc(sizeof(bool)*model->num_dense_layers);
		model->dense_dropouts = malloc(sizeof(double)*model->num_dense_layers); 

		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_dense_layers; conv_idx++){ get_arg32(&model->dense_num_outputs[conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_dense_layers; conv_idx++){ get_argb(&model->dense_dropouts_present [conv_idx], ","); }
		lineSkip(f, buffer, size); for (size_t conv_idx = 0; conv_idx < model->num_dense_layers; conv_idx++){ get_argf(&model->dense_dropouts [conv_idx], ","); }
		
		size = getline(&buffer, &size, f);
		lineSkip(f, buffer, size); get_argf(&model->learning_rate, ",");

		fflush(stdout);
	}
	fclose(f);

}

void readFileDouble(char* filename, char* delimeter, size_t num_cols, size_t srt_line, size_t srt_col, double** data_ret, size_t* lines_read)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads data from input file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	size_t init_num_lines = 100;
	size_t curr_num_lines = init_num_lines;

	double* data = malloc(sizeof(double)*num_cols*init_num_lines);

	char* buffer = NULL; size_t size;

	FILE* file = fopen(filename, "r");

	if (file == NULL)
	{
		printf("Could not find input file! \"%s\"\n", filename);
		return;
	}

	//Skips to start line:
	for (size_t line_idx = 0; line_idx < srt_line; ++line_idx) { size = getline(&buffer, &size, file); }

	size_t line_idx = 0;

	while (getline(&buffer, &size, file) != EOF)
	{
		if(line_idx >= curr_num_lines)
		{
			data = realloc(data, sizeof(double)*(curr_num_lines + init_num_lines)*num_cols);
			curr_num_lines += init_num_lines;
		}

		char* first = strtok(buffer, delimeter);
		data[line_idx*num_cols + 0] = atof(first);

		//Skips to start column
		double empty = 0;
		for (size_t col_idx = 1; col_idx < srt_col; ++col_idx) { get_argf(&empty, delimeter); }

		if (srt_col == 0) { for (size_t col_idx = 1; col_idx < num_cols; ++col_idx) { get_argf(&data[line_idx*num_cols + col_idx], delimeter); } }
		else              { for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) { get_argf(&data[line_idx*num_cols + col_idx], delimeter); } } 

		line_idx++;
	}

	printf("%zu\n", line_idx );

	*lines_read = line_idx;
	*data_ret = data;

	fclose(file);
}

void readDetectConfig(char* filename, config_s config, detect_s* detects)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads detector data from config file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	FILE* f = fopen(filename, "r");

	if (f == NULL){ printf("Warning! Could not find detector config file: \"%s\"! Program Exiting.\n", filename); exit(1); }
	else {

		printf("Loading detector config file... \n");

		char* buffer = NULL;
		size_t size; 

		size = getline(&buffer, &size, f);	

		for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++) {

			buffer = NULL;
			detects[detect_idx].name = malloc(sizeof(char)*10);

			size = getline(&buffer, &size, f);	

			detects[detect_idx].name = strtok(buffer, ","); //<-- Hopefully add dynamic memory allocation at some point.
			get_argf(&detects[detect_idx].sensitivity, ",");
			get_argf(&detects[detect_idx].noise_amp, ",");
			get_arg16(&detects[detect_idx].latt_deg, ","); get_arg16(&detects[detect_idx].latt_min, ","); get_arg16(&detects[detect_idx].latt_sec, ",");
			get_arg16(&detects[detect_idx].long_deg, ","); get_arg16(&detects[detect_idx].long_min, ","); get_arg16(&detects[detect_idx].long_sec, ",");
			get_argf(&detects[detect_idx].hasl, ",");
			get_arg32(&detects[detect_idx].x_arm_bear, ","); get_arg32(&detects[detect_idx].y_arm_bear, ",");
	
		}

		fflush(stdout);

		printf("Detector config loaded.\n");

	}

	fclose(f);
}

bool checkPower2(size_t x) { 
	return x && !(x & (x - 1));
} //Checks if input is power of 2.

int nearestPower2(int x) {
	return pow(2, floor(log(x)/log(2)));
} //Returns the nearest power of 2.

double convLatttoSphere(double latt)
{ 
	return latt - 90.0; 
}

double convDMStoDeg(uint16_t degs, uint16_t mins, uint16_t sec)
{ 
	return (double) degs + (double) mins*(1.0/60.0) + (double) sec*(1.0/3600.0); 
}

double convDegtoRad(double degs){ 
	return degs*((2*M_PI)/360.); 
}

double* genLinArray(double min, double max, size_t num_steps)
{
	
	double step_size = (max - min)/num_steps;
	double* lin_arr = malloc(sizeof(double)*num_steps);

	#pragma omp parallel
	for (size_t step_idx = 0; step_idx < num_steps; step_idx++){
		lin_arr[step_idx] = min + step_idx*step_size;
	}

	return lin_arr;
}

double calcSqrInt(double* arr, size_t arr_len)
{
	double arr_squared_sum = 0;
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
		arr_squared_sum += arr[arr_idx]*arr[arr_idx];
		//printf("%f\n", arr_squared_sum);
	}
	return sqrt(arr_squared_sum);
}

uint32_t calcbArrSum(bool* arr, size_t arr_len)
{
	uint32_t arr_sum = 0;
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
		arr_sum += arr[arr_idx];
	}
	return arr_sum;
}

uint32_t calcArrSum(uint16_t* arr, size_t arr_len)
{
	uint32_t arr_sum = 0;
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
		arr_sum += arr[arr_idx];
	}
	return arr_sum;
}

double findArrMax(double* arr, double arr_len)
{
	double max = 0.0;

    for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
        max = (fabs(arr[arr_idx]) > max) ? fabs(arr[arr_idx]) : max;
    }

    return max;
}

void normaliseArray(double* arr, size_t arr_len, double* normed_arr)
{
	double max = findArrMax(arr, arr_len);
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++ ){
		normed_arr[arr_idx] = arr[arr_idx]/max;
	}
}


double calcVectMag(config_s config, double* vect)
{

	double mag = 0.0;
	for (size_t dim_idx = 0; dim_idx < config.num_dim; dim_idx++) { mag += vect[dim_idx]*vect[dim_idx]; }

	return sqrt(mag);
}

double* calcUnitVect(config_s config, double* vect)
{

	double* unit_vect = malloc(config.vect_size_db);
	double mag = calcVectMag(config, vect);

	for (size_t dim_idx = 0; dim_idx < config.num_dim; ++dim_idx) { unit_vect[dim_idx] = vect[dim_idx] / mag; }

	return unit_vect;
}

double calcVectDot(config_s config, double* vect1, double* vect2)
{

	double dot = 0; 

	for (size_t dim_idx = 0; dim_idx < config.num_dim; dim_idx++) { dot += vect1[dim_idx]*vect2[dim_idx]; }
	
	return dot;

}

double* calcVectProjectPlane(config_s config, double* vect, double* normal_vect)
{

	double mag = calcVectMag(config, normal_vect); double dot = calcVectDot(config, vect, normal_vect);
	double* plane_vect = malloc(config.vect_size_db);

	for (size_t dim_idx = 0; dim_idx < config.num_dim; dim_idx++) { plane_vect[dim_idx] = vect[dim_idx] - (dot/(mag*mag))*normal_vect[dim_idx]; }

	return plane_vect;
}

double calcSpheroidRad(double latt, double equat_rad, double pole_rad)
{

	double cos_latt = cos(latt); double sin_latt = sin(latt);
	double var_1 = ( equat_rad * equat_rad * cos_latt ); double var_2 = ( pole_rad * pole_rad * sin_latt );
	double var_3 = ( equat_rad * cos_latt ); double var_4 = (pole_rad * sin_latt );

	return sqrt((var_1*var_1 + var_2*var_2)/(var_3*var_3 + var_4*var_4));

}

double* convSpheretoCart(config_s config, double* pos_sphere)
{

	double* pos = malloc(config.vect_size_db);
	double sin_theta = sin(pos_sphere[1]); 

	pos[0] = pos_sphere[0]*sin_theta*cos(pos_sphere[2]);
	pos[1] = pos_sphere[0]*sin_theta*sin(pos_sphere[2]);
	pos[2] = pos_sphere[0]*cos(pos_sphere[1]);

	return pos;

}

double* calcRadUnitVect(config_s config, double* pos_sphere)
{

	double* unit_vect = malloc(config.vect_size_db);
	double sin_theta = sin(pos_sphere[1]); 
	
	unit_vect[0] = sin_theta*cos(pos_sphere[2]);
	unit_vect[1] = sin_theta*sin(pos_sphere[2]);
	unit_vect[2] = cos(pos_sphere[1]);

	return unit_vect;

}

double* calcThetaUnitVect(config_s config, double* pos_sphere)
{

	double* unit_vect = malloc(config.vect_size_db);
	double cos_theta = sin(pos_sphere[1]); 
	
	unit_vect[0] = cos_theta*cos(pos_sphere[2]);
	unit_vect[1] = cos_theta*sin(pos_sphere[2]);
	unit_vect[2] = -sin(pos_sphere[1]);

	return unit_vect;

}

double* calcPhiUnitVect(config_s config, double* pos_sphere)
{

	double* unit_vect = malloc(config.vect_size_db);
	
	unit_vect[0] = -sin(pos_sphere[2]);
	unit_vect[1] = cos(pos_sphere[2]);
	unit_vect[2] = 0.0;

	return unit_vect;

}

double* calcRotatedVector(config_s config, double* init_vector, double* centre, double* rotation, double angle)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Returns (fin_vector) the postion of a 3d vector after rotation (rotation) around a point (centre). 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	//Assigning values to variables for ease of use:

	double* fin_vector = malloc(config.vect_size_db);

	double x = init_vector[0], y = init_vector[1], z = init_vector[2];
	double a = centre[0],      b = centre[1],      c = centre[2];
	double u = rotation[0],    v = rotation[1],    w = rotation[2];

	double cos_angle = cos((2.*M_PI/360.) * angle); double sin_angle = sin((2.*M_PI/360.) * angle);

	//Calculating rotation:

	fin_vector[0] = (a*(v*v + w*w) - u*(b*v + c*w - u*x - v*y - w*z))*(1-cos_angle) + x*cos_angle + (- c*v + b*w - w*y + v*z)*sin_angle;
	fin_vector[1] = (b*(u*u + w*w) - v*(a*u + c*w - u*x - v*y - w*z))*(1-cos_angle) + y*cos_angle + (  c*u - a*w + w*x - u*z)*sin_angle;
	fin_vector[2] = (c*(u*u + v*v) - w*(a*u + b*v - u*x - v*y - w*z))*(1-cos_angle) + z*cos_angle + (- b*u + a*v - v*x + u*y)*sin_angle;

	return fin_vector;
}

uint64_t rand_64()
{

    unsigned long long val;

    while(!_rdrand64_step(&val));

    return (uint64_t)val;

}

double genRandBetween(double min, double max)
{
	return ( (double)rand_64()/(double)RAND_64_MAX ) * (max - min) + min;

}

void genRandArray(double min, double max, size_t arr_len, double* arr)
{
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
		arr[arr_idx] = genRandBetween(min, max);
	}

}

void genRandUIntArray(double min, double max, size_t arr_len, uint16_t* arr)
{
	for (size_t arr_idx = 0; arr_idx < arr_len; arr_idx++){
		arr[arr_idx] = floor(genRandBetween(min, max + 1));
	}
}

void genRandGaussArray(double mu, double sigma, size_t arr_len, double* arr){

	if (arr_len == 0) {return;}
	else if (arr_len == 1) {
		
		double u1 = genRandBetween(0.0, 1.0);
		double u2 = genRandBetween(0.0, 1.0);

		arr[0] = sqrt( -2*log(u1))*cos(2*M_PI*u2);
		arr[0] = arr[0]*sigma + mu;

		return;
	}
	else {

		size_t ceil_c = ceil( (double) arr_len / 2.0);

		double* u1 = malloc(sizeof(double)*ceil_c); genRandArray(0.0, 1.0, ceil_c, u1);
		double* u2 = malloc(sizeof(double)*ceil_c); genRandArray(0.0, 1.0, ceil_c, u2);
		uint16_t fin_idx = 0;

		#pragma omp parallel
		for(size_t arr_idx = 0; arr_idx < floor(arr_len/2); arr_idx++){
			double sqrtlg = sqrt(-2*log(u1[arr_idx])); double pi_b = 2*M_PI*u2[arr_idx];
			arr[2*arr_idx] = sqrtlg*cos(pi_b)*sigma + mu;
			arr[2*arr_idx + 1] = sqrtlg*sin(pi_b)*sigma + mu;
			fin_idx = arr_idx;
		}

		if ((fin_idx*2 + 1) == (arr_len - 2)){ arr[arr_len - 1] = (sqrt(-2*log(u1[ceil_c - 1]))*sin(2*M_PI*u2[ceil_c - 1])) * sigma + mu; } 

		free(u1); free(u2);
	}

}

double* genRandDirectVect(config_s config)
{
	double* direct_vect = calloc(config.num_dim,config.vect_size_db);

	for (uint16_t dim_idx = 0; dim_idx < config.num_dim; dim_idx++) {	genRandGaussArray(0.0, 1.0, 1, &direct_vect[dim_idx]); }
	
	double* unit_vect = calcUnitVect(config, direct_vect);
	free(direct_vect);
	
	return unit_vect;
}

void _calcFFT(double complex* input, double complex* buf, int n, int step)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Fourier transform sub function (recursive)
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	if (step >= n) return;
	_calcFFT(buf, input, n, step*2);
	_calcFFT(buf+step, input+step, n, step*2);

	#pragma omp parallel
	for (int k = 0; k < n; k += 2*step)
	{
		double complex w = cexpf(-I*M_PI*k/n);
		input[k/2] 	= buf[k] + w*buf[k+step];
		input[(k+n)/2] = buf[k] - w*buf[k+step];
	}
}

void calcFFT(double complex* input, double complex* output, int n)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Computes the fourier transform of inputted array
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	double complex* buf = malloc(sizeof(double complex)*n);
	for (int i = 0; i < n; ++i) 
	{
		buf[i] = input[i];
		output[i] = input[i];
	}
	_calcFFT(output, buf, n, 1);

	free(buf);
}

void _calcIFFT(double complex* input, double complex* buf, int n, int step)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Fourier transform sub function (recursive)
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	if (step >= n) return;
	_calcFFT(buf, input, n, step*2);
	_calcFFT(buf+step, input+step, n, step*2);

	#pragma omp parallel
	for (int k = 0; k < n; k += 2*step)
	{
		double complex w = cexpf(I*M_PI*k/n);
		input[k/2] 	= buf[k] + w*buf[k+step];
		input[(k+n)/2] = buf[k] - w*buf[k+step];
	}
}

void calcIFFT(double complex* input, double complex* output, int n)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Computes the fourier transform of inputted array
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	double complex* buf = malloc(sizeof(double complex)*n);
	for (int i = 0; i < n; ++i) 
	{
		buf[i] = input[i];
		output[i] = input[i];
	}
	_calcIFFT(output, buf, n, 1);

	for (int i = 0; i < n; ++i) 
	{
		output[i] *= 1.0/(2.0*M_PI);
	}

	free(buf);
}

double calcRelTime(config_s config, double* wave_direct, double* detect_pos, double speed)
{
	double num = 0; double den = 0;
	#pragma omp parallel
	for (size_t dim_idx = 0; dim_idx < config.num_dim; dim_idx++){
		num += wave_direct[dim_idx]*detect_pos[dim_idx];
		den += speed*wave_direct[dim_idx]*wave_direct[dim_idx];
	}

	return num/den;
}

void* calcConvFFT(double _Complex* f, double _Complex* g, double _Complex* conv, uint32_t len){
	
	double _Complex* f_freq = malloc(sizeof(double _Complex) * len);

	for (uint32_t freq_idx = 0; freq_idx < len; freq_idx++){
		f_freq[freq_idx] = f[freq_idx]*g[freq_idx];
	}

	calcFFT(f_freq, conv, len);

	free(f_freq); 
}

void calcPSD(config_s config, double _Complex* input, double _Complex* window_func_fft, uint32_t input_length, uint32_t num_segments, uint32_t seg_length, double** ret_psd, double** ret_freq_axis){
	

	//Seg length needs to be power of 2. Overlap inverval cannot be larger that seg_div

	if (checkPower2((size_t) seg_length) != 1) { seg_length = nearestPower2( (int) seg_length);
		printf("Seg length not power of 2, ajusting to nearest! Setting as %i. \n", seg_length);

	}

	uint32_t seg_div = (input_length - seg_length)/num_segments;
	uint32_t overlap_interval = ceil((seg_length - seg_div)/2.0);

	if ((num_segments > input_length - seg_length)){ printf("Too many segments. Exiting. \n"); exit(2);}
	double* freq_axis = malloc(sizeof(double)*seg_length);
	double* psd = calloc(seg_length, sizeof(double)*seg_length);

	for (uint32_t freq_idx = 0; freq_idx < seg_length; freq_idx++){
		freq_axis[freq_idx] = freq_idx*(config.stream_sample_rate/seg_length);
	}

	double _Complex* seg_conv = calloc(seg_length, sizeof(double _Complex) * seg_length);

	for (uint32_t seg_idx = 0; seg_idx < num_segments; seg_idx++){
		calcConvFFT(&input[seg_idx*seg_div], window_func_fft, seg_conv, seg_length);
		
		for (uint32_t freq_idx = 0; freq_idx < seg_length; freq_idx++){
			psd[freq_idx] += (double) sqrt(creal(seg_conv[freq_idx])*creal(seg_conv[freq_idx]) + cimag(seg_conv[freq_idx])*cimag(seg_conv[freq_idx]));
		}

	}

	free(seg_conv);

	for (uint32_t freq_idx = 0; freq_idx < seg_length; freq_idx++){
		psd[freq_idx] /= (double) num_segments;

	}

	*ret_psd = psd;
	*ret_freq_axis = freq_axis; 

}

void interpArray(double* orig_arr_x, double* orig_arr_y, uint32_t orig_arr_len, double* interped_arr_x, double* interped_arr_y, uint32_t interped_arr_len){

		//Interpolation:
		for (size_t time_1_idx = 0; time_1_idx < interped_arr_len; time_1_idx++){
			for (size_t time_2_idx = 0; time_2_idx < orig_arr_len; time_2_idx++){
				if ((interped_arr_x[time_1_idx] >= orig_arr_x[time_2_idx]) && (interped_arr_x[time_1_idx] <= orig_arr_x[time_2_idx + 1])){
					double grad = (orig_arr_x[time_2_idx + 1] - orig_arr_x[time_2_idx])/(orig_arr_y[time_2_idx + 1] - orig_arr_y[time_2_idx]);
					interped_arr_y[time_1_idx] = grad*(orig_arr_x[time_2_idx] - interped_arr_x[time_1_idx]) + orig_arr_y[time_2_idx];
				}
				else if(interped_arr_x[time_1_idx] >= orig_arr_x[orig_arr_len - 1]){
					interped_arr_y[time_1_idx] = orig_arr_y[orig_arr_len - 1];
				}
				else if(interped_arr_x[time_1_idx] <= orig_arr_x[0]){
					interped_arr_y[time_1_idx] = orig_arr_y[0];
				}
			}

		}


}

void calcMax(double* array, uint32_t arr_len, double* max, uint32_t* max_idx)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Calculates the maximum value of  an array.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	double temp_max = 0.0;
	uint32_t temp_max_idx = 0;

	for (uint32_t line_idx = 0; line_idx < arr_len; ++line_idx)
	{
		if (array[line_idx] > temp_max) 
		{ 
				temp_max = array[line_idx]; 
				temp_max_idx = line_idx;
		}
	}

	*max = temp_max;
	*max_idx = temp_max_idx;

}


double calcSNR(config_s config, double* strain_axis, double* strain_axis_sig, double* strain_axis_noise){
	
	double _Complex * strain_axis_c = malloc(sizeof(double _Complex) * config.stream_res);
	double _Complex * strain_axis_sig_c = malloc(sizeof(double _Complex) * config.stream_res);
	double _Complex * strain_axis_noise_c = malloc(sizeof(double _Complex) * config.stream_res);

	double _Complex * strain_axis_cf = malloc(sizeof(double _Complex) * config.stream_res);
	double _Complex * strain_axis_sig_cf = malloc(sizeof(double _Complex) * config.stream_res);

	double* freq_axis_c = malloc(sizeof(double) * config.stream_res);

	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		strain_axis_c[time_idx] = (double _Complex) strain_axis[time_idx];
		strain_axis_sig_c[time_idx] = (double _Complex) strain_axis_sig[time_idx];
		strain_axis_noise_c[time_idx] = (double _Complex) strain_axis_noise[time_idx];
 		freq_axis_c[time_idx] = time_idx*(config.stream_sample_rate/config.stream_res);

	}

	calcFFT(strain_axis_c, strain_axis_cf, config.stream_res);
	calcFFT(strain_axis_sig_c, strain_axis_sig_cf, config.stream_res);

	free(strain_axis_c); free(strain_axis_sig_c);

	double* psd; double* freq_axis;

	calcPSD(config, strain_axis_noise_c, config.window_func, config.stream_res, floor(config.stream_res/50), config.seg_length, &psd, &freq_axis); 

	free(strain_axis_noise_c);

	double* interped_psd = calloc(config.stream_res, sizeof(double) * config.stream_res);
	interpArray(freq_axis, psd, config.seg_length, freq_axis_c, interped_psd, config.stream_res);
	free(psd); free(freq_axis);

	double _Complex* optimal_cf = malloc(sizeof(double _Complex) * config.stream_res);
	
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		optimal_cf[time_idx] = (strain_axis_cf[time_idx]*conj(strain_axis_sig_cf[time_idx]))/interped_psd[time_idx];
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
		if (interped_psd[time_idx] != 0){
			sigma += ((2*strain_axis_sig_cf[time_idx]*conj(strain_axis_sig_cf[time_idx]))/interped_psd[time_idx])*freq_step;
		}
	}	

	free(strain_axis_sig_cf); free(interped_psd);

	sigma = sqrt(fabs(sigma)); 

	double* snr = malloc(sizeof(double _Complex) * config.stream_res);

	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		snr[time_idx] = optimal_c[time_idx]/sigma;
	}	

	double max; uint32_t max_idx;
	calcMax(snr, config.stream_res, &max, &max_idx);

	return max;

}

double clip(double num, double min, double max)
{
	
	if ( num < min ) { return min; }
	else if ( num > max) { return max; }
	
	return num;
}

double calcAngBetween(config_s config, double* vect_1, double* vect_2)
{
	double* vect_1_u = calcUnitVect(config, vect_1); double* vect_2_u = calcUnitVect(config, vect_2);

	double output = acos(clip(calcVectDot(config, vect_1_u, vect_2_u), -1.0, 1.0));
	free(vect_1_u); free(vect_2_u);

	return output; 
}

double calcAngBetweeninPlane(config_s config, double* vect_1, double* vect_2, double* normal_vect)
{
	double* vect_3 = calcVectProjectPlane(config, vect_1, normal_vect);
	double output = calcAngBetween(config, vect_2, vect_3);
	free(vect_3);

	return output;
}

double _Complex* genChirpSig_h(double hrss, double tau, double f0, double alpha, double delta, double T0, double* time_axis, size_t len_time_axis)
{
	
	double _Complex* strain_axis = malloc(sizeof(double _Complex) * len_time_axis);

	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx < len_time_axis; time_idx++ ) {
		double _Complex tminusT0 = (time_axis[time_idx] - T0);
		strain_axis[time_idx] = hrss*(cexp( - cpow((1-I*alpha )*(tminusT0), 2)/(4*tau*tau) + I*2*M_PI*tminusT0*f0 + I*delta)/cpow((2*M_PI*tau*tau),0.25));
	}

	return strain_axis;

}

void genChirpSig(config_s config, double hrss, double tau, double f0, double alpha, double delta, double ciota, double T0, double* time_axis, double plus_polar, double* strain_axes)
{
	double _Complex* strain_axis = genChirpSig_h(hrss, tau, f0, alpha, delta, T0, time_axis, config.stream_res);
	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx <  config.stream_res; time_idx++ ) { 
		strain_axes[time_idx] = (0.5*(1 + ciota*ciota) * creal(strain_axis[time_idx]))*(plus_polar) + (ciota * cimag(strain_axis[time_idx]))*(1 - plus_polar);
	}

	free(strain_axis);
}

void genGaussSig(config_s config, double hrss, double tau, double T0, double* time_axis, double* strain_axes)
{
	double h_peak = hrss * pow(2/(M_PI*tau*tau),0.25);
	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx <  config.stream_res; time_idx++ ) { 
		double tminusT0 = (time_axis[time_idx] - T0);
		strain_axes[time_idx] = h_peak*exp(-(tminusT0*tminusT0)/(tau*tau));
	}

}

void genRingSig(config_s config, double h_peak, double tau, double f0, double delta, double ciota, double T0, double* time_axis, double plus_polar, double* strain_axes)
{
	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		double tminusT0 = (time_axis[time_idx] - T0);
		if (time_axis[time_idx] >= T0){
			double hp = (h_peak*0.5*(1 + ciota*ciota)*cos(2*M_PI*(tminusT0)*f0 + delta))*exp(-(tminusT0)/tau);
			double hc = (h_peak*ciota*sin(2*M_PI*(tminusT0)*f0 + delta))*exp(-(tminusT0)/tau);
			strain_axes[time_idx]  = plus_polar*hp + (1- plus_polar)*hc;
		}
		else{
			double hp = (h_peak*0.5*(1 + ciota*ciota)*cos(2*M_PI*(tminusT0)*f0 + delta))*exp((tminusT0)/(tau/10.));
			double hc = (h_peak*ciota*sin(2*M_PI*(tminusT0)*f0 + delta))*exp((tminusT0)/(tau/10.));
			strain_axes[time_idx] = plus_polar*hp + (1- plus_polar)*hc;
		} 

	}
}

void genSineSig(config_s config, double h_peak, double f0, double T0, double* time_axis, double* strain_axes)
{
	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		double tminusT0 = (time_axis[time_idx] - T0);
		if (fabs(tminusT0) < 1/(2*f0)){
			strain_axes[time_idx] = h_peak*sin(2*M_PI*tminusT0*f0);
		}
		else{
			strain_axes[time_idx] = 0.0;
		}

	}
}

void genStatSig(config_s config, double amp, double tau, double T0, double* time_axis, double* strain_axis)
{
	#pragma omp parallel
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		double tminusT0 = (time_axis[time_idx] - T0);
		strain_axis[time_idx] = 0;
		if (fabs(tminusT0) < 0.5*tau ){
			genRandGaussArray(0, amp, 1, &strain_axis[time_idx]);
		}
	}

}

void shiftSig(config_s config, double tau, double T0, double* strain_axis, double* centre_time_disps, uint16_t detect_idx){

	int inc_disp = round(centre_time_disps[detect_idx]*config.stream_sample_rate);
	double* new_strain_axis = calloc(config.stream_res, sizeof(double) * config.stream_res);
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		if ((time_idx + inc_disp >= 0) && (time_idx + inc_disp < config.stream_res)) {
			new_strain_axis[time_idx] = strain_axis[time_idx + inc_disp];
		}
	}
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		strain_axis[time_idx] = 0.0;
		strain_axis[time_idx] = new_strain_axis[time_idx];
	}
	free(new_strain_axis);
}

void sigSwitch(config_s config, uint16_t type, double amp, double tau, double f0, double alpha, double delta, double ciota, double mu, double* time_axis, double plus_polar, double* strain_axis, uint16_t detect_idx, double* centre_time_disps, double* orig_strain_axis)
{

	switch(type){
		case 0: genChirpSig(config, amp, tau, f0, alpha, delta, ciota, mu, time_axis, plus_polar, strain_axis); break;
		case 1: genGaussSig(config, amp, tau, mu, time_axis, strain_axis); break;
		case 2: genRingSig(config, amp, tau, f0, delta, ciota, mu, time_axis, plus_polar, strain_axis); break;
		case 3: genSineSig(config, amp, f0,  mu,  time_axis, strain_axis); break;
		case 4: { 
			if (detect_idx == 0) { 
				genStatSig(config, 2*amp, tau, mu, time_axis, strain_axis); break; 
			}
			else {
				for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
					strain_axis[time_idx] = orig_strain_axis[time_idx];
				}
				shiftSig(config, tau, mu, strain_axis, centre_time_disps, detect_idx);
			}
			if (detect_idx == config.num_detects - 1){
				shiftSig(config, tau, mu, strain_axis, centre_time_disps, 0);
			}
			break; 
		}	
	}

}

double calcDetectResponce(config_s config, double* x_arm_direct_vect, double* y_arm_direct_vect, double* wave_polar_vect, double* wave_direct_vect, double wave_amp, double detect_sensitivity)
{

	//Add rotataion due to realtive times.

	double x_arm_orient_responce = sin(calcAngBetween(config, x_arm_direct_vect, wave_direct_vect));
	double y_arm_orient_responce = sin(calcAngBetween(config, y_arm_direct_vect, wave_direct_vect));

	double orient_responce = (x_arm_orient_responce + y_arm_orient_responce)/2.0;

	double x_arm_polar_responce = cos(2*calcAngBetweeninPlane(config, x_arm_direct_vect, wave_polar_vect, wave_direct_vect));
	double y_arm_polar_responce = cos(2*calcAngBetweeninPlane(config, y_arm_direct_vect, wave_polar_vect, wave_direct_vect));

	x_arm_polar_responce *= x_arm_polar_responce; y_arm_polar_responce *= y_arm_polar_responce;

	double polar_responce = (x_arm_polar_responce + y_arm_polar_responce)/2.0;

	return polar_responce*orient_responce*wave_amp*detect_sensitivity;

}


double calcCiota(config_s config, detect_s detect)
{

	double* rand_vect = genRandDirectVect(config);
	double output = calcAngBetween(config, detect.up_direct, rand_vect);
	
	free(rand_vect);

	return output;
}

double calcPolarResponce(config_s config, double* x_arm_direct_vect, double* y_arm_direct_vect, double* wave_polar_vect, double* wave_direct_vect)
{

	double x_arm_polar_responce = cos(2*calcAngBetweeninPlane(config, x_arm_direct_vect, wave_polar_vect, wave_direct_vect));
	double y_arm_polar_responce = cos(2*calcAngBetweeninPlane(config, y_arm_direct_vect, wave_polar_vect, wave_direct_vect));

	x_arm_polar_responce *= x_arm_polar_responce; y_arm_polar_responce *= y_arm_polar_responce;

	double polar_responce = (x_arm_polar_responce + y_arm_polar_responce)/2.0;

	return polar_responce;

}

void calcDetectPos(config_s config, detect_s* detects, detect_s** ret_detects)
{

	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++) {
		detects[detect_idx].pos_sphere = malloc(config.vect_size_db);

		detects[detect_idx].pos_sphere[1] = convDMStoDeg(detects[detect_idx].latt_deg, detects[detect_idx].latt_min, detects[detect_idx].latt_sec);
		detects[detect_idx].pos_sphere[1] = convLatttoSphere(detects[detect_idx].pos_sphere[1]);
		detects[detect_idx].pos_sphere[1] = convDegtoRad(detects[detect_idx].pos_sphere[1]);

		detects[detect_idx].pos_sphere[2] = convDMStoDeg(detects[detect_idx].long_deg, detects[detect_idx].long_min, detects[detect_idx].long_sec);
		detects[detect_idx].pos_sphere[2] = convDegtoRad(detects[detect_idx].pos_sphere[2]);

		detects[detect_idx].pos_sphere[0] = calcSpheroidRad(detects[detect_idx].pos_sphere[0], config.equat_rad, config.pole_rad) + detects[detect_idx].hasl;
		
		detects[detect_idx].pos = convSpheretoCart(config, detects[detect_idx].pos_sphere);

		detects[detect_idx].up_direct = calcRadUnitVect(config, detects[detect_idx].pos_sphere);

		detects[detect_idx].x_arm_direct_vect = calcThetaUnitVect(config, detects[detect_idx].pos_sphere); detects[detect_idx].x_arm_direct_vect[1] *= -1; 
		detects[detect_idx].x_arm_direct_vect = calcRotatedVector(config, detects[detect_idx].x_arm_direct_vect, config.origin, detects[detect_idx].up_direct, detects[detect_idx].x_arm_bear);

		detects[detect_idx].y_arm_direct_vect = calcThetaUnitVect(config, detects[detect_idx].pos_sphere); detects[detect_idx].y_arm_direct_vect[1] *= -1; 
		detects[detect_idx].y_arm_direct_vect = calcRotatedVector(config, detects[detect_idx].y_arm_direct_vect, config.origin, detects[detect_idx].up_direct, detects[detect_idx].y_arm_bear);
	}

	*ret_detects = detects;
}

void constructFilePath(char** ret_file_path)
{

	time_t t = time(NULL);
	struct tm *tm = localtime(&t);

	char* file_path = malloc(sizeof(char)*500);

	sprintf(file_path, "./models/%d-%d-%d_%d:%d:%d", tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
	struct stat st = {0};

	printf("%s\n", file_path);

	if (stat("./models", &st) == -1){
		mkdir("./models", 0700);
	}

	if (stat(file_path, &st) == -1){
		mkdir(file_path, 0700);
	}


	*ret_file_path = file_path;

}

void genHamming(size_t intervals, double** data_ret)
{
	double* data = malloc(sizeof(double)*intervals);
	for (size_t idx = 0; idx < intervals; ++idx)
	{
		data[idx] = 0.54 - 0.46*cos((2*M_PI*idx)/(intervals-1));
	}

	*data_ret = data;
}

void setupConfig(config_s config, int argc, char** argv, config_s* ret_config)
{
	
	readConfig("pseudo_grav_config.csv", &config);
	readInputArgs(argc, argv, config, &config);

	config.vect_size_db = config.num_dim*sizeof(double); config.vect_size_16 = config.num_dim*sizeof(uint16_t); config.vect_size_32 = config.num_dim*sizeof(uint32_t);
	config.origin = calloc(config.num_dim, config.vect_size_db);
		
	config.stream_res = config.stream_sample_rate*config.stream_duration; 

	if (checkPower2((size_t) config.stream_res) != 1) {
		config.stream_res = nearestPower2( (int) config.stream_res);
		printf("Stream resoloution not power of 2, ajusting to nearest! Setting as %i. \n", config.stream_res);

	}

	config.amp_step = (config.amp_max - config.amp_min)/config.num_gens;
	config.snr_cutoff_step = (config.snr_cutoff_max - config.snr_cutoff_min)/config.num_gens;
	config.amp_sigma_step = (config.amp_sigma_max - config.amp_sigma_min)/config.num_gens;

	config.time_axis = genLinArray(0.0, config.stream_duration, config.stream_res);

	constructFilePath(&config.file_path);

	double* wind_temp;
	genHamming(config.seg_length, &wind_temp);

	config.window_func = malloc(sizeof(double _Complex)*config.seg_length);

	for (uint32_t time_idx = 0; time_idx < config.seg_length; time_idx++){
		config.window_func[time_idx] = (double _Complex) wind_temp[time_idx];
	}

	free(wind_temp);

	config.window_func_fft = malloc(sizeof(double _Complex)*config.seg_length); 

	calcFFT(config.window_func, config.window_func_fft, config.seg_length);

	
	*ret_config = config;

}

void calcInterpedNoiseProfile(config_s config, detect_s* detects, detect_s** ret_detects)
{

	double* freq_axis = malloc(sizeof(double)*config.stream_res);
	double freq_step = 1.0/(config.time_axis[1] - config.time_axis[0]);
	
	for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
		freq_axis[time_idx] = time_idx*(config.stream_sample_rate/config.stream_res);

	}

	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		
		size_t lines_read = 0;
		
		char detect_path_name[strlen(detects[detect_idx].name) + strlen("./detect_noise_profiles/_noise_profile.csv") + 3];
		sprintf(detect_path_name, "./detect_noise_profiles/%s_noise_profile.csv", detects[detect_idx].name);
		readFileDouble(detect_path_name, ",", 2, 0, 0, &detects[detect_idx].noise_profile, &lines_read);

		detects[detect_idx].noise_profile_interp = calloc(config.stream_res, sizeof(double)*config.stream_res);

		//Interpolation:

		for (size_t time_1_idx = 0; time_1_idx < config.stream_res; time_1_idx++){
			for (size_t time_2_idx = 0; time_2_idx < 2*lines_read; time_2_idx += 2){
				if ((freq_axis[time_1_idx] >= detects[detect_idx].noise_profile[time_2_idx]) && (freq_axis[time_1_idx] <= detects[detect_idx].noise_profile[time_2_idx + 2])){
					double grad = (detects[detect_idx].noise_profile[time_2_idx + 1] - detects[detect_idx].noise_profile[time_2_idx + 3])/(detects[detect_idx].noise_profile[time_2_idx] - detects[detect_idx].noise_profile[time_2_idx + 2]);
					detects[detect_idx].noise_profile_interp[time_1_idx] = 	grad*(detects[detect_idx].noise_profile[time_2_idx] - freq_axis[time_1_idx]) + detects[detect_idx].noise_profile[time_2_idx + 1];
				}
			}

		}

		normaliseArray(detects[detect_idx].noise_profile_interp, config.stream_res, detects[detect_idx].noise_profile_interp);
	}

	*ret_detects = detects;

	free(freq_axis);

}

config_s initConfig()
{
	
	config_s config;

	config.str_t = clock();

	return config;

}

stream_s* initStreams(config_s* config)
{
	
	stream_s* streams = malloc(sizeof(stream_s)*config->num_streams);

	return streams;
}



stream_s* setupStreams(config_s* config, stream_s* streams)
{

	double* num_glitches_present = malloc(sizeof(double)*config->num_streams*config->num_detects);
	genRandGaussArray(config->glitches_present_mu,config->glitches_present_sigma,config->num_streams*config->num_detects,num_glitches_present);

	config->num_glitches = 0;
	config->num_waves = 0;

	for (uint32_t stream_idx = 0; stream_idx < config->num_streams; stream_idx++){

		streams[stream_idx].num_waves_present = (uint16_t) round((double)rand_64()/(double)RAND_64_MAX);
		config->num_waves += (uint32_t) streams[stream_idx].num_waves_present;
		streams[stream_idx].num_glitches_present = malloc(sizeof(uint16_t)*config->num_detects);
		streams[stream_idx].snr = calloc(config->num_detects, sizeof(double)*config->num_detects);

		for (uint16_t detect_idx = 0; detect_idx < config->num_detects; detect_idx++){
			streams[stream_idx].num_glitches_present[detect_idx] = (uint16_t) abs(round(num_glitches_present[stream_idx*3 + detect_idx]));
			config->num_glitches += streams[stream_idx].num_glitches_present[detect_idx];
		}

		streams[stream_idx].strain_axes = calloc(config->stream_res*config->num_detects, sizeof(double)*config->stream_res*config->num_detects);
		streams[stream_idx].strain_axes_noise = calloc(config->stream_res*config->num_detects, sizeof(double)*config->stream_res*config->num_detects);

	}

	free(num_glitches_present);

	return streams;

}

void freeStreams_p(config_s config, stream_s* streams)
{

	for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){
		free(streams[stream_idx].strain_axes);
		free(streams[stream_idx].strain_axes_noise);
		free(streams[stream_idx].snr);
	}

}

detect_s* initDetects(config_s config)
{

	detect_s* detects = malloc(sizeof(detect_s)*config.num_detects);
	readDetectConfig("detect_config.csv", config, detects);

	printf("%i detectors loaded: ", config.num_detects);
	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		printf(" %s", detects[detect_idx].name);
		if (detect_idx != config.num_detects - 1) { printf(",");}
		else {printf(".\n");}
	}

	calcDetectPos(config, detects, &detects);

	printf("Altering noise amplitude by: %f.\n", config.noise_ajust);

	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		detects[detect_idx].noise_amp += config.noise_ajust;
	}

	calcInterpedNoiseProfile(config, detects, &detects);

	//Add method to print location data.
	printf("Locations: \n");
	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		printf("%s: ", detects[detect_idx].name);
		printf("%f, %f, %f: ", detects[detect_idx].pos_sphere[0], detects[detect_idx].pos_sphere[1], detects[detect_idx].pos_sphere[2]);
		printf("%f, %f, %f. \n", detects[detect_idx].pos[0], detects[detect_idx].pos[1], detects[detect_idx].pos[2]);
	}

	return detects;
}

wave_s* initWaves(config_s config, size_t num_waves)
{

	wave_s* waves = malloc(sizeof(wave_s)*num_waves);

	return waves;
}

wave_s* setupWaves(config_s config, wave_s* waves, size_t num_waves, bool fixed_type)
{

	uint16_t* wave_types = malloc(sizeof(uint16_t)*num_waves); 
	genRandUIntArray(0, config.num_wave_types, num_waves, wave_types);
	double* wave_amps = malloc(sizeof(double)*num_waves);
	genRandGaussArray(config.wave_amp_mu, config.wave_amp_sigma, num_waves, wave_amps); 
	double* wave_taus = malloc(sizeof(double)*num_waves);
	genRandGaussArray(config.wave_tau_mu, config.wave_tau_sigma, num_waves, wave_taus);
	double* wave_f0s = malloc(sizeof(double)*num_waves);
	genRandGaussArray(config.wave_f0_mu, config.wave_f0_sigma, num_waves, wave_f0s);
	double* wave_alphas = malloc(sizeof(double)*num_waves);
	genRandGaussArray(config.wave_alpha_mu, config.wave_alpha_sigma, num_waves, wave_alphas); 

	for (size_t wave_idx = 0; wave_idx < num_waves; wave_idx++){

		if (fixed_type == 0) { waves[wave_idx].type = wave_types[wave_idx]; }
		waves[wave_idx].amp = clip(wave_amps[wave_idx], config.wave_amp_min, config.wave_amp_max);
		waves[wave_idx].tau = clip(wave_taus[wave_idx], config.wave_tau_min, config.wave_tau_max);
		waves[wave_idx].f0 = clip(wave_f0s[wave_idx], config.wave_f0_min, config.wave_f0_max);
		waves[wave_idx].alpha = clip(wave_alphas[wave_idx], config.wave_alpha_min, config.wave_alpha_max);
		waves[wave_idx].delta = genRandBetween(0, 2*M_PI);
		
		waves[wave_idx].centre_time = genRandBetween(config.wave_centre_time_min, config.wave_centre_time_max);
		waves[wave_idx].direct_vect = genRandDirectVect(config);

		waves[wave_idx].polar_vect = genRandDirectVect(config); 
		waves[wave_idx].polar_vect = calcVectProjectPlane(config, waves[wave_idx].polar_vect, waves[wave_idx].direct_vect);
		waves[wave_idx].polar_vect = calcUnitVect(config, waves[wave_idx].polar_vect);

	}

	free(wave_amps); free(wave_taus); free(wave_f0s); free(wave_alphas); free(wave_types);

	return waves;

}

glitch_s* initGlitches(config_s config)
{

	glitch_s* glitches = malloc(sizeof(glitch_s)*config.num_glitches);

	return glitches;
}


glitch_s* setupGlitches(config_s config, glitch_s* glitches)
{
	
	uint16_t* glitch_types = malloc(sizeof(uint16_t)*config.num_glitches); 
	genRandUIntArray(0, config.num_glitch_types, config.num_glitches, glitch_types);
	double* glitch_amp = malloc(sizeof(double)*config.num_glitches);
	genRandGaussArray(config.glitch_amp_mu, config.glitch_amp_sigma, config.num_glitches, glitch_amp);
	double* glitch_tau = malloc(sizeof(double)*config.num_glitches);
	genRandGaussArray(config.glitch_tau_mu, config.glitch_tau_sigma, config.num_glitches, glitch_tau);
	double* glitch_f0s = malloc(sizeof(double)*config.num_glitches);
	genRandGaussArray(config.glitch_f0_mu, config.glitch_f0_sigma, config.num_glitches, glitch_f0s);
	double* glitch_alphas = malloc(sizeof(double)*config.num_glitches);
	genRandGaussArray(config.glitch_alpha_mu, config.glitch_alpha_sigma, config.num_glitches, glitch_alphas); 

	for (uint32_t glitch_idx = 0; glitch_idx < config.num_glitches; glitch_idx++){
		glitches[glitch_idx].type = glitch_types[glitch_idx];
		glitches[glitch_idx].amp = clip(glitch_amp[glitch_idx], config.glitch_amp_min, config.glitch_amp_max);
		glitches[glitch_idx].tau = clip(fabs(glitch_tau[glitch_idx]), config.glitch_tau_min, config.glitch_tau_max);
		glitches[glitch_idx].f0 = clip(glitch_f0s[glitch_idx], config.glitch_f0_min, config.glitch_f0_max);
		glitches[glitch_idx].alpha = clip(glitch_alphas[glitch_idx], config.glitch_alpha_min, config.glitch_alpha_max);
		glitches[glitch_idx].delta = genRandBetween(0, 2*M_PI);
		
		glitches[glitch_idx].mu = genRandBetween(config.glitch_time_min, config.glitch_time_max);

	}

	free(glitch_types); free(glitch_amp); free(glitch_tau); free(glitch_f0s); free(glitch_alphas);

	return glitches;
}

void genNoise(config_s config, uint32_t stream_idx, stream_s* streams, detect_s* detects, stream_s** ret_streams)
{

	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){

		double* phaze_noise = calloc(config.stream_res, sizeof(double) * config.stream_res);
		double* gaus_noise = calloc(config.stream_res, sizeof(double) * config.stream_res);

		genRandGaussArray(config.noise_amp_mu, config.noise_amp_sigma, config.stream_res, gaus_noise);
		genRandArray(-2.0*M_PI, 2.0*M_PI, config.stream_res, phaze_noise);

		double _Complex* complex_noise = malloc(sizeof(double _Complex)*config.stream_res);

		#pragma omp parallel
		for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
			gaus_noise[time_idx] *= detects[detect_idx].noise_profile_interp[time_idx]; 
			complex_noise[time_idx] = (double _Complex) gaus_noise[time_idx] + I*(double _Complex) phaze_noise[time_idx];
		}

		free(phaze_noise);
		free(gaus_noise);

		calcIFFT(complex_noise, complex_noise, config.stream_res);

		#pragma omp parallel
		for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
			streams[stream_idx].strain_axes_noise[detect_idx*config.stream_res + time_idx] += (double) complex_noise[time_idx]*detects[detect_idx].noise_amp;
		}

		free(complex_noise);
	}

	*ret_streams = streams;
}

uint32_t genWaves(config_s config, uint32_t stream_idx, uint32_t wave_count, stream_s* streams, wave_s* waves, detect_s* detects, stream_s** ret_streams)
{
	if (streams[stream_idx].num_waves_present > 0) {
				
		streams[stream_idx].centre_time_disps = malloc(sizeof(double)*config.num_detects);
		streams[stream_idx].plus_polar = malloc(sizeof(double)*config.num_detects);
		streams[stream_idx].ciota = malloc(sizeof(double)*config.num_detects);
		streams[stream_idx].wave_resps = malloc(sizeof(double)*config.num_detects);

		bool* amp_greater = malloc(sizeof(bool)*config.num_detects);

		for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){

			streams[stream_idx].centre_time_disps[detect_idx] = calcRelTime(config, waves[wave_count].direct_vect, detects[detect_idx].pos, config.wave_speed);
			streams[stream_idx].wave_resps[detect_idx] = calcDetectResponce(config, detects[detect_idx].x_arm_direct_vect, detects[detect_idx].y_arm_direct_vect, waves[wave_count].polar_vect, waves[wave_count].direct_vect, waves[wave_count].amp, detects[detect_idx].sensitivity);

			streams[stream_idx].plus_polar[detect_idx] = calcPolarResponce(config, detects[detect_idx].x_arm_direct_vect, detects[detect_idx].y_arm_direct_vect, waves[wave_count].polar_vect, waves[wave_count].direct_vect);
			streams[stream_idx].ciota[detect_idx] = calcCiota(config, detects[detect_idx]);
		
			sigSwitch(config, waves[wave_count].type, waves[wave_count].amp, waves[wave_count].tau, waves[wave_count].f0, waves[wave_count].alpha, waves[wave_count].delta, streams[stream_idx].ciota[detect_idx], waves[wave_count].centre_time - streams[stream_idx].centre_time_disps[detect_idx], config.time_axis, streams[stream_idx].plus_polar[detect_idx], &streams[stream_idx].strain_axes[detect_idx*config.stream_res], detect_idx, streams[stream_idx].centre_time_disps, &streams[stream_idx].strain_axes[0]);

			double* strain_axis = calloc(config.stream_res, sizeof(double)*config.stream_res);

			for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
				streams[stream_idx].strain_axes[detect_idx*config.stream_res + time_idx] *= streams[stream_idx].wave_resps[detect_idx];
				strain_axis[time_idx] = streams[stream_idx].strain_axes[detect_idx*config.stream_res + time_idx] + streams[stream_idx].strain_axes_noise[detect_idx*config.stream_res + time_idx];
			}

			double amp_square_int = calcSqrInt(&streams[stream_idx].strain_axes[detect_idx*config.stream_res], config.stream_res);

			if (amp_square_int != 0){
				streams[stream_idx].snr[detect_idx] = calcSNR(config, strain_axis, &streams[stream_idx].strain_axes[detect_idx*config.stream_res], &streams[stream_idx].strain_axes_noise[detect_idx*config.stream_res ]);
			}

			free(strain_axis);

			if ((streams[stream_idx].snr[detect_idx] >= config.snr_cutoff)) { amp_greater[detect_idx] = 1;  }
			else { amp_greater[detect_idx] = 0;  }
		}


		if (calcbArrSum(amp_greater, config.num_detects) <= config.req_detects) { 

			waves[wave_count] = setupWaves(config, &waves[wave_count], 1, 1)[0]; 

			for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
				for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
					streams[stream_idx].strain_axes[detect_idx*config.stream_res + time_idx] = 0.0;
				}
			}

			free(amp_greater); 

			wave_count = genWaves(config, stream_idx, wave_count, streams,  waves, detects, &streams);

			*ret_streams = streams;
			return wave_count;

		}
		

		//#pragma omp parallel
		for (uint32_t time_idx = 0; time_idx < config.stream_res*config.num_detects; time_idx++){	
			streams[stream_idx].strain_axes_noise[time_idx] += streams[stream_idx].strain_axes[time_idx];

		}
		wave_count++;
		

		free(streams[stream_idx].centre_time_disps);
		free(streams[stream_idx].plus_polar);
		free(streams[stream_idx].ciota);
		free(streams[stream_idx].wave_resps);

	}

	*ret_streams = streams;

	return wave_count;
}

uint32_t genGlitches(config_s config, uint32_t stream_idx, uint32_t glitch_count, stream_s* streams, glitch_s* glitches, detect_s* detects, stream_s** ret_streams)
{
	uint32_t glitch_count_s = 0;
	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		if (streams[stream_idx].num_glitches_present[detect_idx] != 0){
					
			streams[stream_idx].glitches = malloc(sizeof(glitch_s)*calcArrSum(streams[stream_idx].num_glitches_present, config.num_detects));

			for (uint32_t glitch_idx = 0; glitch_idx < streams[stream_idx].num_glitches_present[detect_idx]; glitch_idx++){

				streams[stream_idx].glitches[glitch_count_s] = glitches[glitch_count]; 
				double* strain_axis_glitch = calloc(config.stream_res, sizeof(double)*config.stream_res);

				double arr[3] = {1,1,1};
				double ciota = cos(calcAngBetween(config, genRandDirectVect(config), arr));
				double plus_polar = genRandBetween(0,1);

				sigSwitch(config, glitches[glitch_count].type, glitches[glitch_count].amp, glitches[glitch_count].tau, glitches[glitch_count].f0, glitches[glitch_count].alpha, glitches[glitch_count].delta, ciota, glitches[glitch_count].mu, config.time_axis, plus_polar, strain_axis_glitch, 0, NULL, NULL);
				
				#pragma omp parallel		
				for (uint32_t time_idx = 0; time_idx < config.stream_res; time_idx++){
					streams[stream_idx].strain_axes_noise[detect_idx*config.stream_res + time_idx] += strain_axis_glitch[time_idx];
				}

				free(strain_axis_glitch);

				glitch_count++; glitch_count_s++;
			}
		
			free(streams[stream_idx].glitches);
			free(streams[stream_idx].num_glitches_present);
		}

	}

	*ret_streams = streams;

	return glitch_count;
}

void progressCheck(config_s config, uint32_t stream_idx, loading_s* loading)
{
	
	if (loading->chck_idx != 0) { 
		loading->end_t = clock();
		loading->loop_t = (double)(loading->end_t - loading->srt_t) / CLOCKS_PER_SEC;
		loading->total_t += loading->loop_t;
		loading->avg_t = loading->total_t/(config.prg_chck_int*loading->chck_idx);
		loading->time_to_cmplt = loading->avg_t*((double)config.num_streams - (double)stream_idx);
		printProgress(stream_idx, loading->time_to_cmplt, config.num_streams, config);
	}

	loading->srt_t = clock();

	loading->chck_idx++;
} 

void genStreams(config_s config, stream_s* streams, wave_s* waves, detect_s* detects, glitch_s* glitches, stream_s** ret_streams)
{

	bool prg_chck = 0;

	loading_s loading;

	loading.loop_t = 0; loading.total_t = 0; loading.avg_t = 0; 
	loading.chck_idx = 0; loading.time_to_cmplt = 0;

	size_t wave_count = 0;
	size_t glitch_count = 0;
	
	for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){

		if (stream_idx % config.prg_chck_int == 0 && stream_idx != 0){ prg_chck = 1;}

		if (config.gen_noise){
			genNoise(config, stream_idx, streams, detects, &streams);
		}

		if (config.gen_waves){
			wave_count = genWaves(config, stream_idx, wave_count, streams, waves, detects, &streams);
		}

		if (config.gen_glitches){
			glitch_count = genGlitches(config, stream_idx, glitch_count, streams, glitches, detects, &streams);
		}

		if (prg_chck){
			progressCheck(config, stream_idx, &loading);
			prg_chck = 0;
		}
	
	}

	printf("\n");

	*ret_streams = streams; 
}

void printdoubles(char* output_filename, int srt_line, int end_line, double* data, size_t num_cols, int print_number)
{
	int num_lines = (end_line - srt_line);

	if (output_filename != NULL)
	{

		char filename_positions[512] = ""; 
   		sprintf(filename_positions, "%s_test_%d.csv", output_filename, print_number);

		FILE* f = fopen(filename_positions, "w");
		for (size_t line_idx = srt_line; line_idx < (srt_line + num_lines); ++line_idx)
		{
			for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
			{
				fprintf(f, "%f", data[line_idx*num_cols + col_idx]);
			}

			fprintf(f, "\n");

		}

		fclose(f);
	}
	else printf("No output selected. Canceling printing. \n" );
}

void writeStreams(char* fn, config_s config, stream_s* streams, double* time_axis)
{
    FILE* f = fopen(fn, "wb");

    fwrite(&config.num_dim, sizeof(uint16_t), 1, f);
    fwrite(&config.equat_rad, sizeof(double), 1, f); 
    fwrite(&config.pole_rad, sizeof(double), 1, f);

    fwrite(&config.num_streams, sizeof(uint32_t), 1, f);
    fwrite(&config.stream_res, sizeof(uint32_t), 1, f);
    fwrite(&config.stream_duration, sizeof(double), 1, f);

    fwrite(&config.num_waves, sizeof(uint32_t), 1, f);
    fwrite(&config.wave_centre_time_min, sizeof(double), 1, f);
    fwrite(&config.wave_centre_time_max, sizeof(double), 1, f);
    fwrite(&config.wave_amp_min, sizeof(double), 1, f);
    fwrite(&config.wave_speed, sizeof(double), 1, f);

    fwrite(&config.num_detects, sizeof(uint16_t), 1, f);  
    fwrite(&config.snr_cutoff, sizeof(double), 1, f);

    for (uint32_t stream_idx = 0; stream_idx < config.num_streams; ++stream_idx){
    	fwrite(&streams[stream_idx].num_waves_present, sizeof(int16_t), 1, f);
    }

    fwrite(time_axis, sizeof(double), config.stream_res, f);

    for (uint32_t stream_idx = 0; stream_idx < config.num_streams; ++stream_idx){   
    	fwrite(streams[stream_idx].strain_axes_noise, sizeof(double), config.stream_res*config.num_detects, f);
    }

    for (uint32_t stream_idx = 0; stream_idx < config.num_streams; ++stream_idx){   
    	fwrite(streams[stream_idx].strain_axes, sizeof(double), config.stream_res*config.num_detects, f);
    }

    fclose(f);

}

void freeStreams(config_s config, stream_s* streams)
{
	for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){
		free(streams[stream_idx].strain_axes);
		free(streams[stream_idx].strain_axes_noise);
	}
	free(streams);
}

void freeWaves(config_s config, wave_s* waves)
{
	for (size_t wave_idx = 0; wave_idx < config.num_waves; wave_idx++){
		free(waves[wave_idx].direct_vect);
		free(waves[wave_idx].polar_vect);
	}
	free(waves);
}

void freeDetects(config_s config, detect_s* detects)
{
	for (uint16_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){
		free(detects[detect_idx].name);
		free(detects[detect_idx].pos);
		free(detects[detect_idx].pos_sphere);
		free(detects[detect_idx].noise_profile);
		free(detects[detect_idx].noise_profile_interp);
		free(detects[detect_idx].x_arm_direct_vect);
		free(detects[detect_idx].y_arm_direct_vect);
		free(detects[detect_idx].up_direct);
	}
	free(detects);
}

void freeModel(model_s model){
	free(model.accuracy);	
	free(model.false_positives);
	free(model.missed_positives);
	free(model.wave_amp);
	free(model.avg_snr);

}

void plotPowerSpect(config_s config, stream_s* streams)
{
	
	printf("*~~~~~~~~~~~~~~~~~~~~~* Calculating Noise Power Spectrum *~~~~~~~~~~~~~~~~~~~~* \n");

		double* power_spect_real = calloc(config.stream_res, sizeof(double) * config.stream_res);

		double _Complex* strain_axes_c = malloc(sizeof(double _Complex) * config.stream_res);
		double _Complex* power_spect = malloc(sizeof(double _Complex) * config.stream_res);

		for (size_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){

			for (size_t time_idx = 0; time_idx < config.stream_res; time_idx++){
				strain_axes_c[time_idx] = (double _Complex) streams[stream_idx].strain_axes_noise[time_idx];
			}
			
			calcFFT(strain_axes_c, power_spect, config.stream_res);

			for (size_t time_idx = 0; time_idx < config.stream_res; time_idx++){
				power_spect_real[time_idx] += (double) creal(power_spect[time_idx]) * (double) creal(power_spect[time_idx]) + (double) cimag(power_spect[time_idx]) * (double) cimag(power_spect[time_idx]);
			}
		}


		for (size_t time_idx = 0; time_idx < config.stream_res; time_idx++){
			power_spect_real[time_idx] /= config.num_streams;
			power_spect_real[time_idx] = sqrt(power_spect_real[time_idx]);
		}

			
		FILE* f = fopen("power_spect.dat", "wb");
		fwrite(&config.stream_res, sizeof(uint32_t), 1, f);
		fwrite(power_spect_real, sizeof(double), config.stream_res, f);
		fwrite(config.time_axis, sizeof(double), config.stream_res, f);

		fclose(f);

		free(strain_axes_c);
		free(power_spect);
}

void plotSkyHist(config_s config, stream_s* streams)
{

		printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~* Printing Sky Histogram *~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

		double* wave_rel_times = malloc(sizeof(double)*config.num_waves);
		double* wave_angle = malloc(sizeof(double)*config.num_waves);

		uint16_t wave_count = 0;

		for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){
			if (streams[stream_idx].num_waves_present > 0){
				wave_rel_times[wave_count] = streams[stream_idx].centre_time_disps[0] - streams[stream_idx].centre_time_disps[1];
				wave_angle[wave_count] = 0;
				wave_count++;
			}

		}

		FILE* q = fopen("sky_hist.dat", "wb");
		fwrite(&config.num_waves, sizeof(uint32_t), 1, q);
		fwrite(wave_rel_times, sizeof(double), config.num_waves, q);
		fwrite(wave_angle, sizeof(double), config.num_waves, q);

		fclose(q);

		free(wave_rel_times);
		free(wave_angle);
}

PyObject* makeListd(double* array, size_t arr_size)
{
    PyObject* list = PyList_New(arr_size);
    for (size_t list_idx = 0; list_idx < arr_size; ++list_idx) {
        PyList_SetItem(list, list_idx, PyFloat_FromDouble(array[list_idx]));
    }
    return list;
}

PyObject* makeListu(uint16_t* array, size_t arr_size) 
{
    PyObject* list = PyList_New(arr_size);
    for (size_t list_idx = 0; list_idx < arr_size; ++list_idx) {
        PyList_SetItem(list, list_idx, PyLong_FromUnsignedLong((unsigned long) array[list_idx]));
    }
    return list;
}

PyObject** makeArgArr(config_s config, stream_s* streams, size_t gen_idx, PyObject** args) 
{

    	args[0] = PyLong_FromUnsignedLong((unsigned long) config.num_streams);
		args[1] = PyLong_FromUnsignedLong((unsigned long) config.num_detects);
		args[2] = PyLong_FromUnsignedLong((unsigned long) config.stream_res );
		args[3] = PyLong_FromUnsignedLong((unsigned long) gen_idx);
		args[4] = PyLong_FromUnsignedLong((unsigned long) config.num_epocs);
		args[5] = PyList_New(config.num_streams);
		args[6] = PyList_New(config.num_streams*config.num_detects*config.stream_res);
		args[7] = PyList_New(config.num_streams*config.num_detects*config.stream_res);
		args[8] = makeListd(config.time_axis, config.stream_res);
		args[9] = PyBytes_FromString(config.file_path);


		for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){
			PyList_SetItem(args[5] , stream_idx, PyLong_FromUnsignedLong((unsigned long) streams[stream_idx].num_waves_present));
				
			for (uint32_t time_idx = 0; time_idx < config.num_detects*config.stream_res; time_idx++){
				size_t idx = stream_idx*config.num_detects*config.stream_res + time_idx; 
				PyList_SetItem(args[6], idx, PyFloat_FromDouble(streams[stream_idx].strain_axes[time_idx]));
				PyList_SetItem(args[7], idx, PyFloat_FromDouble(streams[stream_idx].strain_axes_noise[time_idx]));
			} 
		}

    return args;
}

double* makeArr(PyObject* list, size_t arr_size) 
{
    double* array = malloc(sizeof(double)*arr_size);
    for (size_t list_idx = 0; list_idx < arr_size; ++list_idx) {
        array[list_idx] = PyFloat_AsDouble(PyList_GetItem(list, list_idx));
    }
    return array;
}

int runPythonFunc(uint16_t num_args, char* module_name, char* func_name, PyObject** args, double** output_arr, size_t output_len)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    /* Error checking of pName left out */

    pModule = PyImport_ImportModule(module_name);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule,  func_name);

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(num_args);
            for (i = 0; i < num_args; ++i) {
                pValue = args[i];
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                PyTuple_SetItem(pArgs, i, pValue);
            }

            pValue = PyObject_CallObject(pFunc, pArgs);

            *output_arr = makeArr(pValue, output_len);

            Py_DECREF(pArgs);
            if (pValue != NULL) {
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", func_name);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", module_name);
        return 1;
    }

    return 0;
}

void plot(double* x_arr, uint32_t x_arr_len, double* y_arr, uint32_t y_arr_len, char* x_label, char* y_label, char* file_name)
{
	if (x_arr_len != y_arr_len){ printf("Arrays must be equal len. Exiting.\n"); return;}

	uint16_t num_args = 5;
	uint16_t output_len = 1;

	PyObject** args = malloc(sizeof(PyObject*) * num_args);
	double* output_arr = malloc(sizeof(double)); 
	
	args[0] = makeListd(y_arr, y_arr_len);
	args[1] = makeListd(x_arr, x_arr_len);
	args[2] = PyBytes_FromString(x_label);
	args[3] = PyBytes_FromString(y_label);
	args[4] = PyBytes_FromString(file_name);

	runPythonFunc(num_args, "nnt_v5", "plotGraph", args, &output_arr, output_len);
}

void constructModel(config_s config, double* model_plan, size_t plan_length, uint16_t output_len, double** ret_output_arr)
{
	uint16_t num_args = 5;

	PyObject** args = malloc(sizeof(PyObject*) * num_args);
	double* output_arr = malloc(sizeof(double));
	
	args[0] = PyLong_FromUnsignedLong((unsigned long) config.stream_res);
	args[1] = PyLong_FromUnsignedLong((unsigned long) config.num_detects);
	args[2] = PyLong_FromUnsignedLong((unsigned long) config.num_classes);
	args[3] = makeListd(model_plan, plan_length);
	args[4] = PyBytes_FromString(config.file_path);

	runPythonFunc(num_args, "nnt_v5", "constructModel", args, &output_arr, output_len);

	Py_DECREF(args[0]); Py_DECREF(args[1]); Py_DECREF(args[2]); 

	*ret_output_arr = output_arr; 
}

void printModel(config_s config, model_s model)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads data from config file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	char filename_positions[512];
   	sprintf(filename_positions, "%s/model.csv", config.file_path);

	FILE* f = fopen(filename_positions, "w");

	fprintf(f, "num_conv_layers:,%u\n", model.num_conv_layers);
	fprintf(f, "\n");
	fprintf(f, "conv_kern_sizes_x:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%u,", model.conv_kern_sizes_x[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "conv_kern_sizes_y:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%u,", model.conv_kern_sizes_y[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "conv_num_filters:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%u,", model.conv_num_filters[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "conv_batch_norm_present:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%u,", model.conv_batch_norm_present[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "conv_dropout_present:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%u,", model.conv_dropout_present[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "conv_dropouts:,"); for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){ fprintf(f, "%f,",model.conv_dropouts[conv_idx]); }; fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "num_dense_layers:,%u\n", model.num_dense_layers);
	fprintf(f, "dense_num_output:,"); for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){ fprintf(f, "%u,", model.dense_num_outputs[dense_idx]); }; fprintf(f, "\n");
	fprintf(f, "dense_dropouts_present:,"); for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){ fprintf(f, "%u,", model.dense_dropouts_present[dense_idx]); }; fprintf(f, "\n");
	fprintf(f, "dense_num_output:,"); for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){ fprintf(f, "%f,", model.dense_dropouts[dense_idx]); }; fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "learning_rate:,%f\n", model.learning_rate);

	fflush(stdout);
	fclose(f);

}

model_s initModel(config_s config)
{
	model_s model;
	model.accuracy = malloc(sizeof(double)*config.num_gens);
	model.false_positives = malloc(sizeof(double)*config.num_gens);
	model.missed_positives = malloc(sizeof(double)*config.num_gens);
	model.wave_amp = malloc(sizeof(double)*config.num_gens);
	model.avg_snr = malloc(sizeof(double)*config.num_gens);
	model.snr_cutoff = malloc(sizeof(double)*config.num_gens);
	model.wave_amp_sigma = malloc(sizeof(double)*config.num_gens);

	readModel("model_config.csv", &config, &model);
	printModel(config, model);

	size_t plan_length = model.num_conv_layers*6  + model.num_dense_layers*3 + 3;
	double* model_plan = malloc(sizeof(double)*(plan_length));

	model_plan[0] = (double) model.num_conv_layers; size_t curr_idx = 1;
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] = (double) model.conv_kern_sizes_x[conv_idx];
		curr_idx++;
	}
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] = (double) model.conv_kern_sizes_y[conv_idx];
		curr_idx++; 
	}
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] = (double) model.conv_num_filters[conv_idx];
		curr_idx++; 
	}
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] = (double) model.conv_batch_norm_present[conv_idx];
		curr_idx++; 
	}
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] = (double) model.conv_dropout_present[conv_idx];
		curr_idx++;
	}
	for (size_t conv_idx = 0; conv_idx < model.num_conv_layers; conv_idx++){
		model_plan[curr_idx] =  model.conv_dropouts[conv_idx];
		curr_idx++;
	}
	model_plan[curr_idx] = model.num_dense_layers; curr_idx++;
	for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){
		model_plan[curr_idx] = (double) model.dense_num_outputs[dense_idx];
		curr_idx++; 
	}
	for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){
		model_plan[curr_idx] = (double) model.dense_dropouts_present[dense_idx];
		curr_idx++; 
	}
	for (size_t dense_idx = 0; dense_idx < model.num_dense_layers; dense_idx++){
		model_plan[curr_idx] =  model.dense_dropouts[dense_idx];
		curr_idx++;
	}
	model_plan[curr_idx] = model.learning_rate;

	double* output_arr;
	constructModel(config, model_plan, plan_length, 1, &output_arr);
	free(output_arr);

	return model;
}

void trainModel(config_s config, stream_s* streams, uint32_t gen_idx, model_s model){

		uint16_t output_len = 4;
		uint16_t num_args = 10; 
		PyObject** args = malloc(sizeof(PyObject*)* num_args);
		args = makeArgArr(config, streams, gen_idx, args);

		freeStreams_p(config, streams);

		double* output_arr = malloc(sizeof(double)*output_len);
		runPythonFunc(num_args, "nnt_v5", "main", args, &output_arr, output_len);

		Py_DECREF(args[5]); Py_DECREF(args[6]); Py_DECREF(args[7]); Py_DECREF(args[8]);

		uint16_t validation_set_size = (uint16_t) output_arr[0] + (uint16_t) output_arr[1] + (uint16_t) output_arr[2] + (uint16_t) output_arr[3];

		model.accuracy[gen_idx] = (output_arr[0] + output_arr[3])/(double) validation_set_size ;
		model.false_positives[gen_idx] = (output_arr[1]/(output_arr[3] + output_arr[1]));
		model.missed_positives[gen_idx] = (output_arr[2]/(output_arr[2] + output_arr[3]));

		free(output_arr);

}

void printOutput(config_s config, model_s model){
	char filename_positions[512];
   	sprintf(filename_positions, "%s/output.csv", config.file_path);

	FILE* f = fopen(filename_positions, "w");
	for (size_t gen_idx = 0; gen_idx < config.num_gens; ++gen_idx){
		fprintf(f, "%f,", model.accuracy[gen_idx]);
		fprintf(f, "%f,", model.false_positives[gen_idx]);
		fprintf(f, "%f", model.missed_positives[gen_idx]);
		fprintf(f, "%f", model.wave_amp[gen_idx]);
		fprintf(f, "%f", model.wave_amp_sigma[gen_idx]);
		fprintf(f, "%f", model.avg_snr[gen_idx]);
		fprintf(f, "%f", model.snr_cutoff[gen_idx]);

		fprintf(f, "\n");

	}

	fclose(f);

}

void printInfo(config_s config){
	
	char filename_positions[512] = ""; 
   	sprintf(filename_positions, "%s/info.txt", config.file_path);

	FILE* f = fopen(filename_positions, "w");
	fprintf(f, "%u,", config.num_epocs);
	fprintf(f, "%u,", config.num_gens);
	fprintf(f, "%u,", config.stream_res);
	fprintf(f, "%u,", config.num_streams);
	fprintf(f, "%f", config.amp_max);
	fprintf(f, "%f", config.amp_min);

	fclose(f);
		
}

void plotAccuracyGraph(config_s config, model_s model){
	PyObject** args = malloc(sizeof(PyObject*)*5);

	args[0] = makeListd(model.accuracy,config.num_gens);
	args[1] = makeListd(model.false_positives,config.num_gens);
	args[2] = makeListd(model.missed_positives,config.num_gens);
	args[3] = makeListd(model.avg_snr, config.num_gens);
	args[4] = PyBytes_FromString(config.file_path);

	double* output_arr;
	runPythonFunc(5, "nnt_v5", "plotAccuracyGraph", args, &output_arr, 1);
	free(args); free(output_arr);
}

void initPy(){
	Py_Initialize();

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");
}

int main(int argc, char** argv){

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* Program Start *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	printf("Running Pseudo Gravitational Wave Generator: v5. Created by Michael Norman. \n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~~* Initilizing Config *~~~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	config_s config = initConfig();

	config.bar_len = 40;

	config.num_dim = 3;
	config.equat_rad = 6378137.0;
	config.pole_rad = 6356752.3;

	config.num_streams = 10000;
	config.stream_sample_rate = 1000;
	config.stream_duration = 5.0;

	config.gen_waves = 1; 
	config.num_wave_types = 4;
	config.waves_present_mu = 0; config.waves_present_sigma = 0.7;
	config.wave_amp_mu = 0; config.wave_amp_sigma = 0.4;
	config.wave_centre_time_min = 1.0; 	config.wave_centre_time_max = 2.0; 
	config.wave_amp_min = 0.0; 	config.wave_amp_max = 2.0; 
	config.wave_speed = 299792458;

	config.num_detects = 3;	

	config.req_detects = 2; 
	config.snr_cutoff = 10;

	config.gen_glitches = 1;
	config.num_glitch_types = 3;
	config.glitches_present_mu = 0.0; config.glitches_present_sigma = 0.3;
	config.glitch_amp_mu = 0; config.glitch_amp_sigma = 0.4;
	config.glitch_time_min = 1.0; config.glitch_time_max = 2.0; 

	config.prg_chck_int = 10;
	config.noise_ajust = 0;

	config.num_classes = 2; 
	config.num_epocs = 4; 
	config.num_gens = 10;

	config.amp_max = 2, config.amp_min = 1; 
	config.snr_cutoff_max = 1, config.snr_cutoff_min = 0.1; 
	config.amp_sigma_max = 0.3, config.amp_sigma_min = 0.03;

	config.seg_length = 256*2;

	config.snr_max = 10;

	setupConfig(config, argc, argv, &config);
	printInfo(config);

	printf("Complete.\n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~* Initilizing Detectors *~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	detect_s* detects = initDetects(config);

	printf("Complete.\n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~* Initilizing Streams *~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	stream_s* streams = initStreams(&config); 

	printf("Complete.\n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~* Initilising Python Instance *~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	//system("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/"); <-- very important for some reason

	initPy();

	printf("Complete.\n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~* Initilizing Model *~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	model_s model = initModel(config);

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~~* Begin Training. *~~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	for (size_t gen_idx = 0; gen_idx < config.num_gens; gen_idx++){

		printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~~* Generation: %zu/%u *~~~~~~~~~~~~~~~~~~~~~~~~~~~* \n", gen_idx + 1, config.num_gens);

		printf("*~~~~~~~~~~~~* Initilizing Streams *~~~~~~~~~~~~~~* \n");

		config.wave_amp_mu = config.amp_max - gen_idx*config.amp_step;
		config.wave_amp_sigma = config.amp_sigma_max - gen_idx*config.amp_sigma_step;
		config.snr_cutoff = config.snr_cutoff_max - gen_idx*config.snr_cutoff_step;

		model.wave_amp[gen_idx] = config.wave_amp_mu; 
		model.wave_amp_sigma[gen_idx] = config.wave_amp_sigma; 
		model.snr_cutoff[gen_idx] = config.snr_cutoff; 

		printf("Wave amplitude: %f.\n", config.wave_amp_mu);
		
		streams = setupStreams(&config, streams);
		
		printf("Complete.\n");
		
		printf("*~~~~~~~~~~~~~* Initilizing Waves *~~~~~~~~~~~~~~* \n");

		wave_s* waves = initWaves(config, config.num_waves);
		waves = setupWaves(config, waves, config.num_waves, 0);

		double percent_waves = ((double) config.num_waves / (double) config.num_streams) * 100.;

		printf("Number of waves to be generated: %u, out of %u streams. %f%% of streams contain waves. \n", config.num_waves, config.num_streams, percent_waves);

		printf("Complete.\n");

		printf("*~~~~~~~~~~* Initilizing Glitches *~~~~~~~~~~~~~* \n");

		glitch_s* glitches = initGlitches(config);
		glitches = setupGlitches(config, glitches);

		printf("Number of glitches to be generated: %u. \n", config.num_glitches);

		printf("Complete.\n");

		printf("*~~~~~~~~~~~* Generating Streams *~~~~~~~~~~~~* \n");

		genStreams(config, streams, waves, detects, glitches, &streams);

		config.avg_snr = 0;
		for (uint32_t stream_idx = 0; stream_idx < config.num_streams; stream_idx++){
			for (uint32_t detect_idx = 0; detect_idx < config.num_detects; detect_idx++){

				if (streams[stream_idx].num_waves_present == 1){
					printf("%f \n", config.avg_snr);
					config.avg_snr += streams[stream_idx].snr[detect_idx];
				}
			}
		}

		config.avg_snr /= (double) (config.num_waves*config.num_detects);

		printf("Average SNR: %f \n", config.avg_snr);
		model.avg_snr[gen_idx] = config.avg_snr;

		freeWaves(config, waves);
		free(glitches);

		printf("Complete.\n");
		
		printf("*~~~~~~~~~~~~* Training Network *~~~~~~~~~~~~~~~* \n");

		trainModel(config, streams, gen_idx, model);

		printf("Accuracy: %f %%\n", model.accuracy[gen_idx]* 100.0);
		printf("False Positives: %f %%\n", model.false_positives[gen_idx]* 100.0);
		printf("Missed Positives: %f %%\n", model.missed_positives[gen_idx]* 100.0);

		printf("Complete.\n");

		printf("*~~~~~~~~~~~~~* Running Tests *~~~~~~~~~~~~~~* \n");

		if (((config.gen_glitches != 1) && (config.gen_waves != 1)) && (config.gen_noise == 1)){
			//plotPowerSpect(config, streams);
		}

		if (config.gen_waves){
			//plotSkyHist(config_s config, stream_s* streams)
		}

	}

	printf("*~~~~~~~~~~~~~* Plotting Graphs *~~~~~~~~~~~~~~* \n");

	plotAccuracyGraph(config, model);
	printOutput(config, model);

	printf("*~~~~~~~~~~~~~* Finalizing *~~~~~~~~~~~~~~* \n");

	freeModel(model);
	freeDetects(config, detects);
	//freeStreams(config, streams);

	free(config.origin);
	free(config.time_axis);

	printf("Program completed. Exiting. \n");

	printf("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* End Program *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~* \n");

	return 0;
}


/*

TO DO: 

- SNR standard deviation

-fix tests make them print every time

-Train/Create mode
-Amp max on arg
-Unify config args and stuff at some point
-General tidying
-Sort out functions
-Check through parameters
-Comments
-Some error handling -> Ie if req_detects > num_detects raise an error
looping infinitely if cutoff too high, when stream res is too high for noise profile
if times are outside duration


*/
