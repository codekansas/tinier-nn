/*
 * run.c
 *
 * Sample program for loading a model and running it.
 *
 * tinier-nn/models/model.def (the sample definition) contains weights trained
 * for an XOR function. This file demonstrates how to initialize a model and
 * perform a classification task.
 *
 * To run normally, the model needs to be piped to stdin. From this directory,
 * run the following:
 *
 *   make run  # To build the file itself.
 *   cat ../models/model.def | ./run
 */

#include "model.h"

#define MAX_LAYERS 10

vector input_vec[4];
dense layers[MAX_LAYERS];

int main() {

	// Builds the network.
	int num_layers = load_model(layers, MAX_LAYERS);  // Load from stdin.
	printf("%d layers in model.\n", num_layers);

	// Run on XOR input vectors.
	for (int i = 0; i < 4; i++) {

		// Actually allocates space for the vector itself.
		instantiate_vector(&input_vec[i], 32);

		// Adds the XOR data.
		input_vec[i].data[0] = i << 30;

		// Runs the network and prints the output.
		vector *result = get_result(&input_vec[i], layers);
		print_vec(result);
	}

	return 0;
}
