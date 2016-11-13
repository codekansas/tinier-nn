/*
 * model.h
 *
 * Components associated with neural network architecture.
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "matrix.h"


typedef struct _dense {
	matrix weights;
	vector outputs;
	struct _dense *next;
} dense;


dense build_layer(dim_t num_input, dim_t num_output) {
	/* Builds a single feedforward dense layer.
	 *
	 * Args:
	 * 	 num_input:		number of input dimensions
	 * 	 num_output:	number of output dimensions
	 *
	 * Returns:
	 * 	 A dense binary layer which performs a linear transform from
	 * 	 num_input to num_output dimensions.
	 */

	dense result;
	result.weights = instantiate_matrix(num_input, num_output);
	result.outputs = instantiate_vector(num_output);
	result.next = NULL;
	return result;
}


vector* get_result(vector *input, dense *head) {
	/* Passes through every layer in the network and returns a pointer to the
	 * result.
	 *
	 * Args:
	 * 	 input:   pointer to the input vector (to be processed)
	 * 	 dense:   pointer to the first dense layer in a network.
	 *
	 * Returns:
	 * 	 A pointer to the vector containing the output data.
	 */

	while (head != NULL) {
		matmul(input, &head->outputs, &head->weights);
		input = &head->outputs;
		head = head->next;
	}
	return input;
}


dense load_model() {
	/* Loads a model into SRAM from a text file.
	 *
	 * Returns:
	 * 	 A dense struct containing the first layer in the network.
	 */

	if (next_char() != 'b' || next_char() != 'n' || next_char() != 'n') {
		log_str("Invalid magic string.");
		exit_failure();
	}

	int w = next_int(), h = next_int();
	dense input_layer = build_layer(w, h);
	dense *head = &input_layer;

	while (1) {

		// Reads in data for the current model.
		dim_t max_v = (head->weights.w * head->weights.h) / INT_SIZE;
		for (dim_t i = 0; i < max_v; i++) {
			data_t d = 0;
			dim_t j = 0;
			while (j < INT_SIZE) {
				char c = next_char();
				if (c != '0' && c != '1') {
					continue;
				}

				d <<= 1;
				if (c == '1') {
					d |= 1;
				}
				j++;
			}
			head->weights.data[i] = d;
		}

		// Reads in width and height;
		w = next_int();
		h = next_int();
		if (w == 0 || h == 0) break;

		// Adds another layer to the network.
		dense next_layer = build_layer(w, h);
		head->next = &next_layer;
		head = &next_layer;
	}

	return input_layer;
}


void load_vector(vector *input) {
	/* Reads inputs until a vector is full, then returns;
	 *
	 * Args:
	 * 	 input:    pointer to the vector to load.
	 */
	for (dim_t i = 0; i < input->h / INT_SIZE; i++) {
		data_t d = 0;
		dim_t j = 0;
		while (j < INT_SIZE) {
			char c = next_char();
			if (c != '0' && c != '1') {
				continue;
			}

			d <<= 1;
			if (c == '1') {
				d |= 1;
			}
			j++;
		}
		input->data[i] = d;
	}
}


#endif /* MODEL_H_ */
