/*
 * run.c
 *
 *  Created on: Oct 12, 2016
 *      Author: moloch
 */

#include "model.h"

vector input_vec;
dense layer1, layer2;

int main() {

	// Builds the network.
	input_vec = instantiate_vector(32);
	layer1 = build_layer(32, 64);
	layer2 = build_layer(64, 32);
	layer1.next = &layer2;

	// Adds random weights for testing.
	fill_vec(&input_vec, 0);
	fill_mat(&layer1.weights, 0);
	fill_mat(&layer2.weights, -1);

	get_result(&input_vec, &layer1);

	return 0;
}
