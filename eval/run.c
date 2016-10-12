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

	vector *output = get_result(&input_vec, &layer1);
	print_vec(output);

	return 0;
}
