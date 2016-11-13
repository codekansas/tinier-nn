/*
 * run.c
 *
 * Sample program for loading a model and running it.
 */

#include "model.h"

vector input_vec;
dense head;

int main() {

	// Builds the network.
	input_vec = instantiate_vector(32);
	head = load_model();  // Reads from stdin.

	// Provide a sample input vector.
	// load_vector(&input_vec);
	// print_vec(&input_vec);

	// matmul(&input_vec, &head.outputs, &head.weights);
	// matmul(&head.outputs, &head.next->outputs, &head.next->weights);

	printf("Outputs:\n");
	printf("h = %d\n", head.next->outputs.h);
	printf("w = %d, h = %d\n", head.next->weights.w, head.next->weights.h);
	printf("Done\n");

	return 0;
}
