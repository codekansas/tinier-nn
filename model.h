/*
 * model.h
 *
 *  Created on: Oct 12, 2016
 *      Author: moloch
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
	dense result;
	result.weights = instantiate_matrix(num_input, num_output);
	result.outputs = instantiate_vector(num_output);
	result.next = NULL;
	return result;
}


vector* get_result(vector *input, dense *head) {
	while (head != NULL) {
		matmul(input, &head->outputs, &head->weights);
		input = &head->outputs;
		head = head->next;
	}
	return input;
}


#endif /* MODEL_H_ */
