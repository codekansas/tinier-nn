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


void build_layer(dense *buffer, dim_t num_input, dim_t num_output) {
    /* Builds a single feedforward dense layer.
     *
     * Args:
     *   buffer:        pointer to location to hold data
     *      num_input:        number of input dimensions
     *      num_output:    number of output dimensions
     *
     * Returns:
     *      A dense binary layer which performs a linear transform from
     *      num_input to num_output dimensions.
     */

    instantiate_matrix(&buffer->weights, num_input, num_output);
    instantiate_vector(&buffer->outputs, num_output);
    buffer->next = NULL;
}


vector* get_result(vector *input, dense *head) {
    /* Passes through every layer in the network and returns a pointer to the
     * result.
     *
     * Args:
     *      input:   pointer to the input vector (to be processed)
     *      dense:   pointer to the first dense layer in a network.
     *
     * Returns:
     *      A pointer to the vector containing the output data.
     */

    while (head != NULL) {
        matmul(input, &head->outputs, &head->weights);
        input = &head->outputs;
        head = head->next;
    }
    return input;
}


int load_model(dense *buffer, unsigned max_layers) {
    /* Loads a model into SRAM from a text file.
     *
     * Returns:
     *      number of layers in the model (-1 if there was an error);
     */

    if (next_char() != 'b' || next_char() != 'n' || next_char() != 'n') {
        log_str("Invalid magic string.");
        exit_failure();
    }

    // Holds the dimensions of the layer.
    dim_t w, h;

    for (int layer = 0; layer < max_layers; layer++) {

        // Reads in width and height;
        get_dims(&w, &h);
        if (w == 0 || h == 0) {
            return layer;
        }

        // Instantiate layer itself.
        build_layer(&buffer[layer], w, h);

        // Connect previous layer to current layer.
        if (layer > 0) {
            buffer[layer-1].next = &buffer[layer];
        }

        // Reads in data for the current model.
        dim_t max_v = (buffer[layer].weights.w *
                       buffer[layer].weights.h) / INT_SIZE;
        for (dim_t i = 0; i < max_v; i++) {
            data_t d = 0;
            dim_t j = 0;
            while (j < INT_SIZE) {
                char c = next_char();
                if (c == '\n') {
                    continue;
                }

                d <<= 1;
                if (c == '1') {
                    d |= 1;
                }
                j++;
            }
            buffer[layer].weights.data[i] = d;
        }
    }

    return -1;
}


#endif /* MODEL_H_ */
