/*
 * matrix.h
 *
 * Performs vector-matrix operations associated with the Binarized Neural
 * Network architecture.
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "utils.h"
#include <time.h>


typedef struct _matrix {
    /* Defines a matrix struct. */
    dim_t w, h;
    data_t *data;
} matrix;


typedef struct _vector {
    /* Defines a vector struct. */
    dim_t h;
    data_t *data;
} vector;


void instantiate_matrix(matrix *m, int w, int h) {
    /* Instantiates a matrix with width w and height h.
     *
     * This should only be called once; it provides the buffer,
     * and data is fed into it.
     */
    if (w % INT_SIZE != 0 || h % INT_SIZE != 0) {
        log_str("Invalid matrix size requested.\n");
        exit_failure();
    }

    m->w = w;
    m->h = h;
    m->data = allocate_memory(w * h * INT_SIZE);
}


void instantiate_vector(vector *v, int h) {
    /* Instantiates a vector with height h.
     *
     * This should only be called once; it provides the buffer,
     * and data is fed into it.
     */
    if (h % INT_SIZE != 0) {
        log_str("Invalid vector size requested.\n");
        exit_failure();
    }

    v->h = h;
    v->data = allocate_memory(h);
}


data_t bitsum(data_t x) {
    /* Calculates the bitsum of a data_t, e.g. the number of 1's.
     * This is useful for matrix multiplication.
     */
    data_t c = 0;
    for (int i = 0; i < INT_SIZE; i++) {
        c += (x & 1);
        x >>= 1;
    }
    return c;
}


void matmul(vector *from, vector *to, matrix *by) {
    /* Multiples binary vector "from" into "by" and puts the
     * result in "to". It is assumed that from.h == by.w
     * and to.h == by.h.
     *
     * The data in the matrix should be row-major, e.g. go through
     * all of 1 .. w each step of 1 .. h.
     */
    dim_t from_h = from->h / INT_SIZE, to_h = to->h / INT_SIZE;
    for (dim_t i = 0; i < to_h; i++) {
        data_t d = 0;
        for (dim_t j = 0; j < INT_SIZE; j++) {
            d <<= 1;
            data_t c = 0;
            for (dim_t k = 0; k < from_h; k++) {
                c += bitsum(by->data[((i * INT_SIZE) + j) * from_h + k] ^
                            from->data[k]);
            }

            // Threshold function.
            d |= (c >= from->h / 2) ? 0 : 1;
        }
        to->data[i] = d;
    }
}


void print_mat(matrix *m) {
    for (dim_t i = 0; i < m->h; i++) {
        for (dim_t j = 0; j < m->w; j++) {
            log_str((m->data[(i * m->w + j) / INT_SIZE] &
                    (1 << (j % INT_SIZE))) ? "1" : "0");
        }
        log_str("\n");
    }
}


void print_vec(vector *v) {
    for (dim_t i = 0; i < v->h; i++) {
        log_str((v->data[i / INT_SIZE] &
                (1 << (i % INT_SIZE))) ? "1" : "0");
    }
    log_str("\n");
}


#endif /* MATRIX_H_ */
