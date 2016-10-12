/*
 * utils.h
 *
 *  Created on: Oct 12, 2016
 *      Author: moloch
 *
 * When porting to Arduino, some values may have to be updated.
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>

// Defines types associated with matrices / vectors.
typedef unsigned int dim_t;
typedef unsigned int data_t;
typedef int bool_t;

// Different for other Arduinos.
#define INT_SIZE (sizeof(int) * 8)

// Failure exit (should show something to the user).
void exit_failure() {
	exit(EXIT_FAILURE);
}

void log_str(const char *x) {
	fprintf(stderr, "%s", x);
}

data_t* allocate_memory(dim_t n) {
	if (n % INT_SIZE != 0) {
		log_str("Invalid shape requested for memory allocation.\n");
		exit_failure();
	}
	return (data_t*) malloc(n / INT_SIZE);
}

#endif /* UTILS_H_ */
