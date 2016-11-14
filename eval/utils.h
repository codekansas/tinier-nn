/*
 * utils.h
 *
 * When porting to a different embedded system, these values should be updated
 * for the particular compiler.
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdlib.h>

/*
 * Defines types associated with matrices / vectors.
 */
typedef unsigned int dim_t;
typedef unsigned int data_t;
typedef int bool_t;

/*
 * Defines the number of bits in a single int.
 */
#define INT_SIZE (sizeof(int) * 8)

/*
 * Failure exit. Uses a system call, so it should probably be generic in order
 * to work on other platforms.
 */
void exit_failure() {
	exit(EXIT_FAILURE);
}

/*
 * Generic operation for logging string information. It could just be ignored.
 */
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

/*
 * next_char and next_int define generic operations for reading and writing
 * from some data source, for loading models. These operations could be
 * changed to read from a serial port.
 */
char next_char() {
	return getchar();
}

void get_dims(dim_t *w, dim_t *h) {
	scanf("%d,%d", w, h);
}

#endif /* UTILS_H_ */
